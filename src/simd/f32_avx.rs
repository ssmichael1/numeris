//! AVX-accelerated f32 kernels for x86_64.
//!
//! AVX provides 256-bit registers → 8×f32 lanes.
//! Every kernel carries `#[target_feature(enable = "avx,fma")]`, so this module
//! compiles on any x86_64 target. The dispatcher in `super` calls into it only
//! when AVX and FMA are compile-time target features (`-C target-cpu=native` on
//! Haswell+) or, under the `runtime-dispatch` feature, when runtime detection
//! has confirmed the CPU supports both. All multiply-adds are fused (`fmadd`),
//! which is why the tier requires FMA — every AVX2 CPU has it; the AVX-only
//! Sandy / Ivy Bridge parts fall back to SSE2.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Dot product of two f32 slices using AVX.
///
/// Uses 4 independent accumulators (32 f32 per iteration) to hide
/// multiply-add latency.
#[inline]
#[target_feature(enable = "avx,fma")]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let (mut acc0, mut acc1, mut acc2, mut acc3) = (
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );

    // 4 accumulators × 8 lanes = 32 elements per iteration.
    let mut ai = a.chunks_exact(32);
    let mut bi = b.chunks_exact(32);
    for (ac, bc) in (&mut ai).zip(&mut bi) {
        // SAFETY: `chunks_exact(32)` yields chunks of exactly 32 `f32`, so the
        // four 8-lane loads at offsets 0, 8, 16 and 24 cover each chunk exactly.
        unsafe {
            let (ap, bp) = (ac.as_ptr(), bc.as_ptr());
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap), _mm256_loadu_ps(bp), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(8)), _mm256_loadu_ps(bp.add(8)), acc1);
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(ap.add(16)),
                _mm256_loadu_ps(bp.add(16)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(ap.add(24)),
                _mm256_loadu_ps(bp.add(24)),
                acc3,
            );
        }
    }

    let mut sum = {
        let s01 = _mm256_add_ps(acc0, acc1);
        let s23 = _mm256_add_ps(acc2, acc3);
        let s = _mm256_add_ps(s01, s23);
        // Horizontal sum: 8 lanes → 1
        let hi128 = _mm256_extractf128_ps(s, 1);
        let lo128 = _mm256_castps256_ps128(s);
        let sum128 = _mm_add_ps(hi128, lo128);
        let shuf = _mm_movehl_ps(sum128, sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 1);
        _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
    };

    // Remainder: up to 31 elements — 8-wide vectors first, then scalar.
    let mut acc_rem = _mm256_setzero_ps();
    let mut ar = ai.remainder().chunks_exact(8);
    let mut br = bi.remainder().chunks_exact(8);
    for (ac, bc) in (&mut ar).zip(&mut br) {
        // SAFETY: each chunk is exactly 8 `f32` — one vector load each.
        unsafe {
            acc_rem = _mm256_fmadd_ps(
                _mm256_loadu_ps(ac.as_ptr()),
                _mm256_loadu_ps(bc.as_ptr()),
                acc_rem,
            );
        }
    }
    sum += {
        let rhi = _mm256_extractf128_ps(acc_rem, 1);
        let rlo = _mm256_castps256_ps128(acc_rem);
        let rs = _mm_add_ps(rhi, rlo);
        let rs2 = _mm_movehl_ps(rs, rs);
        let rs3 = _mm_add_ps(rs, rs2);
        let rs4 = _mm_shuffle_ps(rs3, rs3, 1);
        _mm_cvtss_f32(_mm_add_ss(rs3, rs4))
    };

    for (&x, &y) in ar.remainder().iter().zip(br.remainder()) {
        sum += x * y;
    }
    sum
}

/// Matrix multiply C += A * B using AVX with register-blocked micro-kernel.
///
/// Uses an MR×NR (16×4) register-blocked micro-kernel that accumulates the full
/// k-sum in 8 AVX registers before writing back to C, reducing memory traffic
/// from O(m·n·p) to O(m·p) stores. Technique inspired by nano-gemm
/// (Sarah Quinones, <https://github.com/sarah-quinones/nano-gemm>).
///
/// `a` is m×n, `b` is n×p, `c` is m×p (column-major flat slices).
/// Column-major indexing: element (row, col) is at `col * nrows + row`.
///
/// # Panics
///
/// Panics unless `a.len() == m·n`, `b.len() == n·p` and `c.len() == m·p`.
/// The microkernels' `# Safety` bounds contracts assume these dimensions, so
/// they are checked in release builds, not just under `debug_assertions`.
#[inline]
#[target_feature(enable = "avx,fma")]
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, p: usize) {
    assert_eq!(a.len(), m * n, "matmul: a.len() != m*n");
    assert_eq!(b.len(), n * p, "matmul: b.len() != n*p");
    assert_eq!(c.len(), m * p, "matmul: c.len() != m*p");

    const MR: usize = 16; // 2 __m256 registers × 8 f32 lanes
    const NR: usize = 4;
    const KC: usize = 256;

    let m_full = (m / MR) * MR;
    let p_full = (p / NR) * NR;

    let mut kb = 0;
    while kb < n {
        let k_end = (kb + KC).min(n);

        // Interior: full MR×NR tiles, register-blocked
        for jb in 0..p_full / NR {
            let j0 = jb * NR;
            for ib in 0..m_full / MR {
                let i0 = ib * MR;
                // SAFETY: `i0 + 16 <= m_full <= m` and `j0 + 4 <= p_full <= p` by the
                // tile loops' construction, and `kb <= k_end <= n` — exactly the
                // microkernel's `# Safety` bounds contract.
                unsafe {
                    microkernel_16x4(a, b, c, m, n, i0, j0, kb, k_end);
                }
            }
        }

        // Bottom edge: cascade 8×4 → 4×4 → scalar
        let mut i0 = m_full;
        while i0 + 8 <= m {
            for jb in 0..p_full / NR {
                let j0 = jb * NR;
                // SAFETY: the loop condition guarantees `i0 + 8 <= m`, the `jb` loop
                // gives `j0 + 4 <= p_full <= p`, and `kb <= k_end <= n` — exactly the
                // microkernel's `# Safety` bounds contract.
                unsafe {
                    microkernel_8x4(a, b, c, m, n, i0, j0, kb, k_end);
                }
            }
            i0 += 8;
        }
        while i0 + 4 <= m {
            for jb in 0..p_full / NR {
                let j0 = jb * NR;
                // SAFETY: the loop condition guarantees `i0 + 4 <= m`, the `jb` loop
                // gives `j0 + 4 <= p_full <= p`, and `kb <= k_end <= n` — exactly the
                // microkernel's `# Safety` bounds contract.
                unsafe {
                    microkernel_4x4(a, b, c, m, n, i0, j0, kb, k_end);
                }
            }
            i0 += 4;
        }
        if i0 < m {
            for j in 0..p_full {
                for k in kb..k_end {
                    let b_kj = b[j * n + k];
                    for i in i0..m {
                        c[j * m + i] += a[k * m + i] * b_kj;
                    }
                }
            }
        }

        // Right edge: cols p_full..p, all rows (SIMD j-k-i on inner loop)
        let i_simd = m / 8;
        let i_tail = i_simd * 8;
        for j in p_full..p {
            for k in kb..k_end {
                let b_kj = b[j * n + k];
                let a_col = k * m;
                let c_col = j * m;
                // SAFETY: the broadcast touches no memory. Each iteration loads and
                // stores one 8-lane vector at `offset = i·8` with `i < i_simd = m / 8`,
                // so `offset + 8 <= m`: every access stays inside column `k` of `a`
                // (`a_col = k·m`, `k < n`) and column `j` of `c` (`c_col = j·m`, `j < p`).
                unsafe {
                    let vb = _mm256_set1_ps(b_kj);
                    for i in 0..i_simd {
                        let offset = i * 8;
                        let vc = _mm256_loadu_ps(c.as_ptr().add(c_col + offset));
                        let va = _mm256_loadu_ps(a.as_ptr().add(a_col + offset));
                        let result = _mm256_fmadd_ps(va, vb, vc);
                        _mm256_storeu_ps(c.as_mut_ptr().add(c_col + offset), result);
                    }
                }
                for i in i_tail..m {
                    c[c_col + i] += a[a_col + i] * b_kj;
                }
            }
        }

        kb += KC;
    }
}

/// Register-blocked 16×4 micro-kernel: accumulates C[i0..i0+16, j0..j0+4] in
/// 8 AVX registers across the k-loop, writing C only once.
///
/// # Safety
///
/// With `a` an `m×n`, `b` an `n×p` and `c` an `m×p` column-major matrix — so
/// `a.len() == m * n`, `b.len() == n * p` and `c.len() == m * p` — the caller
/// must guarantee that the tile and k-range lie inside them:
///
/// - `i0 + 16 <= m`, so the tile's 16 rows are within `a`'s and `c`'s columns;
/// - `j0 + 4 <= p`, so the tile's 4 columns are within `b` and `c`;
/// - `k_start <= k_end <= n`, so every `k` indexes a real column of `a` / row of `b`.
///
/// Every load and store below is then in bounds. AVX availability is guaranteed by the
/// caller's `#[target_feature(enable = "avx,fma")]` (this helper is always inlined into it).
#[inline(always)]
unsafe fn microkernel_16x4(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    i0: usize,
    j0: usize,
    k_start: usize,
    k_end: usize,
) {
    // SAFETY: the caller upholds the `# Safety` contract above, which puts
    // every pointer offset below in bounds of `a`, `b` and `c`; the
    // broadcasts and vector arithmetic touch no memory.
    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // 8 accumulator registers: 2 vectors × 4 columns
        let mut acc00 = _mm256_setzero_ps();
        let mut acc10 = _mm256_setzero_ps();
        let mut acc01 = _mm256_setzero_ps();
        let mut acc11 = _mm256_setzero_ps();
        let mut acc02 = _mm256_setzero_ps();
        let mut acc12 = _mm256_setzero_ps();
        let mut acc03 = _mm256_setzero_ps();
        let mut acc13 = _mm256_setzero_ps();

        for k in k_start..k_end {
            let a_off = k * m + i0;
            let a0 = _mm256_loadu_ps(a_ptr.add(a_off));
            let a1 = _mm256_loadu_ps(a_ptr.add(a_off + 8));

            let b0 = _mm256_set1_ps(*b_ptr.add(j0 * n + k));
            acc00 = _mm256_fmadd_ps(a0, b0, acc00);
            acc10 = _mm256_fmadd_ps(a1, b0, acc10);

            let b1 = _mm256_set1_ps(*b_ptr.add((j0 + 1) * n + k));
            acc01 = _mm256_fmadd_ps(a0, b1, acc01);
            acc11 = _mm256_fmadd_ps(a1, b1, acc11);

            let b2 = _mm256_set1_ps(*b_ptr.add((j0 + 2) * n + k));
            acc02 = _mm256_fmadd_ps(a0, b2, acc02);
            acc12 = _mm256_fmadd_ps(a1, b2, acc12);

            let b3 = _mm256_set1_ps(*b_ptr.add((j0 + 3) * n + k));
            acc03 = _mm256_fmadd_ps(a0, b3, acc03);
            acc13 = _mm256_fmadd_ps(a1, b3, acc13);
        }

        // Write back: C += acc
        let c_ptr = c.as_mut_ptr();

        let off0 = j0 * m + i0;
        _mm256_storeu_ps(
            c_ptr.add(off0),
            _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(off0)), acc00),
        );
        _mm256_storeu_ps(
            c_ptr.add(off0 + 8),
            _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(off0 + 8)), acc10),
        );

        let off1 = (j0 + 1) * m + i0;
        _mm256_storeu_ps(
            c_ptr.add(off1),
            _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(off1)), acc01),
        );
        _mm256_storeu_ps(
            c_ptr.add(off1 + 8),
            _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(off1 + 8)), acc11),
        );

        let off2 = (j0 + 2) * m + i0;
        _mm256_storeu_ps(
            c_ptr.add(off2),
            _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(off2)), acc02),
        );
        _mm256_storeu_ps(
            c_ptr.add(off2 + 8),
            _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(off2 + 8)), acc12),
        );

        let off3 = (j0 + 3) * m + i0;
        _mm256_storeu_ps(
            c_ptr.add(off3),
            _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(off3)), acc03),
        );
        _mm256_storeu_ps(
            c_ptr.add(off3 + 8),
            _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(off3 + 8)), acc13),
        );
    }
}

/// 8×4 mini-kernel (1 __m256 per column, 8 f32 rows).
///
/// # Safety
///
/// With `a` an `m×n`, `b` an `n×p` and `c` an `m×p` column-major matrix — so
/// `a.len() == m * n`, `b.len() == n * p` and `c.len() == m * p` — the caller
/// must guarantee that the tile and k-range lie inside them:
///
/// - `i0 + 8 <= m`, so the tile's 8 rows are within `a`'s and `c`'s columns;
/// - `j0 + 4 <= p`, so the tile's 4 columns are within `b` and `c`;
/// - `k_start <= k_end <= n`, so every `k` indexes a real column of `a` / row of `b`.
///
/// Every load and store below is then in bounds. AVX availability is guaranteed by the
/// caller's `#[target_feature(enable = "avx,fma")]` (this helper is always inlined into it).
#[inline(always)]
unsafe fn microkernel_8x4(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    i0: usize,
    j0: usize,
    k_start: usize,
    k_end: usize,
) {
    // SAFETY: the caller upholds the `# Safety` contract above, which puts
    // every pointer offset below in bounds of `a`, `b` and `c`; the
    // broadcasts and vector arithmetic touch no memory.
    unsafe {
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        for k in k_start..k_end {
            let av = _mm256_loadu_ps(ap.add(k * m + i0));
            a0 = _mm256_fmadd_ps(av, _mm256_set1_ps(*bp.add(j0 * n + k)), a0);
            a1 = _mm256_fmadd_ps(av, _mm256_set1_ps(*bp.add((j0 + 1) * n + k)), a1);
            a2 = _mm256_fmadd_ps(av, _mm256_set1_ps(*bp.add((j0 + 2) * n + k)), a2);
            a3 = _mm256_fmadd_ps(av, _mm256_set1_ps(*bp.add((j0 + 3) * n + k)), a3);
        }
        let cp = c.as_mut_ptr();
        for (j, acc) in [(j0, a0), (j0 + 1, a1), (j0 + 2, a2), (j0 + 3, a3)] {
            let off = j * m + i0;
            _mm256_storeu_ps(
                cp.add(off),
                _mm256_add_ps(_mm256_loadu_ps(cp.add(off)), acc),
            );
        }
    }
}

/// 4×4 mini-kernel (1 __m128 per column, 4 f32 rows).
///
/// # Safety
///
/// With `a` an `m×n`, `b` an `n×p` and `c` an `m×p` column-major matrix — so
/// `a.len() == m * n`, `b.len() == n * p` and `c.len() == m * p` — the caller
/// must guarantee that the tile and k-range lie inside them:
///
/// - `i0 + 4 <= m`, so the tile's 4 rows are within `a`'s and `c`'s columns;
/// - `j0 + 4 <= p`, so the tile's 4 columns are within `b` and `c`;
/// - `k_start <= k_end <= n`, so every `k` indexes a real column of `a` / row of `b`.
///
/// Every load and store below is then in bounds. AVX availability is guaranteed by the
/// caller's `#[target_feature(enable = "avx,fma")]` (this helper is always inlined into it).
#[inline(always)]
unsafe fn microkernel_4x4(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    i0: usize,
    j0: usize,
    k_start: usize,
    k_end: usize,
) {
    // SAFETY: the caller upholds the `# Safety` contract above, which puts
    // every pointer offset below in bounds of `a`, `b` and `c`; the
    // broadcasts and vector arithmetic touch no memory.
    unsafe {
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let mut a0 = _mm_setzero_ps();
        let mut a1 = _mm_setzero_ps();
        let mut a2 = _mm_setzero_ps();
        let mut a3 = _mm_setzero_ps();
        for k in k_start..k_end {
            let av = _mm_loadu_ps(ap.add(k * m + i0));
            a0 = _mm_fmadd_ps(av, _mm_set1_ps(*bp.add(j0 * n + k)), a0);
            a1 = _mm_fmadd_ps(av, _mm_set1_ps(*bp.add((j0 + 1) * n + k)), a1);
            a2 = _mm_fmadd_ps(av, _mm_set1_ps(*bp.add((j0 + 2) * n + k)), a2);
            a3 = _mm_fmadd_ps(av, _mm_set1_ps(*bp.add((j0 + 3) * n + k)), a3);
        }
        let cp = c.as_mut_ptr();
        for (j, acc) in [(j0, a0), (j0 + 1, a1), (j0 + 2, a2), (j0 + 3, a3)] {
            let off = j * m + i0;
            _mm_storeu_ps(cp.add(off), _mm_add_ps(_mm_loadu_ps(cp.add(off)), acc));
        }
    }
}

// ── Fused multiply-add, accumulator-first ──────────────────────────────────
//
// The shared `_fma` kernel macros in `super` were written against NEON's
// `vfmaq(acc, a, b)` (= acc + a·b) and `vfmsq(acc, a, b)` (= acc − a·b). Intel's
// `_mm256_fmadd_ps(a, b, c)` puts the accumulator last, so these adapters give the
// macros the NEON argument shape. Register-only; inlined into the attributed
// kernels.

#[inline]
#[target_feature(enable = "avx,fma")]
fn fmadd_acc(acc: __m256, a: __m256, b: __m256) -> __m256 {
    _mm256_fmadd_ps(a, b, acc)
}

#[inline]
#[target_feature(enable = "avx,fma")]
fn fnmadd_acc(acc: __m256, a: __m256, b: __m256) -> __m256 {
    _mm256_fnmadd_ps(a, b, acc)
}

// Element-wise add/sub/scale and AXPY kernels are generated from the shared
// macros in `super` (identical across ISAs bar width + intrinsic names).
simd_elementwise_kernels!(
    @feature "avx"
    f32,
    8,
    _mm256_loadu_ps,
    _mm256_storeu_ps,
    _mm256_add_ps,
    _mm256_sub_ps,
    _mm256_mul_ps,
    _mm256_set1_ps
);
simd_fft_butterfly_kernel!(
    @feature "avx"
    f32,
    8,
    _mm256_loadu_ps,
    _mm256_storeu_ps,
    _mm256_add_ps,
    _mm256_sub_ps,
    _mm256_mul_ps
);
simd_fft_butterfly4_kernel!(
    @feature "avx"
    f32,
    8,
    _mm256_loadu_ps,
    _mm256_storeu_ps,
    _mm256_add_ps,
    _mm256_sub_ps,
    _mm256_mul_ps
);
simd_axpy_kernels_fma!(
    @feature "avx,fma"
    f32,
    8,
    _mm256_loadu_ps,
    _mm256_storeu_ps,
    fmadd_acc,
    fnmadd_acc,
    _mm256_set1_ps
);
simd_conv1d_kernel_fma!(
    @feature "avx,fma"
    f32,
    8,
    _mm256_loadu_ps,
    _mm256_storeu_ps,
    fmadd_acc,
    _mm256_set1_ps
);
