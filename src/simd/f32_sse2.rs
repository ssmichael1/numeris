//! SSE2-accelerated f32 kernels for x86_64.
//!
//! SSE2 provides 128-bit registers → 4×f32 lanes.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Dot product of two f32 slices using SSE2.
///
/// Uses 4 independent accumulators (16 f32 per iteration) to hide
/// multiply-add latency.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    // SAFETY: register broadcasts of zero; they touch no memory.
    let (mut acc0, mut acc1, mut acc2, mut acc3) = unsafe {
        (
            _mm_setzero_ps(),
            _mm_setzero_ps(),
            _mm_setzero_ps(),
            _mm_setzero_ps(),
        )
    };

    // 4 accumulators × 4 lanes = 16 elements per iteration.
    let mut ai = a.chunks_exact(16);
    let mut bi = b.chunks_exact(16);
    for (ac, bc) in (&mut ai).zip(&mut bi) {
        // SAFETY: `chunks_exact(16)` yields chunks of exactly 16 `f32`, so the
        // four 4-lane loads at offsets 0, 4, 8 and 12 cover each chunk exactly.
        unsafe {
            let (ap, bp) = (ac.as_ptr(), bc.as_ptr());
            acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(ap), _mm_loadu_ps(bp)));
            acc1 = _mm_add_ps(
                acc1,
                _mm_mul_ps(_mm_loadu_ps(ap.add(4)), _mm_loadu_ps(bp.add(4))),
            );
            acc2 = _mm_add_ps(
                acc2,
                _mm_mul_ps(_mm_loadu_ps(ap.add(8)), _mm_loadu_ps(bp.add(8))),
            );
            acc3 = _mm_add_ps(
                acc3,
                _mm_mul_ps(_mm_loadu_ps(ap.add(12)), _mm_loadu_ps(bp.add(12))),
            );
        }
    }

    // SAFETY: register arithmetic only — no memory is touched.
    let mut sum = unsafe {
        let s01 = _mm_add_ps(acc0, acc1);
        let s23 = _mm_add_ps(acc2, acc3);
        let s = _mm_add_ps(s01, s23);
        // Horizontal sum of 4 lanes
        let shuf = _mm_movehl_ps(s, s);
        let sums = _mm_add_ps(s, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 1);
        _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
    };

    // Remainder: up to 15 elements — 4-wide vectors first, then scalar.
    // SAFETY: register broadcast of zero.
    let mut acc_rem = unsafe { _mm_setzero_ps() };
    let mut ar = ai.remainder().chunks_exact(4);
    let mut br = bi.remainder().chunks_exact(4);
    for (ac, bc) in (&mut ar).zip(&mut br) {
        // SAFETY: each chunk is exactly 4 `f32` — one vector load each.
        unsafe {
            acc_rem = _mm_add_ps(
                acc_rem,
                _mm_mul_ps(_mm_loadu_ps(ac.as_ptr()), _mm_loadu_ps(bc.as_ptr())),
            );
        }
    }
    // SAFETY: register arithmetic only.
    sum += unsafe {
        {
            let rs = _mm_movehl_ps(acc_rem, acc_rem);
            let rs2 = _mm_add_ps(acc_rem, rs);
            let rs3 = _mm_shuffle_ps(rs2, rs2, 1);
            _mm_cvtss_f32(_mm_add_ss(rs2, rs3))
        }
    };

    for (&x, &y) in ar.remainder().iter().zip(br.remainder()) {
        sum += x * y;
    }
    sum
}

/// Matrix multiply C += A * B using SSE2 with register-blocked micro-kernel.
///
/// Uses an MR×NR (8×4) register-blocked micro-kernel that accumulates the full
/// k-sum in 8 SSE2 registers before writing back to C, reducing memory traffic
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
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, p: usize) {
    assert_eq!(a.len(), m * n, "matmul: a.len() != m*n");
    assert_eq!(b.len(), n * p, "matmul: b.len() != n*p");
    assert_eq!(c.len(), m * p, "matmul: c.len() != m*p");

    const MR: usize = 8; // 2 __m128 vectors × 4 f32 lanes
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
                // SAFETY: `i0 + 8 <= m_full <= m` and `j0 + 4 <= p_full <= p` by the
                // tile loops' construction, and `kb <= k_end <= n` — exactly the
                // microkernel's `# Safety` bounds contract.
                unsafe {
                    microkernel_8x4(a, b, c, m, n, i0, j0, kb, k_end);
                }
            }
        }

        // Bottom edge: rows m_full..m, cols 0..p_full
        let mut i0 = m_full;
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
                    let a_col = k * m;
                    let c_col = j * m;
                    for i in i0..m {
                        c[c_col + i] += a[a_col + i] * b_kj;
                    }
                }
            }
        }

        // Right edge: cols p_full..p, all rows (SIMD j-k-i on inner loop)
        let i_simd = m / 4;
        let i_tail = i_simd * 4;
        for j in p_full..p {
            for k in kb..k_end {
                let b_kj = b[j * n + k];
                let a_col = k * m;
                let c_col = j * m;
                // SAFETY: the broadcast touches no memory. Each iteration loads and
                // stores one 4-lane vector at `offset = i·4` with `i < i_simd = m / 4`,
                // so `offset + 4 <= m`: every access stays inside column `k` of `a`
                // (`a_col = k·m`, `k < n`) and column `j` of `c` (`c_col = j·m`, `j < p`).
                unsafe {
                    let vb = _mm_set1_ps(b_kj);
                    for i in 0..i_simd {
                        let offset = i * 4;
                        let vc = _mm_loadu_ps(c.as_ptr().add(c_col + offset));
                        let va = _mm_loadu_ps(a.as_ptr().add(a_col + offset));
                        _mm_storeu_ps(
                            c.as_mut_ptr().add(c_col + offset),
                            _mm_add_ps(vc, _mm_mul_ps(va, vb)),
                        );
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

/// Register-blocked 8×4 micro-kernel: accumulates C[i0..i0+8, j0..j0+4] in
/// 8 SSE2 registers across the full k-loop, writing C only once.
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
/// Every load and store below is then in bounds. SSE2 is part of the `x86_64` baseline,
/// which the module's `#[cfg(target_arch = "x86_64")]` gate guarantees.
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
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // 8 accumulator registers: 2 vectors × 4 columns
        let mut acc00 = _mm_setzero_ps();
        let mut acc10 = _mm_setzero_ps();
        let mut acc01 = _mm_setzero_ps();
        let mut acc11 = _mm_setzero_ps();
        let mut acc02 = _mm_setzero_ps();
        let mut acc12 = _mm_setzero_ps();
        let mut acc03 = _mm_setzero_ps();
        let mut acc13 = _mm_setzero_ps();

        for k in k_start..k_end {
            let a_off = k * m + i0;
            let a0 = _mm_loadu_ps(a_ptr.add(a_off));
            let a1 = _mm_loadu_ps(a_ptr.add(a_off + 4));

            let b0 = _mm_set1_ps(*b_ptr.add(j0 * n + k));
            acc00 = _mm_add_ps(acc00, _mm_mul_ps(a0, b0));
            acc10 = _mm_add_ps(acc10, _mm_mul_ps(a1, b0));

            let b1 = _mm_set1_ps(*b_ptr.add((j0 + 1) * n + k));
            acc01 = _mm_add_ps(acc01, _mm_mul_ps(a0, b1));
            acc11 = _mm_add_ps(acc11, _mm_mul_ps(a1, b1));

            let b2 = _mm_set1_ps(*b_ptr.add((j0 + 2) * n + k));
            acc02 = _mm_add_ps(acc02, _mm_mul_ps(a0, b2));
            acc12 = _mm_add_ps(acc12, _mm_mul_ps(a1, b2));

            let b3 = _mm_set1_ps(*b_ptr.add((j0 + 3) * n + k));
            acc03 = _mm_add_ps(acc03, _mm_mul_ps(a0, b3));
            acc13 = _mm_add_ps(acc13, _mm_mul_ps(a1, b3));
        }

        // Write back: C += acc
        let c_ptr = c.as_mut_ptr();

        let off0 = j0 * m + i0;
        _mm_storeu_ps(
            c_ptr.add(off0),
            _mm_add_ps(_mm_loadu_ps(c_ptr.add(off0)), acc00),
        );
        _mm_storeu_ps(
            c_ptr.add(off0 + 4),
            _mm_add_ps(_mm_loadu_ps(c_ptr.add(off0 + 4)), acc10),
        );

        let off1 = (j0 + 1) * m + i0;
        _mm_storeu_ps(
            c_ptr.add(off1),
            _mm_add_ps(_mm_loadu_ps(c_ptr.add(off1)), acc01),
        );
        _mm_storeu_ps(
            c_ptr.add(off1 + 4),
            _mm_add_ps(_mm_loadu_ps(c_ptr.add(off1 + 4)), acc11),
        );

        let off2 = (j0 + 2) * m + i0;
        _mm_storeu_ps(
            c_ptr.add(off2),
            _mm_add_ps(_mm_loadu_ps(c_ptr.add(off2)), acc02),
        );
        _mm_storeu_ps(
            c_ptr.add(off2 + 4),
            _mm_add_ps(_mm_loadu_ps(c_ptr.add(off2 + 4)), acc12),
        );

        let off3 = (j0 + 3) * m + i0;
        _mm_storeu_ps(
            c_ptr.add(off3),
            _mm_add_ps(_mm_loadu_ps(c_ptr.add(off3)), acc03),
        );
        _mm_storeu_ps(
            c_ptr.add(off3 + 4),
            _mm_add_ps(_mm_loadu_ps(c_ptr.add(off3 + 4)), acc13),
        );
    }
}

/// Register-blocked 4×4 mini-kernel for bottom-edge rows (1 __m128 per col).
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
/// Every load and store below is then in bounds. SSE2 is part of the `x86_64` baseline,
/// which the module's `#[cfg(target_arch = "x86_64")]` gate guarantees.
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
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut acc0 = _mm_setzero_ps();
        let mut acc1 = _mm_setzero_ps();
        let mut acc2 = _mm_setzero_ps();
        let mut acc3 = _mm_setzero_ps();

        for k in k_start..k_end {
            let a0 = _mm_loadu_ps(a_ptr.add(k * m + i0));

            acc0 = _mm_add_ps(acc0, _mm_mul_ps(a0, _mm_set1_ps(*b_ptr.add(j0 * n + k))));
            acc1 = _mm_add_ps(
                acc1,
                _mm_mul_ps(a0, _mm_set1_ps(*b_ptr.add((j0 + 1) * n + k))),
            );
            acc2 = _mm_add_ps(
                acc2,
                _mm_mul_ps(a0, _mm_set1_ps(*b_ptr.add((j0 + 2) * n + k))),
            );
            acc3 = _mm_add_ps(
                acc3,
                _mm_mul_ps(a0, _mm_set1_ps(*b_ptr.add((j0 + 3) * n + k))),
            );
        }

        let c_ptr = c.as_mut_ptr();
        let off0 = j0 * m + i0;
        _mm_storeu_ps(
            c_ptr.add(off0),
            _mm_add_ps(_mm_loadu_ps(c_ptr.add(off0)), acc0),
        );
        let off1 = (j0 + 1) * m + i0;
        _mm_storeu_ps(
            c_ptr.add(off1),
            _mm_add_ps(_mm_loadu_ps(c_ptr.add(off1)), acc1),
        );
        let off2 = (j0 + 2) * m + i0;
        _mm_storeu_ps(
            c_ptr.add(off2),
            _mm_add_ps(_mm_loadu_ps(c_ptr.add(off2)), acc2),
        );
        let off3 = (j0 + 3) * m + i0;
        _mm_storeu_ps(
            c_ptr.add(off3),
            _mm_add_ps(_mm_loadu_ps(c_ptr.add(off3)), acc3),
        );
    }
}

// Element-wise add/sub/scale and AXPY kernels are generated from the shared
// macros in `super` (identical across ISAs bar width + intrinsic names).
simd_elementwise_kernels!(
    f32,
    4,
    _mm_loadu_ps,
    _mm_storeu_ps,
    _mm_add_ps,
    _mm_sub_ps,
    _mm_mul_ps,
    _mm_set1_ps
);
simd_fft_butterfly_kernel!(
    f32,
    4,
    _mm_loadu_ps,
    _mm_storeu_ps,
    _mm_add_ps,
    _mm_sub_ps,
    _mm_mul_ps
);
simd_axpy_kernels_muladd!(
    f32,
    4,
    _mm_loadu_ps,
    _mm_storeu_ps,
    _mm_add_ps,
    _mm_sub_ps,
    _mm_mul_ps,
    _mm_set1_ps
);
simd_conv1d_kernel_muladd!(
    f32,
    4,
    _mm_loadu_ps,
    _mm_storeu_ps,
    _mm_add_ps,
    _mm_mul_ps,
    _mm_set1_ps
);
