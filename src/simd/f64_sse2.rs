//! SSE2-accelerated f64 kernels for x86_64.
//!
//! SSE2 provides 128-bit registers → 2×f64 lanes.
//! SSE2 is baseline on x86_64 (always available).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Dot product of two f64 slices using SSE2.
///
/// Uses 4 independent accumulators (8 f64 per iteration) to hide
/// multiply-add latency.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    // SAFETY: register broadcasts of zero; they touch no memory.
    let (mut acc0, mut acc1, mut acc2, mut acc3) = unsafe {
        (
            _mm_setzero_pd(),
            _mm_setzero_pd(),
            _mm_setzero_pd(),
            _mm_setzero_pd(),
        )
    };

    // 4 accumulators × 2 lanes = 8 elements per iteration.
    let mut ai = a.chunks_exact(8);
    let mut bi = b.chunks_exact(8);
    for (ac, bc) in (&mut ai).zip(&mut bi) {
        // SAFETY: `chunks_exact(8)` yields chunks of exactly 8 `f64`, so the
        // four 2-lane loads at offsets 0, 2, 4 and 6 cover each chunk exactly.
        unsafe {
            let (ap, bp) = (ac.as_ptr(), bc.as_ptr());
            acc0 = _mm_add_pd(acc0, _mm_mul_pd(_mm_loadu_pd(ap), _mm_loadu_pd(bp)));
            acc1 = _mm_add_pd(
                acc1,
                _mm_mul_pd(_mm_loadu_pd(ap.add(2)), _mm_loadu_pd(bp.add(2))),
            );
            acc2 = _mm_add_pd(
                acc2,
                _mm_mul_pd(_mm_loadu_pd(ap.add(4)), _mm_loadu_pd(bp.add(4))),
            );
            acc3 = _mm_add_pd(
                acc3,
                _mm_mul_pd(_mm_loadu_pd(ap.add(6)), _mm_loadu_pd(bp.add(6))),
            );
        }
    }

    // SAFETY: register arithmetic only — no memory is touched.
    let mut sum = unsafe {
        let s01 = _mm_add_pd(acc0, acc1);
        let s23 = _mm_add_pd(acc2, acc3);
        let s = _mm_add_pd(s01, s23);
        let high = _mm_unpackhi_pd(s, s);
        _mm_cvtsd_f64(_mm_add_sd(s, high))
    };

    // Remainder: up to 7 elements — 2-wide vectors first, then scalar.
    // SAFETY: register broadcast of zero.
    let mut acc_rem = unsafe { _mm_setzero_pd() };
    let mut ar = ai.remainder().chunks_exact(2);
    let mut br = bi.remainder().chunks_exact(2);
    for (ac, bc) in (&mut ar).zip(&mut br) {
        // SAFETY: each chunk is exactly 2 `f64` — one vector load each.
        unsafe {
            acc_rem = _mm_add_pd(
                acc_rem,
                _mm_mul_pd(_mm_loadu_pd(ac.as_ptr()), _mm_loadu_pd(bc.as_ptr())),
            );
        }
    }
    // SAFETY: register arithmetic only.
    sum += unsafe {
        {
            let rh = _mm_unpackhi_pd(acc_rem, acc_rem);
            _mm_cvtsd_f64(_mm_add_sd(acc_rem, rh))
        }
    };

    for (&x, &y) in ar.remainder().iter().zip(br.remainder()) {
        sum += x * y;
    }
    sum
}

/// Matrix multiply C += A * B using SSE2 with register-blocked micro-kernel.
///
/// Uses an MR×NR (4×4) register-blocked micro-kernel with k-blocking (KC=256)
/// to keep the A panel and B micro-panel in L2 cache. Accumulates the full
/// k-block in SSE2 registers before writing back to C. Technique inspired by
/// nano-gemm (Sarah Quinones, <https://github.com/sarah-quinones/nano-gemm>).
///
/// `a` is m×n, `b` is n×p, `c` is m×p (column-major flat slices).
/// Column-major indexing: element (row, col) is at `col * nrows + row`.
#[inline]
pub fn matmul(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, p: usize) {
    debug_assert_eq!(a.len(), m * n);
    debug_assert_eq!(b.len(), n * p);
    debug_assert_eq!(c.len(), m * p);

    const MR: usize = 4; // 2 __m128d vectors × 2 f64 lanes
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
                // SAFETY: `i0 + 4 <= m_full <= m` and `j0 + 4 <= p_full <= p` by the
                // tile loops' construction, and `kb <= k_end <= n` — exactly the
                // microkernel's `# Safety` bounds contract.
                unsafe {
                    microkernel_4x4(a, b, c, m, n, i0, j0, kb, k_end);
                }
            }
        }

        // Bottom edge: rows m_full..m, cols 0..p_full
        let mut i0 = m_full;
        while i0 + 2 <= m {
            for jb in 0..p_full / NR {
                let j0 = jb * NR;
                // SAFETY: the loop condition guarantees `i0 + 2 <= m`, the `jb` loop
                // gives `j0 + 4 <= p_full <= p`, and `kb <= k_end <= n` — exactly the
                // microkernel's `# Safety` bounds contract.
                unsafe {
                    microkernel_2x4(a, b, c, m, n, i0, j0, kb, k_end);
                }
            }
            i0 += 2;
        }

        // Scalar tail: any single remaining row
        if i0 < m {
            for j in 0..p_full {
                for k in kb..k_end {
                    c[j * m + i0] += a[k * m + i0] * b[j * n + k];
                }
            }
        }

        // Right edge: cols p_full..p, all rows (SIMD j-k-i on inner loop)
        let i_simd = m / 2;
        let i_tail = i_simd * 2;
        for j in p_full..p {
            for k in kb..k_end {
                let b_kj = b[j * n + k];
                let a_col = k * m;
                let c_col = j * m;
                // SAFETY: the broadcast touches no memory. Each iteration loads and
                // stores one 2-lane vector at `offset = i·2` with `i < i_simd = m / 2`,
                // so `offset + 2 <= m`: every access stays inside column `k` of `a`
                // (`a_col = k·m`, `k < n`) and column `j` of `c` (`c_col = j·m`, `j < p`).
                unsafe {
                    let vb = _mm_set1_pd(b_kj);
                    for i in 0..i_simd {
                        let offset = i * 2;
                        let vc = _mm_loadu_pd(c.as_ptr().add(c_col + offset));
                        let va = _mm_loadu_pd(a.as_ptr().add(a_col + offset));
                        _mm_storeu_pd(
                            c.as_mut_ptr().add(c_col + offset),
                            _mm_add_pd(vc, _mm_mul_pd(va, vb)),
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

/// Register-blocked 4×4 micro-kernel: accumulates C[i0..i0+4, j0..j0+4] in
/// 8 SSE2 registers across a k-block, writing C only once per block.
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
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
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
        let mut acc00 = _mm_setzero_pd();
        let mut acc10 = _mm_setzero_pd();
        let mut acc01 = _mm_setzero_pd();
        let mut acc11 = _mm_setzero_pd();
        let mut acc02 = _mm_setzero_pd();
        let mut acc12 = _mm_setzero_pd();
        let mut acc03 = _mm_setzero_pd();
        let mut acc13 = _mm_setzero_pd();

        for k in k_start..k_end {
            let a_off = k * m + i0;
            let a0 = _mm_loadu_pd(a_ptr.add(a_off));
            let a1 = _mm_loadu_pd(a_ptr.add(a_off + 2));

            let b0 = _mm_set1_pd(*b_ptr.add(j0 * n + k));
            acc00 = _mm_add_pd(acc00, _mm_mul_pd(a0, b0));
            acc10 = _mm_add_pd(acc10, _mm_mul_pd(a1, b0));

            let b1 = _mm_set1_pd(*b_ptr.add((j0 + 1) * n + k));
            acc01 = _mm_add_pd(acc01, _mm_mul_pd(a0, b1));
            acc11 = _mm_add_pd(acc11, _mm_mul_pd(a1, b1));

            let b2 = _mm_set1_pd(*b_ptr.add((j0 + 2) * n + k));
            acc02 = _mm_add_pd(acc02, _mm_mul_pd(a0, b2));
            acc12 = _mm_add_pd(acc12, _mm_mul_pd(a1, b2));

            let b3 = _mm_set1_pd(*b_ptr.add((j0 + 3) * n + k));
            acc03 = _mm_add_pd(acc03, _mm_mul_pd(a0, b3));
            acc13 = _mm_add_pd(acc13, _mm_mul_pd(a1, b3));
        }

        // Write back: C += acc
        let c_ptr = c.as_mut_ptr();

        let off0 = j0 * m + i0;
        _mm_storeu_pd(
            c_ptr.add(off0),
            _mm_add_pd(_mm_loadu_pd(c_ptr.add(off0)), acc00),
        );
        _mm_storeu_pd(
            c_ptr.add(off0 + 2),
            _mm_add_pd(_mm_loadu_pd(c_ptr.add(off0 + 2)), acc10),
        );

        let off1 = (j0 + 1) * m + i0;
        _mm_storeu_pd(
            c_ptr.add(off1),
            _mm_add_pd(_mm_loadu_pd(c_ptr.add(off1)), acc01),
        );
        _mm_storeu_pd(
            c_ptr.add(off1 + 2),
            _mm_add_pd(_mm_loadu_pd(c_ptr.add(off1 + 2)), acc11),
        );

        let off2 = (j0 + 2) * m + i0;
        _mm_storeu_pd(
            c_ptr.add(off2),
            _mm_add_pd(_mm_loadu_pd(c_ptr.add(off2)), acc02),
        );
        _mm_storeu_pd(
            c_ptr.add(off2 + 2),
            _mm_add_pd(_mm_loadu_pd(c_ptr.add(off2 + 2)), acc12),
        );

        let off3 = (j0 + 3) * m + i0;
        _mm_storeu_pd(
            c_ptr.add(off3),
            _mm_add_pd(_mm_loadu_pd(c_ptr.add(off3)), acc03),
        );
        _mm_storeu_pd(
            c_ptr.add(off3 + 2),
            _mm_add_pd(_mm_loadu_pd(c_ptr.add(off3 + 2)), acc13),
        );
    }
}

/// Register-blocked 2×4 mini-kernel for bottom-edge rows: accumulates
/// C[i0..i0+2, j0..j0+4] in 4 SSE2 registers across a k-block.
///
/// # Safety
///
/// With `a` an `m×n`, `b` an `n×p` and `c` an `m×p` column-major matrix — so
/// `a.len() == m * n`, `b.len() == n * p` and `c.len() == m * p` — the caller
/// must guarantee that the tile and k-range lie inside them:
///
/// - `i0 + 2 <= m`, so the tile's 2 rows are within `a`'s and `c`'s columns;
/// - `j0 + 4 <= p`, so the tile's 4 columns are within `b` and `c`;
/// - `k_start <= k_end <= n`, so every `k` indexes a real column of `a` / row of `b`.
///
/// Every load and store below is then in bounds. SSE2 is part of the `x86_64` baseline,
/// which the module's `#[cfg(target_arch = "x86_64")]` gate guarantees.
#[inline(always)]
unsafe fn microkernel_2x4(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
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

        let mut acc0 = _mm_setzero_pd();
        let mut acc1 = _mm_setzero_pd();
        let mut acc2 = _mm_setzero_pd();
        let mut acc3 = _mm_setzero_pd();

        for k in k_start..k_end {
            let a0 = _mm_loadu_pd(a_ptr.add(k * m + i0));

            acc0 = _mm_add_pd(acc0, _mm_mul_pd(a0, _mm_set1_pd(*b_ptr.add(j0 * n + k))));
            acc1 = _mm_add_pd(
                acc1,
                _mm_mul_pd(a0, _mm_set1_pd(*b_ptr.add((j0 + 1) * n + k))),
            );
            acc2 = _mm_add_pd(
                acc2,
                _mm_mul_pd(a0, _mm_set1_pd(*b_ptr.add((j0 + 2) * n + k))),
            );
            acc3 = _mm_add_pd(
                acc3,
                _mm_mul_pd(a0, _mm_set1_pd(*b_ptr.add((j0 + 3) * n + k))),
            );
        }

        let c_ptr = c.as_mut_ptr();
        let off0 = j0 * m + i0;
        _mm_storeu_pd(
            c_ptr.add(off0),
            _mm_add_pd(_mm_loadu_pd(c_ptr.add(off0)), acc0),
        );
        let off1 = (j0 + 1) * m + i0;
        _mm_storeu_pd(
            c_ptr.add(off1),
            _mm_add_pd(_mm_loadu_pd(c_ptr.add(off1)), acc1),
        );
        let off2 = (j0 + 2) * m + i0;
        _mm_storeu_pd(
            c_ptr.add(off2),
            _mm_add_pd(_mm_loadu_pd(c_ptr.add(off2)), acc2),
        );
        let off3 = (j0 + 3) * m + i0;
        _mm_storeu_pd(
            c_ptr.add(off3),
            _mm_add_pd(_mm_loadu_pd(c_ptr.add(off3)), acc3),
        );
    }
}

// Element-wise add/sub/scale and AXPY kernels are generated from the shared
// macros in `super` (identical across ISAs bar width + intrinsic names).
simd_elementwise_kernels!(
    f64,
    2,
    _mm_loadu_pd,
    _mm_storeu_pd,
    _mm_add_pd,
    _mm_sub_pd,
    _mm_mul_pd,
    _mm_set1_pd
);
simd_axpy_kernels_muladd!(
    f64,
    2,
    _mm_loadu_pd,
    _mm_storeu_pd,
    _mm_add_pd,
    _mm_sub_pd,
    _mm_mul_pd,
    _mm_set1_pd
);
simd_conv1d_kernel_muladd!(
    f64,
    2,
    _mm_loadu_pd,
    _mm_storeu_pd,
    _mm_add_pd,
    _mm_mul_pd,
    _mm_set1_pd
);
