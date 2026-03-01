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
    let n = a.len();
    let chunks = n / 8; // 4 accumulators × 2 lanes

    unsafe {
        let ap = a.as_ptr();
        let bp = b.as_ptr();

        let mut acc0 = _mm_setzero_pd();
        let mut acc1 = _mm_setzero_pd();
        let mut acc2 = _mm_setzero_pd();
        let mut acc3 = _mm_setzero_pd();

        for i in 0..chunks {
            let off = i * 8;
            acc0 = _mm_add_pd(acc0, _mm_mul_pd(_mm_loadu_pd(ap.add(off)), _mm_loadu_pd(bp.add(off))));
            acc1 = _mm_add_pd(acc1, _mm_mul_pd(_mm_loadu_pd(ap.add(off + 2)), _mm_loadu_pd(bp.add(off + 2))));
            acc2 = _mm_add_pd(acc2, _mm_mul_pd(_mm_loadu_pd(ap.add(off + 4)), _mm_loadu_pd(bp.add(off + 4))));
            acc3 = _mm_add_pd(acc3, _mm_mul_pd(_mm_loadu_pd(ap.add(off + 6)), _mm_loadu_pd(bp.add(off + 6))));
        }

        // Reduce 4 accumulators
        acc0 = _mm_add_pd(acc0, acc1);
        acc2 = _mm_add_pd(acc2, acc3);
        acc0 = _mm_add_pd(acc0, acc2);
        let high = _mm_unpackhi_pd(acc0, acc0);
        let sum_vec = _mm_add_sd(acc0, high);
        let mut sum = _mm_cvtsd_f64(sum_vec);

        // Remainder: up to 7 elements — handle pairs then scalar
        let tail = chunks * 8;
        let remaining = n - tail;
        let rem_pairs = remaining / 2;
        let mut acc_rem = _mm_setzero_pd();
        for i in 0..rem_pairs {
            let off = tail + i * 2;
            acc_rem = _mm_add_pd(acc_rem, _mm_mul_pd(_mm_loadu_pd(ap.add(off)), _mm_loadu_pd(bp.add(off))));
        }
        let rh = _mm_unpackhi_pd(acc_rem, acc_rem);
        sum += _mm_cvtsd_f64(_mm_add_sd(acc_rem, rh));

        let scalar_start = tail + rem_pairs * 2;
        for i in scalar_start..n {
            sum += a[i] * b[i];
        }
        sum
    }
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
                unsafe { microkernel_4x4(a, b, c, m, n, i0, j0, kb, k_end); }
            }
        }

        // Bottom edge: rows m_full..m, cols 0..p_full
        let mut i0 = m_full;
        while i0 + 2 <= m {
            for jb in 0..p_full / NR {
                let j0 = jb * NR;
                unsafe { microkernel_2x4(a, b, c, m, n, i0, j0, kb, k_end); }
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
                unsafe {
                    let vb = _mm_set1_pd(b_kj);
                    for i in 0..i_simd {
                        let offset = i * 2;
                        let vc = _mm_loadu_pd(c.as_ptr().add(c_col + offset));
                        let va = _mm_loadu_pd(a.as_ptr().add(a_col + offset));
                        _mm_storeu_pd(c.as_mut_ptr().add(c_col + offset), _mm_add_pd(vc, _mm_mul_pd(va, vb)));
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
#[inline(always)]
unsafe fn microkernel_4x4(
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, n: usize, i0: usize, j0: usize,
    k_start: usize, k_end: usize,
) {
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
        _mm_storeu_pd(c_ptr.add(off0), _mm_add_pd(_mm_loadu_pd(c_ptr.add(off0)), acc00));
        _mm_storeu_pd(c_ptr.add(off0 + 2), _mm_add_pd(_mm_loadu_pd(c_ptr.add(off0 + 2)), acc10));

        let off1 = (j0 + 1) * m + i0;
        _mm_storeu_pd(c_ptr.add(off1), _mm_add_pd(_mm_loadu_pd(c_ptr.add(off1)), acc01));
        _mm_storeu_pd(c_ptr.add(off1 + 2), _mm_add_pd(_mm_loadu_pd(c_ptr.add(off1 + 2)), acc11));

        let off2 = (j0 + 2) * m + i0;
        _mm_storeu_pd(c_ptr.add(off2), _mm_add_pd(_mm_loadu_pd(c_ptr.add(off2)), acc02));
        _mm_storeu_pd(c_ptr.add(off2 + 2), _mm_add_pd(_mm_loadu_pd(c_ptr.add(off2 + 2)), acc12));

        let off3 = (j0 + 3) * m + i0;
        _mm_storeu_pd(c_ptr.add(off3), _mm_add_pd(_mm_loadu_pd(c_ptr.add(off3)), acc03));
        _mm_storeu_pd(c_ptr.add(off3 + 2), _mm_add_pd(_mm_loadu_pd(c_ptr.add(off3 + 2)), acc13));
    }
}

/// Register-blocked 2×4 mini-kernel for bottom-edge rows: accumulates
/// C[i0..i0+2, j0..j0+4] in 4 SSE2 registers across a k-block.
#[inline(always)]
unsafe fn microkernel_2x4(
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, n: usize, i0: usize, j0: usize,
    k_start: usize, k_end: usize,
) {
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
            acc1 = _mm_add_pd(acc1, _mm_mul_pd(a0, _mm_set1_pd(*b_ptr.add((j0 + 1) * n + k))));
            acc2 = _mm_add_pd(acc2, _mm_mul_pd(a0, _mm_set1_pd(*b_ptr.add((j0 + 2) * n + k))));
            acc3 = _mm_add_pd(acc3, _mm_mul_pd(a0, _mm_set1_pd(*b_ptr.add((j0 + 3) * n + k))));
        }

        let c_ptr = c.as_mut_ptr();
        let off0 = j0 * m + i0;
        _mm_storeu_pd(c_ptr.add(off0), _mm_add_pd(_mm_loadu_pd(c_ptr.add(off0)), acc0));
        let off1 = (j0 + 1) * m + i0;
        _mm_storeu_pd(c_ptr.add(off1), _mm_add_pd(_mm_loadu_pd(c_ptr.add(off1)), acc1));
        let off2 = (j0 + 2) * m + i0;
        _mm_storeu_pd(c_ptr.add(off2), _mm_add_pd(_mm_loadu_pd(c_ptr.add(off2)), acc2));
        let off3 = (j0 + 3) * m + i0;
        _mm_storeu_pd(c_ptr.add(off3), _mm_add_pd(_mm_loadu_pd(c_ptr.add(off3)), acc3));
    }
}

/// Element-wise addition: out[i] = a[i] + b[i].
#[inline]
pub fn add_slices(a: &[f64], b: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 2;

    unsafe {
        for i in 0..chunks {
            let offset = i * 2;
            let va = _mm_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm_loadu_pd(b.as_ptr().add(offset));
            _mm_storeu_pd(out.as_mut_ptr().add(offset), _mm_add_pd(va, vb));
        }
    }

    let tail = chunks * 2;
    for i in tail..n {
        out[i] = a[i] + b[i];
    }
}

/// Element-wise subtraction: out[i] = a[i] - b[i].
#[inline]
pub fn sub_slices(a: &[f64], b: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 2;

    unsafe {
        for i in 0..chunks {
            let offset = i * 2;
            let va = _mm_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm_loadu_pd(b.as_ptr().add(offset));
            _mm_storeu_pd(out.as_mut_ptr().add(offset), _mm_sub_pd(va, vb));
        }
    }

    let tail = chunks * 2;
    for i in tail..n {
        out[i] = a[i] - b[i];
    }
}

/// Scalar multiplication: out[i] = a[i] * scalar.
#[inline]
pub fn scale_slices(a: &[f64], scalar: f64, out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 2;

    unsafe {
        let vs = _mm_set1_pd(scalar);
        for i in 0..chunks {
            let offset = i * 2;
            let va = _mm_loadu_pd(a.as_ptr().add(offset));
            _mm_storeu_pd(out.as_mut_ptr().add(offset), _mm_mul_pd(va, vs));
        }
    }

    let tail = chunks * 2;
    for i in tail..n {
        out[i] = a[i] * scalar;
    }
}

/// AXPY: y[i] -= alpha * x[i].
#[inline]
pub fn axpy_neg(y: &mut [f64], alpha: f64, x: &[f64]) {
    debug_assert_eq!(y.len(), x.len());
    let n = y.len();
    let chunks = n / 2;

    unsafe {
        let va = _mm_set1_pd(alpha);
        for i in 0..chunks {
            let offset = i * 2;
            let vy = _mm_loadu_pd(y.as_ptr().add(offset));
            let vx = _mm_loadu_pd(x.as_ptr().add(offset));
            let result = _mm_sub_pd(vy, _mm_mul_pd(va, vx));
            _mm_storeu_pd(y.as_mut_ptr().add(offset), result);
        }
    }

    let tail = chunks * 2;
    for i in tail..n {
        y[i] -= alpha * x[i];
    }
}
