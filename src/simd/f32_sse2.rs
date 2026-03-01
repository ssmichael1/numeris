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
    let n = a.len();
    let chunks = n / 16; // 4 accumulators × 4 lanes

    unsafe {
        let ap = a.as_ptr();
        let bp = b.as_ptr();

        let mut acc0 = _mm_setzero_ps();
        let mut acc1 = _mm_setzero_ps();
        let mut acc2 = _mm_setzero_ps();
        let mut acc3 = _mm_setzero_ps();

        for i in 0..chunks {
            let off = i * 16;
            acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(ap.add(off)), _mm_loadu_ps(bp.add(off))));
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(ap.add(off + 4)), _mm_loadu_ps(bp.add(off + 4))));
            acc2 = _mm_add_ps(acc2, _mm_mul_ps(_mm_loadu_ps(ap.add(off + 8)), _mm_loadu_ps(bp.add(off + 8))));
            acc3 = _mm_add_ps(acc3, _mm_mul_ps(_mm_loadu_ps(ap.add(off + 12)), _mm_loadu_ps(bp.add(off + 12))));
        }

        acc0 = _mm_add_ps(acc0, acc1);
        acc2 = _mm_add_ps(acc2, acc3);
        acc0 = _mm_add_ps(acc0, acc2);
        // Horizontal sum of 4 lanes
        let shuf = _mm_movehl_ps(acc0, acc0);
        let sums = _mm_add_ps(acc0, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 1);
        let total = _mm_add_ss(sums, shuf2);
        let mut sum = _mm_cvtss_f32(total);

        // Remainder: up to 15 elements — handle quads then scalar
        let tail = chunks * 16;
        let remaining = n - tail;
        let rem_quads = remaining / 4;
        let mut acc_rem = _mm_setzero_ps();
        for i in 0..rem_quads {
            let off = tail + i * 4;
            acc_rem = _mm_add_ps(acc_rem, _mm_mul_ps(_mm_loadu_ps(ap.add(off)), _mm_loadu_ps(bp.add(off))));
        }
        let rs = _mm_movehl_ps(acc_rem, acc_rem);
        let rs2 = _mm_add_ps(acc_rem, rs);
        let rs3 = _mm_shuffle_ps(rs2, rs2, 1);
        sum += _mm_cvtss_f32(_mm_add_ss(rs2, rs3));

        let scalar_start = tail + rem_quads * 4;
        for i in scalar_start..n {
            sum += a[i] * b[i];
        }
        sum
    }
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
#[inline]
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, p: usize) {
    debug_assert_eq!(a.len(), m * n);
    debug_assert_eq!(b.len(), n * p);
    debug_assert_eq!(c.len(), m * p);

    const MR: usize = 8; // 2 __m128 vectors × 4 f32 lanes
    const NR: usize = 4;

    let m_full = (m / MR) * MR;
    let p_full = (p / NR) * NR;

    // Interior: full MR×NR tiles, register-blocked
    for jb in 0..p_full / NR {
        let j0 = jb * NR;
        for ib in 0..m_full / MR {
            let i0 = ib * MR;
            unsafe { microkernel_8x4(a, b, c, m, n, i0, j0); }
        }
    }

    // Bottom edge: rows m_full..m, cols 0..p_full
    let mut i0 = m_full;
    while i0 + 4 <= m {
        for jb in 0..p_full / NR {
            let j0 = jb * NR;
            unsafe { microkernel_4x4(a, b, c, m, n, i0, j0); }
        }
        i0 += 4;
    }
    if i0 < m {
        for j in 0..p_full {
            for k in 0..n {
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
        for k in 0..n {
            let b_kj = b[j * n + k];
            let a_col = k * m;
            let c_col = j * m;
            unsafe {
                let vb = _mm_set1_ps(b_kj);
                for i in 0..i_simd {
                    let offset = i * 4;
                    let vc = _mm_loadu_ps(c.as_ptr().add(c_col + offset));
                    let va = _mm_loadu_ps(a.as_ptr().add(a_col + offset));
                    _mm_storeu_ps(c.as_mut_ptr().add(c_col + offset), _mm_add_ps(vc, _mm_mul_ps(va, vb)));
                }
            }
            for i in i_tail..m {
                c[c_col + i] += a[a_col + i] * b_kj;
            }
        }
    }
}

/// Register-blocked 8×4 micro-kernel: accumulates C[i0..i0+8, j0..j0+4] in
/// 8 SSE2 registers across the full k-loop, writing C only once.
#[inline(always)]
unsafe fn microkernel_8x4(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, i0: usize, j0: usize,
) {
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

        for k in 0..n {
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
        _mm_storeu_ps(c_ptr.add(off0), _mm_add_ps(_mm_loadu_ps(c_ptr.add(off0)), acc00));
        _mm_storeu_ps(c_ptr.add(off0 + 4), _mm_add_ps(_mm_loadu_ps(c_ptr.add(off0 + 4)), acc10));

        let off1 = (j0 + 1) * m + i0;
        _mm_storeu_ps(c_ptr.add(off1), _mm_add_ps(_mm_loadu_ps(c_ptr.add(off1)), acc01));
        _mm_storeu_ps(c_ptr.add(off1 + 4), _mm_add_ps(_mm_loadu_ps(c_ptr.add(off1 + 4)), acc11));

        let off2 = (j0 + 2) * m + i0;
        _mm_storeu_ps(c_ptr.add(off2), _mm_add_ps(_mm_loadu_ps(c_ptr.add(off2)), acc02));
        _mm_storeu_ps(c_ptr.add(off2 + 4), _mm_add_ps(_mm_loadu_ps(c_ptr.add(off2 + 4)), acc12));

        let off3 = (j0 + 3) * m + i0;
        _mm_storeu_ps(c_ptr.add(off3), _mm_add_ps(_mm_loadu_ps(c_ptr.add(off3)), acc03));
        _mm_storeu_ps(c_ptr.add(off3 + 4), _mm_add_ps(_mm_loadu_ps(c_ptr.add(off3 + 4)), acc13));
    }
}

/// Register-blocked 4×4 mini-kernel for bottom-edge rows (1 __m128 per col).
#[inline(always)]
unsafe fn microkernel_4x4(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, i0: usize, j0: usize,
) {
    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut acc0 = _mm_setzero_ps();
        let mut acc1 = _mm_setzero_ps();
        let mut acc2 = _mm_setzero_ps();
        let mut acc3 = _mm_setzero_ps();

        for k in 0..n {
            let a0 = _mm_loadu_ps(a_ptr.add(k * m + i0));

            acc0 = _mm_add_ps(acc0, _mm_mul_ps(a0, _mm_set1_ps(*b_ptr.add(j0 * n + k))));
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(a0, _mm_set1_ps(*b_ptr.add((j0 + 1) * n + k))));
            acc2 = _mm_add_ps(acc2, _mm_mul_ps(a0, _mm_set1_ps(*b_ptr.add((j0 + 2) * n + k))));
            acc3 = _mm_add_ps(acc3, _mm_mul_ps(a0, _mm_set1_ps(*b_ptr.add((j0 + 3) * n + k))));
        }

        let c_ptr = c.as_mut_ptr();
        let off0 = j0 * m + i0;
        _mm_storeu_ps(c_ptr.add(off0), _mm_add_ps(_mm_loadu_ps(c_ptr.add(off0)), acc0));
        let off1 = (j0 + 1) * m + i0;
        _mm_storeu_ps(c_ptr.add(off1), _mm_add_ps(_mm_loadu_ps(c_ptr.add(off1)), acc1));
        let off2 = (j0 + 2) * m + i0;
        _mm_storeu_ps(c_ptr.add(off2), _mm_add_ps(_mm_loadu_ps(c_ptr.add(off2)), acc2));
        let off3 = (j0 + 3) * m + i0;
        _mm_storeu_ps(c_ptr.add(off3), _mm_add_ps(_mm_loadu_ps(c_ptr.add(off3)), acc3));
    }
}

/// Element-wise addition: out[i] = a[i] + b[i].
#[inline]
pub fn add_slices(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 4;

    unsafe {
        for i in 0..chunks {
            let offset = i * 4;
            let va = _mm_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm_loadu_ps(b.as_ptr().add(offset));
            _mm_storeu_ps(out.as_mut_ptr().add(offset), _mm_add_ps(va, vb));
        }
    }

    let tail = chunks * 4;
    for i in tail..n {
        out[i] = a[i] + b[i];
    }
}

/// Element-wise subtraction: out[i] = a[i] - b[i].
#[inline]
pub fn sub_slices(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 4;

    unsafe {
        for i in 0..chunks {
            let offset = i * 4;
            let va = _mm_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm_loadu_ps(b.as_ptr().add(offset));
            _mm_storeu_ps(out.as_mut_ptr().add(offset), _mm_sub_ps(va, vb));
        }
    }

    let tail = chunks * 4;
    for i in tail..n {
        out[i] = a[i] - b[i];
    }
}

/// Scalar multiplication: out[i] = a[i] * scalar.
#[inline]
pub fn scale_slices(a: &[f32], scalar: f32, out: &mut [f32]) {
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 4;

    unsafe {
        let vs = _mm_set1_ps(scalar);
        for i in 0..chunks {
            let offset = i * 4;
            let va = _mm_loadu_ps(a.as_ptr().add(offset));
            _mm_storeu_ps(out.as_mut_ptr().add(offset), _mm_mul_ps(va, vs));
        }
    }

    let tail = chunks * 4;
    for i in tail..n {
        out[i] = a[i] * scalar;
    }
}

/// AXPY: y[i] -= alpha * x[i].
#[inline]
pub fn axpy_neg(y: &mut [f32], alpha: f32, x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());
    let n = y.len();
    let chunks = n / 4;

    unsafe {
        let va = _mm_set1_ps(alpha);
        for i in 0..chunks {
            let offset = i * 4;
            let vy = _mm_loadu_ps(y.as_ptr().add(offset));
            let vx = _mm_loadu_ps(x.as_ptr().add(offset));
            let result = _mm_sub_ps(vy, _mm_mul_ps(va, vx));
            _mm_storeu_ps(y.as_mut_ptr().add(offset), result);
        }
    }

    let tail = chunks * 4;
    for i in tail..n {
        y[i] -= alpha * x[i];
    }
}
