//! NEON-accelerated f64 kernels for aarch64.
//!
//! NEON provides 128-bit registers → 2×f64 lanes.

use core::arch::aarch64::*;

/// Dot product of two f64 slices using NEON.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 2;
    let remainder = n % 2;

    unsafe {
        let mut acc = vdupq_n_f64(0.0);

        for i in 0..chunks {
            let va = vld1q_f64(a.as_ptr().add(i * 2));
            let vb = vld1q_f64(b.as_ptr().add(i * 2));
            acc = vfmaq_f64(acc, va, vb);
        }

        let mut sum = vaddvq_f64(acc);

        // Handle remainder
        let tail = chunks * 2;
        for i in 0..remainder {
            sum += a[tail + i] * b[tail + i];
        }
        sum
    }
}

/// Matrix multiply C += A * B using NEON with register-blocked micro-kernel.
///
/// Uses an MR×NR (4×4) register-blocked micro-kernel that accumulates the full
/// k-sum in 8 NEON registers before writing back to C, reducing memory traffic
/// from O(m·n·p) to O(m·p) stores. Technique inspired by nano-gemm
/// (Sarah Quinones, <https://github.com/sarah-quinones/nano-gemm>).
///
/// `a` is m×n, `b` is n×p, `c` is m×p (column-major flat slices).
/// Column-major indexing: element (row, col) is at `col * nrows + row`.
#[inline]
pub fn matmul(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, p: usize) {
    debug_assert_eq!(a.len(), m * n);
    debug_assert_eq!(b.len(), n * p);
    debug_assert_eq!(c.len(), m * p);

    const MR: usize = 4; // 2 NEON registers × 2 f64 lanes
    const NR: usize = 4;

    let m_full = (m / MR) * MR;
    let p_full = (p / NR) * NR;

    // Interior: full MR×NR tiles, register-blocked
    for jb in 0..p_full / NR {
        let j0 = jb * NR;
        for ib in 0..m_full / MR {
            let i0 = ib * MR;
            unsafe { microkernel_4x4(a, b, c, m, n, i0, j0); }
        }
    }

    // Bottom edge: rows m_full..m, cols 0..p_full (scalar)
    for j in 0..p_full {
        for k in 0..n {
            let b_kj = b[j * n + k];
            let a_col = k * m;
            let c_col = j * m;
            for i in m_full..m {
                c[c_col + i] += a[a_col + i] * b_kj;
            }
        }
    }

    // Right edge: cols p_full..p, all rows (SIMD j-k-i on inner loop)
    let i_simd = m / 2;
    let i_tail = i_simd * 2;
    for j in p_full..p {
        for k in 0..n {
            let b_kj = b[j * n + k];
            let a_col = k * m;
            let c_col = j * m;
            unsafe {
                let vb = vdupq_n_f64(b_kj);
                for i in 0..i_simd {
                    let offset = i * 2;
                    let vc = vld1q_f64(c.as_ptr().add(c_col + offset));
                    let va = vld1q_f64(a.as_ptr().add(a_col + offset));
                    vst1q_f64(c.as_mut_ptr().add(c_col + offset), vfmaq_f64(vc, va, vb));
                }
            }
            for i in i_tail..m {
                c[c_col + i] += a[a_col + i] * b_kj;
            }
        }
    }
}

/// Register-blocked 4×4 micro-kernel: accumulates C[i0..i0+4, j0..j0+4] in
/// 8 NEON registers across the full k-loop, writing C only once.
#[inline(always)]
unsafe fn microkernel_4x4(
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, n: usize, i0: usize, j0: usize,
) {
    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // 8 accumulator registers: 2 vectors × 4 columns
        let mut acc00 = vdupq_n_f64(0.0);
        let mut acc10 = vdupq_n_f64(0.0);
        let mut acc01 = vdupq_n_f64(0.0);
        let mut acc11 = vdupq_n_f64(0.0);
        let mut acc02 = vdupq_n_f64(0.0);
        let mut acc12 = vdupq_n_f64(0.0);
        let mut acc03 = vdupq_n_f64(0.0);
        let mut acc13 = vdupq_n_f64(0.0);

        for k in 0..n {
            let a_off = k * m + i0;
            let a0 = vld1q_f64(a_ptr.add(a_off));
            let a1 = vld1q_f64(a_ptr.add(a_off + 2));

            let b0 = vdupq_n_f64(*b_ptr.add(j0 * n + k));
            acc00 = vfmaq_f64(acc00, a0, b0);
            acc10 = vfmaq_f64(acc10, a1, b0);

            let b1 = vdupq_n_f64(*b_ptr.add((j0 + 1) * n + k));
            acc01 = vfmaq_f64(acc01, a0, b1);
            acc11 = vfmaq_f64(acc11, a1, b1);

            let b2 = vdupq_n_f64(*b_ptr.add((j0 + 2) * n + k));
            acc02 = vfmaq_f64(acc02, a0, b2);
            acc12 = vfmaq_f64(acc12, a1, b2);

            let b3 = vdupq_n_f64(*b_ptr.add((j0 + 3) * n + k));
            acc03 = vfmaq_f64(acc03, a0, b3);
            acc13 = vfmaq_f64(acc13, a1, b3);
        }

        // Write back: C += acc
        let c_ptr = c.as_mut_ptr();

        let off0 = j0 * m + i0;
        vst1q_f64(c_ptr.add(off0), vaddq_f64(vld1q_f64(c_ptr.add(off0)), acc00));
        vst1q_f64(c_ptr.add(off0 + 2), vaddq_f64(vld1q_f64(c_ptr.add(off0 + 2)), acc10));

        let off1 = (j0 + 1) * m + i0;
        vst1q_f64(c_ptr.add(off1), vaddq_f64(vld1q_f64(c_ptr.add(off1)), acc01));
        vst1q_f64(c_ptr.add(off1 + 2), vaddq_f64(vld1q_f64(c_ptr.add(off1 + 2)), acc11));

        let off2 = (j0 + 2) * m + i0;
        vst1q_f64(c_ptr.add(off2), vaddq_f64(vld1q_f64(c_ptr.add(off2)), acc02));
        vst1q_f64(c_ptr.add(off2 + 2), vaddq_f64(vld1q_f64(c_ptr.add(off2 + 2)), acc12));

        let off3 = (j0 + 3) * m + i0;
        vst1q_f64(c_ptr.add(off3), vaddq_f64(vld1q_f64(c_ptr.add(off3)), acc03));
        vst1q_f64(c_ptr.add(off3 + 2), vaddq_f64(vld1q_f64(c_ptr.add(off3 + 2)), acc13));
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
            let va = vld1q_f64(a.as_ptr().add(offset));
            let vb = vld1q_f64(b.as_ptr().add(offset));
            vst1q_f64(out.as_mut_ptr().add(offset), vaddq_f64(va, vb));
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
            let va = vld1q_f64(a.as_ptr().add(offset));
            let vb = vld1q_f64(b.as_ptr().add(offset));
            vst1q_f64(out.as_mut_ptr().add(offset), vsubq_f64(va, vb));
        }
    }

    let tail = chunks * 2;
    for i in tail..n {
        out[i] = a[i] - b[i];
    }
}

/// AXPY: y[i] -= alpha * x[i].
#[inline]
pub fn axpy_neg(y: &mut [f64], alpha: f64, x: &[f64]) {
    debug_assert_eq!(y.len(), x.len());
    let n = y.len();
    let chunks = n / 2;

    unsafe {
        let va = vdupq_n_f64(alpha);
        for i in 0..chunks {
            let offset = i * 2;
            let vy = vld1q_f64(y.as_ptr().add(offset));
            let vx = vld1q_f64(x.as_ptr().add(offset));
            // y -= alpha * x  →  y = y - alpha * x  →  vfmsq_f64(y, alpha, x)
            let result = vfmsq_f64(vy, va, vx);
            vst1q_f64(y.as_mut_ptr().add(offset), result);
        }
    }

    let tail = chunks * 2;
    for i in tail..n {
        y[i] -= alpha * x[i];
    }
}

/// Scalar multiplication: out[i] = a[i] * scalar.
#[inline]
pub fn scale_slices(a: &[f64], scalar: f64, out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 2;

    unsafe {
        let vs = vdupq_n_f64(scalar);
        for i in 0..chunks {
            let offset = i * 2;
            let va = vld1q_f64(a.as_ptr().add(offset));
            vst1q_f64(out.as_mut_ptr().add(offset), vmulq_f64(va, vs));
        }
    }

    let tail = chunks * 2;
    for i in tail..n {
        out[i] = a[i] * scalar;
    }
}
