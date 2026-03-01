//! Generic scalar fallback implementations for SIMD-dispatched operations.
//!
//! These are used for types that don't have SIMD specializations (integers,
//! complex numbers) or on architectures without SIMD support.

use crate::traits::Scalar;

/// Dot product of two slices (scalar fallback).
#[inline]
pub fn dot<T: Scalar>(a: &[T], b: &[T]) -> T {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum = sum + a[i] * b[i];
    }
    sum
}

/// Matrix multiply C += A * B with register-blocked micro-kernel and k-blocking.
///
/// Uses a 4×4 register-blocked approach with k-blocking (KC=256) to keep the
/// A panel and B micro-panel in L2 cache. Accumulates the full k-block in
/// local variables before writing back to C. Technique inspired by nano-gemm
/// (Sarah Quinones, <https://github.com/sarah-quinones/nano-gemm>).
///
/// `a` is m×n, `b` is n×p, `c` is m×p (all column-major flat slices).
/// Column-major indexing: element (row, col) is at `col * nrows + row`.
#[inline]
pub fn matmul<T: Scalar>(a: &[T], b: &[T], c: &mut [T], m: usize, n: usize, p: usize) {
    debug_assert_eq!(a.len(), m * n);
    debug_assert_eq!(b.len(), n * p);
    debug_assert_eq!(c.len(), m * p);

    const MR: usize = 4;
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

                // 16 scalar accumulators (4 rows × 4 cols)
                let mut acc = [[T::zero(); NR]; MR];

                for k in kb..k_end {
                    let a_base = k * m + i0;
                    for jj in 0..NR {
                        let b_val = b[(j0 + jj) * n + k];
                        for ii in 0..MR {
                            acc[ii][jj] = acc[ii][jj] + a[a_base + ii] * b_val;
                        }
                    }
                }

                // Write back: C += acc
                for jj in 0..NR {
                    let c_base = (j0 + jj) * m + i0;
                    for ii in 0..MR {
                        c[c_base + ii] = c[c_base + ii] + acc[ii][jj];
                    }
                }
            }
        }

        // Bottom edge: rows m_full..m, cols 0..p_full
        for j in 0..p_full {
            for k in kb..k_end {
                let b_kj = b[j * n + k];
                let a_col = k * m;
                let c_col = j * m;
                for i in m_full..m {
                    c[c_col + i] = c[c_col + i] + a[a_col + i] * b_kj;
                }
            }
        }

        // Right edge: cols p_full..p, all rows
        for j in p_full..p {
            for k in kb..k_end {
                let b_kj = b[j * n + k];
                for i in 0..m {
                    c[j * m + i] = c[j * m + i] + a[k * m + i] * b_kj;
                }
            }
        }

        kb += KC;
    }
}

/// Element-wise addition: out[i] = a[i] + b[i].
#[inline]
pub fn add_slices<T: Scalar>(a: &[T], b: &[T], out: &mut [T]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

/// Element-wise subtraction: out[i] = a[i] - b[i].
#[inline]
pub fn sub_slices<T: Scalar>(a: &[T], b: &[T], out: &mut [T]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] - b[i];
    }
}

/// Scalar multiplication: out[i] = a[i] * scalar.
#[inline]
pub fn scale_slices<T: Scalar>(a: &[T], scalar: T, out: &mut [T]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] * scalar;
    }
}

/// In-place addition: a[i] += b[i].
#[inline]
pub fn add_assign_slices<T: Scalar>(a: &mut [T], b: &[T]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] = a[i] + b[i];
    }
}

/// In-place subtraction: a[i] -= b[i].
#[inline]
pub fn sub_assign_slices<T: Scalar>(a: &mut [T], b: &[T]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] = a[i] - b[i];
    }
}

/// In-place scale: a[i] *= scalar.
#[inline]
pub fn scale_assign_slices<T: Scalar>(a: &mut [T], scalar: T) {
    for i in 0..a.len() {
        a[i] = a[i] * scalar;
    }
}

/// AXPY: y[i] = y[i] - alpha * x[i].
///
/// This is the core operation in Gaussian elimination, Householder reflections,
/// and most linalg inner loops. Uses subtraction because that's the dominant
/// pattern in decompositions.
#[inline]
#[allow(dead_code)]
pub fn axpy_neg<T: Scalar>(y: &mut [T], alpha: T, x: &[T]) {
    debug_assert_eq!(y.len(), x.len());
    for i in 0..y.len() {
        y[i] = y[i] - alpha * x[i];
    }
}
