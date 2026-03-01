//! SIMD-accelerated kernels with compile-time architecture dispatch.
//!
//! This module is private — it provides internal acceleration for matrix
//! and vector operations. The public API is unchanged.
//!
//! ## Dispatch strategy
//!
//! TypeId-based dispatch at monomorphization time: for `f32`/`f64`, the
//! compiler selects SIMD kernels and dead-code-eliminates the fallback.
//! For all other types (integers, complex), the scalar fallback is used.
//!
//! On x86_64, the widest available instruction set is selected at compile
//! time: AVX-512 > AVX > SSE2. Enable via `-C target-cpu=native` or
//! `-C target-feature=+avx2` etc.
//!
//! ## Matrix multiply
//!
//! All matmul kernels use register-blocked MR×NR micro-kernels that
//! accumulate the full k-sum in SIMD registers before writing C once,
//! reducing memory traffic from O(m·n·p) to O(m·p) stores. This technique
//! is inspired by [nano-gemm](https://github.com/sarah-quinones/nano-gemm)
//! and [faer](https://github.com/sarah-quinones/faer-rs) by Sarah Quinones.
//!
//! ## Architecture support
//!
//! | Arch      | ISA       | f64 tile | f32 tile |
//! |-----------|-----------|----------|----------|
//! | `aarch64` | NEON      | 4×4      | 8×4      |
//! | `x86_64`  | SSE2      | 4×4      | 8×4      |
//! | `x86_64`  | AVX       | 8×4      | 16×4     |
//! | `x86_64`  | AVX-512   | 16×4     | 32×4     |
//! | other     | scalar    | 4×4      | 4×4      |

pub(crate) mod scalar;

#[cfg(target_arch = "aarch64")]
pub(crate) mod f64_neon;
#[cfg(target_arch = "aarch64")]
pub(crate) mod f32_neon;

#[cfg(target_arch = "x86_64")]
pub(crate) mod f64_sse2;
#[cfg(target_arch = "x86_64")]
pub(crate) mod f32_sse2;

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
pub(crate) mod f64_avx;
#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
pub(crate) mod f32_avx;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub(crate) mod f64_avx512;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub(crate) mod f32_avx512;

use core::any::TypeId;

use crate::traits::Scalar;

/// Dispatch dot product to SIMD or scalar fallback.
#[inline]
pub(crate) fn dot_dispatch<T: Scalar>(a: &[T], b: &[T]) -> T {
    #[cfg(target_arch = "aarch64")]
    {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let a = unsafe { &*(a as *const [T] as *const [f64]) };
            let b = unsafe { &*(b as *const [T] as *const [f64]) };
            let result = f64_neon::dot(a, b);
            return unsafe { *(&result as *const f64 as *const T) };
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let a = unsafe { &*(a as *const [T] as *const [f32]) };
            let b = unsafe { &*(b as *const [T] as *const [f32]) };
            let result = f32_neon::dot(a, b);
            return unsafe { *(&result as *const f32 as *const T) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let a = unsafe { &*(a as *const [T] as *const [f64]) };
            let b = unsafe { &*(b as *const [T] as *const [f64]) };
            #[cfg(target_feature = "avx512f")]
            let result = f64_avx512::dot(a, b);
            #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
            let result = f64_avx::dot(a, b);
            #[cfg(not(target_feature = "avx"))]
            let result = f64_sse2::dot(a, b);
            return unsafe { *(&result as *const f64 as *const T) };
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let a = unsafe { &*(a as *const [T] as *const [f32]) };
            let b = unsafe { &*(b as *const [T] as *const [f32]) };
            #[cfg(target_feature = "avx512f")]
            let result = f32_avx512::dot(a, b);
            #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
            let result = f32_avx::dot(a, b);
            #[cfg(not(target_feature = "avx"))]
            let result = f32_sse2::dot(a, b);
            return unsafe { *(&result as *const f32 as *const T) };
        }
    }
    scalar::dot(a, b)
}

/// Dispatch matrix multiply to SIMD or scalar fallback.
///
/// `c` must be zero-initialized. Computes `C += A * B` in-place.
///
/// `c` must be zero-initialized. Computes `C += A * B` in-place.
#[inline]
pub(crate) fn matmul_dispatch<T: Scalar>(
    a: &[T],
    b: &[T],
    c: &mut [T],
    m: usize,
    n: usize,
    p: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let a = unsafe { &*(a as *const [T] as *const [f64]) };
            let b = unsafe { &*(b as *const [T] as *const [f64]) };
            let c = unsafe { &mut *(c as *mut [T] as *mut [f64]) };
            f64_neon::matmul(a, b, c, m, n, p);
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let a = unsafe { &*(a as *const [T] as *const [f32]) };
            let b = unsafe { &*(b as *const [T] as *const [f32]) };
            let c = unsafe { &mut *(c as *mut [T] as *mut [f32]) };
            f32_neon::matmul(a, b, c, m, n, p);
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let a = unsafe { &*(a as *const [T] as *const [f64]) };
            let b = unsafe { &*(b as *const [T] as *const [f64]) };
            let c = unsafe { &mut *(c as *mut [T] as *mut [f64]) };
            #[cfg(target_feature = "avx512f")]
            f64_avx512::matmul(a, b, c, m, n, p);
            #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
            f64_avx::matmul(a, b, c, m, n, p);
            #[cfg(not(target_feature = "avx"))]
            f64_sse2::matmul(a, b, c, m, n, p);
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let a = unsafe { &*(a as *const [T] as *const [f32]) };
            let b = unsafe { &*(b as *const [T] as *const [f32]) };
            let c = unsafe { &mut *(c as *mut [T] as *mut [f32]) };
            #[cfg(target_feature = "avx512f")]
            f32_avx512::matmul(a, b, c, m, n, p);
            #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
            f32_avx::matmul(a, b, c, m, n, p);
            #[cfg(not(target_feature = "avx"))]
            f32_sse2::matmul(a, b, c, m, n, p);
            return;
        }
    }
    scalar::matmul(a, b, c, m, n, p);
}

/// Dispatch element-wise addition to SIMD or scalar fallback.
#[inline]
pub(crate) fn add_slices_dispatch<T: Scalar>(a: &[T], b: &[T], out: &mut [T]) {
    #[cfg(target_arch = "aarch64")]
    {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let a = unsafe { &*(a as *const [T] as *const [f64]) };
            let b = unsafe { &*(b as *const [T] as *const [f64]) };
            let out = unsafe { &mut *(out as *mut [T] as *mut [f64]) };
            f64_neon::add_slices(a, b, out);
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let a = unsafe { &*(a as *const [T] as *const [f32]) };
            let b = unsafe { &*(b as *const [T] as *const [f32]) };
            let out = unsafe { &mut *(out as *mut [T] as *mut [f32]) };
            f32_neon::add_slices(a, b, out);
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let a = unsafe { &*(a as *const [T] as *const [f64]) };
            let b = unsafe { &*(b as *const [T] as *const [f64]) };
            let out = unsafe { &mut *(out as *mut [T] as *mut [f64]) };
            #[cfg(target_feature = "avx512f")]
            f64_avx512::add_slices(a, b, out);
            #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
            f64_avx::add_slices(a, b, out);
            #[cfg(not(target_feature = "avx"))]
            f64_sse2::add_slices(a, b, out);
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let a = unsafe { &*(a as *const [T] as *const [f32]) };
            let b = unsafe { &*(b as *const [T] as *const [f32]) };
            let out = unsafe { &mut *(out as *mut [T] as *mut [f32]) };
            #[cfg(target_feature = "avx512f")]
            f32_avx512::add_slices(a, b, out);
            #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
            f32_avx::add_slices(a, b, out);
            #[cfg(not(target_feature = "avx"))]
            f32_sse2::add_slices(a, b, out);
            return;
        }
    }
    scalar::add_slices(a, b, out);
}

/// Dispatch element-wise subtraction to SIMD or scalar fallback.
#[inline]
pub(crate) fn sub_slices_dispatch<T: Scalar>(a: &[T], b: &[T], out: &mut [T]) {
    #[cfg(target_arch = "aarch64")]
    {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let a = unsafe { &*(a as *const [T] as *const [f64]) };
            let b = unsafe { &*(b as *const [T] as *const [f64]) };
            let out = unsafe { &mut *(out as *mut [T] as *mut [f64]) };
            f64_neon::sub_slices(a, b, out);
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let a = unsafe { &*(a as *const [T] as *const [f32]) };
            let b = unsafe { &*(b as *const [T] as *const [f32]) };
            let out = unsafe { &mut *(out as *mut [T] as *mut [f32]) };
            f32_neon::sub_slices(a, b, out);
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let a = unsafe { &*(a as *const [T] as *const [f64]) };
            let b = unsafe { &*(b as *const [T] as *const [f64]) };
            let out = unsafe { &mut *(out as *mut [T] as *mut [f64]) };
            #[cfg(target_feature = "avx512f")]
            f64_avx512::sub_slices(a, b, out);
            #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
            f64_avx::sub_slices(a, b, out);
            #[cfg(not(target_feature = "avx"))]
            f64_sse2::sub_slices(a, b, out);
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let a = unsafe { &*(a as *const [T] as *const [f32]) };
            let b = unsafe { &*(b as *const [T] as *const [f32]) };
            let out = unsafe { &mut *(out as *mut [T] as *mut [f32]) };
            #[cfg(target_feature = "avx512f")]
            f32_avx512::sub_slices(a, b, out);
            #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
            f32_avx::sub_slices(a, b, out);
            #[cfg(not(target_feature = "avx"))]
            f32_sse2::sub_slices(a, b, out);
            return;
        }
    }
    scalar::sub_slices(a, b, out);
}

/// Dispatch AXPY: y[i] -= alpha * x[i].
///
/// For short slices (< 8 elements), uses a scalar loop to avoid the overhead
/// of SIMD dispatch and register setup, which dominates at small sizes.
#[inline]
pub(crate) fn axpy_neg_dispatch<T: Scalar>(y: &mut [T], alpha: T, x: &[T]) {
    let n = y.len();
    if n < 8 {
        for i in 0..n {
            y[i] = y[i] - alpha * x[i];
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let y = unsafe { &mut *(y as *mut [T] as *mut [f64]) };
            let a = unsafe { *(&alpha as *const T as *const f64) };
            let x = unsafe { &*(x as *const [T] as *const [f64]) };
            f64_neon::axpy_neg(y, a, x);
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let y = unsafe { &mut *(y as *mut [T] as *mut [f32]) };
            let a = unsafe { *(&alpha as *const T as *const f32) };
            let x = unsafe { &*(x as *const [T] as *const [f32]) };
            f32_neon::axpy_neg(y, a, x);
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let y = unsafe { &mut *(y as *mut [T] as *mut [f64]) };
            let a = unsafe { *(&alpha as *const T as *const f64) };
            let x = unsafe { &*(x as *const [T] as *const [f64]) };
            #[cfg(target_feature = "avx512f")]
            f64_avx512::axpy_neg(y, a, x);
            #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
            f64_avx::axpy_neg(y, a, x);
            #[cfg(not(target_feature = "avx"))]
            f64_sse2::axpy_neg(y, a, x);
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let y = unsafe { &mut *(y as *mut [T] as *mut [f32]) };
            let a = unsafe { *(&alpha as *const T as *const f32) };
            let x = unsafe { &*(x as *const [T] as *const [f32]) };
            #[cfg(target_feature = "avx512f")]
            f32_avx512::axpy_neg(y, a, x);
            #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
            f32_avx::axpy_neg(y, a, x);
            #[cfg(not(target_feature = "avx"))]
            f32_sse2::axpy_neg(y, a, x);
            return;
        }
    }
    scalar::axpy_neg(y, alpha, x);
}

/// Dispatch scalar multiplication to SIMD or scalar fallback.
#[inline]
pub(crate) fn scale_slices_dispatch<T: Scalar>(a: &[T], scalar: T, out: &mut [T]) {
    #[cfg(target_arch = "aarch64")]
    {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let a = unsafe { &*(a as *const [T] as *const [f64]) };
            let s = unsafe { *(&scalar as *const T as *const f64) };
            let out = unsafe { &mut *(out as *mut [T] as *mut [f64]) };
            f64_neon::scale_slices(a, s, out);
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let a = unsafe { &*(a as *const [T] as *const [f32]) };
            let s = unsafe { *(&scalar as *const T as *const f32) };
            let out = unsafe { &mut *(out as *mut [T] as *mut [f32]) };
            f32_neon::scale_slices(a, s, out);
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let a = unsafe { &*(a as *const [T] as *const [f64]) };
            let s = unsafe { *(&scalar as *const T as *const f64) };
            let out = unsafe { &mut *(out as *mut [T] as *mut [f64]) };
            #[cfg(target_feature = "avx512f")]
            f64_avx512::scale_slices(a, s, out);
            #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
            f64_avx::scale_slices(a, s, out);
            #[cfg(not(target_feature = "avx"))]
            f64_sse2::scale_slices(a, s, out);
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let a = unsafe { &*(a as *const [T] as *const [f32]) };
            let s = unsafe { *(&scalar as *const T as *const f32) };
            let out = unsafe { &mut *(out as *mut [T] as *mut [f32]) };
            #[cfg(target_feature = "avx512f")]
            f32_avx512::scale_slices(a, s, out);
            #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
            f32_avx::scale_slices(a, s, out);
            #[cfg(not(target_feature = "avx"))]
            f32_sse2::scale_slices(a, s, out);
            return;
        }
    }
    scalar::scale_slices(a, scalar, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Dot product boundary tests ─────────────────────────────────

    #[test]
    fn dot_f64_boundary_lengths() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let a: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
            let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64 * 0.5).collect();
            let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let result = dot_dispatch(&a, &b);
            assert!(
                (result - expected).abs() < 1e-10,
                "dot f64 n={n}: got {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn dot_f32_boundary_lengths() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let a: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
            let b: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.5).collect();
            let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let result = dot_dispatch(&a, &b);
            assert!(
                (result - expected).abs() < 1e-4,
                "dot f32 n={n}: got {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn dot_integer_fallback() {
        let a = vec![1_i32, 2, 3, 4, 5];
        let b = vec![6_i32, 7, 8, 9, 10];
        let result = dot_dispatch(&a, &b);
        assert_eq!(result, 1 * 6 + 2 * 7 + 3 * 8 + 4 * 9 + 5 * 10);
    }

    // ── Matmul boundary tests ──────────────────────────────────────

    #[test]
    fn matmul_f64_boundary_sizes() {
        for size in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let n = size;
            let a: Vec<f64> = (0..n * n).map(|i| (i + 1) as f64).collect();
            let b: Vec<f64> = (0..n * n).map(|i| (i + 1) as f64 * 0.1).collect();
            let mut c = vec![0.0_f64; n * n];
            let mut c_ref = vec![0.0_f64; n * n];

            matmul_dispatch(&a, &b, &mut c, n, n, n);
            scalar::matmul(&a, &b, &mut c_ref, n, n, n);

            for i in 0..n * n {
                assert!(
                    (c[i] - c_ref[i]).abs() < 1e-8,
                    "matmul f64 n={n} idx={i}: got {}, expected {}",
                    c[i],
                    c_ref[i]
                );
            }
        }
    }

    #[test]
    fn matmul_f32_boundary_sizes() {
        for size in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let n = size;
            let a: Vec<f32> = (0..n * n).map(|i| (i + 1) as f32).collect();
            let b: Vec<f32> = (0..n * n).map(|i| (i + 1) as f32 * 0.1).collect();
            let mut c = vec![0.0_f32; n * n];
            let mut c_ref = vec![0.0_f32; n * n];

            matmul_dispatch(&a, &b, &mut c, n, n, n);
            scalar::matmul(&a, &b, &mut c_ref, n, n, n);

            for i in 0..n * n {
                assert!(
                    (c[i] - c_ref[i]).abs() < 1e-2,
                    "matmul f32 n={n} idx={i}: got {}, expected {}",
                    c[i],
                    c_ref[i]
                );
            }
        }
    }

    #[test]
    fn matmul_non_square_f64() {
        // (3×5) * (5×7) → (3×7)
        let m = 3;
        let n = 5;
        let p = 7;
        let a: Vec<f64> = (0..m * n).map(|i| (i + 1) as f64).collect();
        let b: Vec<f64> = (0..n * p).map(|i| (i + 1) as f64 * 0.1).collect();
        let mut c = vec![0.0_f64; m * p];
        let mut c_ref = vec![0.0_f64; m * p];

        matmul_dispatch(&a, &b, &mut c, m, n, p);
        scalar::matmul(&a, &b, &mut c_ref, m, n, p);

        for i in 0..m * p {
            assert!(
                (c[i] - c_ref[i]).abs() < 1e-10,
                "matmul non-square idx={i}: got {}, expected {}",
                c[i],
                c_ref[i]
            );
        }
    }

    #[test]
    fn matmul_integer_fallback() {
        // Column-major 2×2: A=[[1,2],[3,4]] stored as [1,3,2,4]
        // B=[[5,6],[7,8]] stored as [5,7,6,8]
        // C=A*B=[[19,22],[43,50]] stored as [19,43,22,50]
        let a = vec![1_i32, 3, 2, 4];
        let b = vec![5_i32, 7, 6, 8];
        let mut c = vec![0_i32; 4];
        matmul_dispatch(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, vec![19, 43, 22, 50]);
    }

    // ── Element-wise ops boundary tests ────────────────────────────

    #[test]
    fn add_slices_f64_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let a: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let b: Vec<f64> = (0..n).map(|i| (i * 10) as f64).collect();
            let mut out = vec![0.0_f64; n];

            add_slices_dispatch(&a, &b, &mut out);

            for i in 0..n {
                assert_eq!(out[i], a[i] + b[i], "add f64 n={n} idx={i}");
            }
        }
    }

    #[test]
    fn sub_slices_f64_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let a: Vec<f64> = (0..n).map(|i| (i * 10) as f64).collect();
            let b: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let mut out = vec![0.0_f64; n];

            sub_slices_dispatch(&a, &b, &mut out);

            for i in 0..n {
                assert_eq!(out[i], a[i] - b[i], "sub f64 n={n} idx={i}");
            }
        }
    }

    #[test]
    fn scale_slices_f64_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let a: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
            let mut out = vec![0.0_f64; n];

            scale_slices_dispatch(&a, 3.0, &mut out);

            for i in 0..n {
                assert_eq!(out[i], a[i] * 3.0, "scale f64 n={n} idx={i}");
            }
        }
    }

    #[test]
    fn add_slices_f32_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..n).map(|i| (i * 10) as f32).collect();
            let mut out = vec![0.0_f32; n];

            add_slices_dispatch(&a, &b, &mut out);

            for i in 0..n {
                assert_eq!(out[i], a[i] + b[i], "add f32 n={n} idx={i}");
            }
        }
    }

    #[test]
    fn sub_slices_f32_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let a: Vec<f32> = (0..n).map(|i| (i * 10) as f32).collect();
            let b: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let mut out = vec![0.0_f32; n];

            sub_slices_dispatch(&a, &b, &mut out);

            for i in 0..n {
                assert_eq!(out[i], a[i] - b[i], "sub f32 n={n} idx={i}");
            }
        }
    }

    #[test]
    fn scale_slices_f32_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let a: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
            let mut out = vec![0.0_f32; n];

            scale_slices_dispatch(&a, 3.0_f32, &mut out);

            for i in 0..n {
                assert_eq!(out[i], a[i] * 3.0, "scale f32 n={n} idx={i}");
            }
        }
    }

    // ── AXPY boundary tests ───────────────────────────────────────────

    #[test]
    fn axpy_neg_f64_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
            let alpha = 2.5_f64;
            let mut y: Vec<f64> = (0..n).map(|i| (i * 10) as f64).collect();
            let expected: Vec<f64> = y.iter().zip(x.iter()).map(|(yi, xi)| yi - alpha * xi).collect();

            axpy_neg_dispatch(&mut y, alpha, &x);

            for i in 0..n {
                assert!(
                    (y[i] - expected[i]).abs() < 1e-10,
                    "axpy f64 n={n} idx={i}: got {}, expected {}",
                    y[i], expected[i]
                );
            }
        }
    }

    #[test]
    fn axpy_neg_f32_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let x: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
            let alpha = 2.5_f32;
            let mut y: Vec<f32> = (0..n).map(|i| (i * 10) as f32).collect();
            let expected: Vec<f32> = y.iter().zip(x.iter()).map(|(yi, xi)| yi - alpha * xi).collect();

            axpy_neg_dispatch(&mut y, alpha, &x);

            for i in 0..n {
                assert!(
                    (y[i] - expected[i]).abs() < 1e-4,
                    "axpy f32 n={n} idx={i}: got {}, expected {}",
                    y[i], expected[i]
                );
            }
        }
    }

    #[test]
    fn axpy_neg_integer_fallback() {
        let x = vec![1_i32, 2, 3, 4, 5];
        let mut y = vec![10_i32, 20, 30, 40, 50];
        axpy_neg_dispatch(&mut y, 3, &x);
        assert_eq!(y, vec![7, 14, 21, 28, 35]);
    }
}
