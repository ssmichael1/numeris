//! FFT-based linear convolution and correlation of real signals (requires `alloc`).

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use num_complex::Complex;

use super::dynfft::DynFft;
use crate::traits::FloatScalar;

/// Linear convolution `a ∗ b` via FFT.
///
/// Returns a vector of length `a.len() + b.len() − 1`. Empty if either input is
/// empty. For small kernels a direct O(n·m) convolution is faster; this pays off
/// once both operands are large.
pub fn fft_convolve<T: FloatScalar>(a: &[T], b: &[T]) -> Vec<T> {
    fft_convolve_impl(a, b, false)
}

/// Linear cross-correlation of `a` and `b` via FFT.
///
/// Defined as the convolution of `a` with the time-reversed `b`, so lag `k` of
/// the result is `Σ_n a[n] · b[n − (k − (b.len()−1))]`. Length
/// `a.len() + b.len() − 1`; empty if either input is empty.
pub fn fft_correlate<T: FloatScalar>(a: &[T], b: &[T]) -> Vec<T> {
    fft_convolve_impl(a, b, true)
}

fn fft_convolve_impl<T: FloatScalar>(a: &[T], b: &[T], reverse_b: bool) -> Vec<T> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let out_len = a.len() + b.len() - 1;

    let zero = Complex::new(T::zero(), T::zero());
    let mut fa = vec![zero; out_len];
    let mut fb = vec![zero; out_len];
    for (dst, &x) in fa.iter_mut().zip(a) {
        dst.re = x;
    }
    if reverse_b {
        for (dst, &x) in fb.iter_mut().zip(b.iter().rev()) {
            dst.re = x;
        }
    } else {
        for (dst, &x) in fb.iter_mut().zip(b) {
            dst.re = x;
        }
    }

    let mut plan = DynFft::new(out_len);
    plan.forward(&mut fa);
    plan.forward(&mut fb);
    for (x, y) in fa.iter_mut().zip(&fb) {
        *x = *x * *y;
    }
    plan.inverse(&mut fa);

    fa.iter().map(|z| z.re).collect()
}
