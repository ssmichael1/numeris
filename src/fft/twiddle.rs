//! Precomputed twiddle factors for the fixed-size FFT.

use num_complex::Complex;

use super::cast;
use crate::traits::FloatScalar;

/// Precomputed twiddle factors `exp(-2πi k / N)` for repeated same-`N` transforms.
///
/// Build once, reuse across many [`fft`](super::fft) / [`ifft`](super::ifft)
/// calls so no `sin`/`cos` runs in the transform loop — the path to prefer on
/// embedded targets that repeat a fixed transform size.
///
/// Storage is `[Complex<T>; N]` (only the first `N/2` entries are used). Full-`N`
/// storage keeps the type on stable Rust: `[Complex<T>; N/2]` would require the
/// unstable `generic_const_exprs` feature, which the crate deliberately never
/// enables. Consequently `TwiddleTable<f64, 4096>` occupies 64 KiB — fine on
/// capable targets; memory-constrained callers should use
/// [`fft_inplace`](super::fft_inplace) (no persistent table) or a smaller `N`.
///
/// `N` must be a power of two and `≤ 4096`, checked at compile time.
pub struct TwiddleTable<T: FloatScalar, const N: usize> {
    pub(crate) factors: [Complex<T>; N],
}

impl<T: FloatScalar, const N: usize> TwiddleTable<T, N> {
    /// Compute the twiddle factors for length `N`.
    ///
    /// Runs `N` `sin`/`cos` evaluations once. Panics at compile time unless `N`
    /// is a power of two and `N ≤ 4096`.
    pub fn new() -> Self {
        const {
            assert!(
                N.is_power_of_two() && N <= 4096,
                "fft: N must be a power of two and <= 4096",
            );
        }
        let two_pi = cast::<T>(core::f64::consts::TAU);
        let n_t = cast::<T>(N as f64);
        let factors = core::array::from_fn(|k| {
            // w[k] = exp(-2πi k / N)
            let theta = -two_pi * cast::<T>(k as f64) / n_t;
            Complex::new(theta.cos(), theta.sin())
        });
        Self { factors }
    }

    /// The `N/2` twiddle factors actually consumed by the butterfly passes.
    #[inline]
    pub(crate) fn as_slice(&self) -> &[Complex<T>] {
        &self.factors[..N / 2]
    }
}

impl<T: FloatScalar, const N: usize> Default for TwiddleTable<T, N> {
    fn default() -> Self {
        Self::new()
    }
}
