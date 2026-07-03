//! Real-input FFTs (`rfft` / `irfft`).
//!
//! A length-`N` real signal has a Hermitian-symmetric spectrum, so only the
//! `N/2 + 1` non-redundant bins (DC … Nyquist) are computed and returned.
//!
//! The forward fixed-size transform uses the classic half-size packing trick:
//! pack the `N` reals into `N/2` complex samples, run one length-`N/2` complex
//! FFT, then untangle — roughly half the work of a full complex FFT. The inverse
//! reconstructs the full Hermitian spectrum and runs a length-`N` inverse FFT
//! (the simpler, correct choice; a half-size inverse is a future optimization).
//!
//! Following the crate convention, the output length is passed by the caller as
//! a slice (returning `[_; N/2 + 1]` would require the unstable
//! `generic_const_exprs`), checked with an `assert_eq!`.

use num_complex::Complex;

use super::cast;
use super::radix::radix2_inline;
use crate::traits::FloatScalar;

/// Untangle a length-`m` complex spectrum `z = FFT(packed even/odd)` into the
/// `m + 1` real-spectrum bins `out[0..=m]`, where `n = 2m`.
fn untangle<T: FloatScalar>(z: &[Complex<T>], out: &mut [Complex<T>], n: usize) {
    let m = z.len();
    debug_assert_eq!(n, 2 * m);
    debug_assert_eq!(out.len(), m + 1);

    let half = cast::<T>(0.5);
    // DC and Nyquist are purely real.
    out[0] = Complex::new(z[0].re + z[0].im, T::zero());
    out[m] = Complex::new(z[0].re - z[0].im, T::zero());

    for k in 1..m {
        let cf = z[k];
        let cb = z[m - k].conj();
        // Even/odd sub-spectra of the real signal.
        let fe = (cf + cb) * half; // (Z[k] + conj(Z[m-k])) / 2
        let d = (cf - cb) * half; // (Z[k] - conj(Z[m-k])) / 2
        let fo = Complex::new(d.im, -d.re); // divide by i  ->  fo = d * (-i)
        let ang = cast::<T>(-core::f64::consts::TAU * (k as f64) / (n as f64));
        let w = Complex::new(ang.cos(), ang.sin()); // exp(-2πi k / N)
        out[k] = fe + w * fo;
    }
}

/// Rebuild the full length-`n` Hermitian spectrum from its `n/2 + 1` bins.
fn hermitian_fill<T: FloatScalar>(bins: &[Complex<T>], full: &mut [Complex<T>]) {
    let n = full.len();
    let half = n / 2;
    for (k, slot) in full.iter_mut().enumerate() {
        *slot = if k <= half {
            bins[k]
        } else {
            bins[n - k].conj()
        };
    }
}

/// Forward real FFT (fixed size). `input` is `N` real samples; `output` receives
/// the `N/2 + 1` complex bins (DC … Nyquist).
///
/// `N` must be a power of two with `2 ≤ N ≤ 4096` (checked at compile time), and
/// `output.len()` must equal `N/2 + 1`.
pub fn rfft<T: FloatScalar, const N: usize>(input: &[T; N], output: &mut [Complex<T>]) {
    const {
        assert!(
            N.is_power_of_two() && N >= 2 && N <= 4096,
            "rfft: N must be a power of two in 2..=4096",
        );
    }
    assert_eq!(
        output.len(),
        N / 2 + 1,
        "rfft: output length must be N/2 + 1"
    );

    let m = N / 2;
    let mut z = [Complex::new(T::zero(), T::zero()); N];
    for j in 0..m {
        z[j] = Complex::new(input[2 * j], input[2 * j + 1]);
    }
    radix2_inline(&mut z[..m], false);
    untangle(&z[..m], output, N);
}

/// Inverse real FFT (fixed size). `input` is the `N/2 + 1` complex bins;
/// `output` receives the `N` reconstructed real samples (normalized by `1/N`).
///
/// `N` must be a power of two with `2 ≤ N ≤ 4096`, and `input.len()` must equal
/// `N/2 + 1`.
pub fn irfft<T: FloatScalar, const N: usize>(input: &[Complex<T>], output: &mut [T; N]) {
    const {
        assert!(
            N.is_power_of_two() && N >= 2 && N <= 4096,
            "irfft: N must be a power of two in 2..=4096",
        );
    }
    assert_eq!(
        input.len(),
        N / 2 + 1,
        "irfft: input length must be N/2 + 1"
    );

    let mut full = [Complex::new(T::zero(), T::zero()); N];
    hermitian_fill(input, &mut full);
    radix2_inline(&mut full, true);
    for (o, f) in output.iter_mut().zip(&full) {
        *o = f.re;
    }
}

#[cfg(feature = "alloc")]
pub use dyn_real::DynRealFft;

#[cfg(feature = "alloc")]
mod dyn_real {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    use num_complex::Complex;

    use super::super::dynfft::DynFft;
    use super::{hermitian_fill, untangle};
    use crate::traits::FloatScalar;

    /// Runtime-sized real FFT (requires `alloc`).
    ///
    /// Handles any signal length `n`. Even lengths use the half-size packing
    /// trick (a length-`n/2` complex plan); odd lengths run a full length-`n`
    /// complex FFT and slice off the non-redundant bins. The inverse always
    /// reconstructs the Hermitian spectrum and runs a length-`n` inverse FFT.
    ///
    /// # Examples
    ///
    /// ```
    /// # use numeris::fft::DynRealFft;
    /// # use numeris::Complex;
    /// let signal = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0];
    /// let mut plan = DynRealFft::<f64>::new(signal.len());
    /// let mut spec = vec![Complex::new(0.0, 0.0); signal.len() / 2 + 1];
    /// plan.forward(&signal, &mut spec);
    /// let mut out = [0.0; 6];
    /// plan.inverse(&spec, &mut out);
    /// assert!((out[0] - 1.0).abs() < 1e-10);
    /// ```
    pub struct DynRealFft<T: FloatScalar> {
        n: usize,
        /// Half-size complex plan for even-length forward transforms.
        half_plan: Option<DynFft<T>>,
        /// Full-size complex plan (odd-length forward and every inverse).
        full_plan: DynFft<T>,
        scratch: Vec<Complex<T>>,
    }

    impl<T: FloatScalar> DynRealFft<T> {
        /// Build a plan for real signals of length `len`. Panics if `len == 0`.
        pub fn new(len: usize) -> Self {
            assert!(len > 0, "DynRealFft length must be non-zero");
            let half_plan = if len % 2 == 0 {
                Some(DynFft::new(len / 2))
            } else {
                None
            };
            Self {
                n: len,
                half_plan,
                full_plan: DynFft::new(len),
                scratch: vec![Complex::new(T::zero(), T::zero()); len],
            }
        }

        /// The real signal length this plan was built for.
        #[inline]
        pub fn len(&self) -> usize {
            self.n
        }

        /// Returns `true` if the signal length is zero. Always `false`.
        #[inline]
        pub fn is_empty(&self) -> bool {
            false
        }

        /// Forward real FFT. `input.len() == n`, `output.len() == n/2 + 1`.
        pub fn forward(&mut self, input: &[T], output: &mut [Complex<T>]) {
            assert_eq!(input.len(), self.n, "DynRealFft: input length must equal n");
            assert_eq!(
                output.len(),
                self.n / 2 + 1,
                "DynRealFft: output length must be n/2 + 1"
            );

            match &mut self.half_plan {
                Some(half) => {
                    let m = self.n / 2;
                    let z = &mut self.scratch[..m];
                    for (j, slot) in z.iter_mut().enumerate() {
                        *slot = Complex::new(input[2 * j], input[2 * j + 1]);
                    }
                    half.forward(z);
                    untangle(&self.scratch[..m], output, self.n);
                }
                None => {
                    for (slot, &x) in self.scratch.iter_mut().zip(input) {
                        *slot = Complex::new(x, T::zero());
                    }
                    self.full_plan.forward(&mut self.scratch);
                    output.copy_from_slice(&self.scratch[..self.n / 2 + 1]);
                }
            }
        }

        /// Inverse real FFT. `input.len() == n/2 + 1`, `output.len() == n`.
        pub fn inverse(&mut self, input: &[Complex<T>], output: &mut [T]) {
            assert_eq!(
                input.len(),
                self.n / 2 + 1,
                "DynRealFft: input length must be n/2 + 1"
            );
            assert_eq!(
                output.len(),
                self.n,
                "DynRealFft: output length must equal n"
            );

            hermitian_fill(input, &mut self.scratch);
            self.full_plan.inverse(&mut self.scratch);
            for (o, f) in output.iter_mut().zip(&self.scratch) {
                *o = f.re;
            }
        }
    }
}
