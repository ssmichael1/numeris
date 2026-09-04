//! Real-input FFTs (`rfft` / `irfft`).
//!
//! A length-`N` real signal has a Hermitian-symmetric spectrum, so only the
//! `N/2 + 1` non-redundant bins (DC … Nyquist) are computed and returned.
//!
//! Both directions use the classic half-size packing trick. Forward: pack the
//! `N` reals into `N/2` complex samples `z[j] = x[2j] + i·x[2j+1]`, run one
//! length-`N/2` complex FFT, then untangle the even/odd sub-spectra. Inverse:
//! re-tangle the `N/2 + 1` bins into the length-`N/2` spectrum of `z`, run one
//! length-`N/2` inverse FFT, and unpack. Each direction is roughly half the work
//! of the corresponding full complex transform.
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

/// Inverse of [`untangle`]: rebuild the length-`m` spectrum `z[0..m]` of the
/// packed sequence `x[2j] + i·x[2j+1]` from the `m + 1` real-spectrum bins,
/// where `n = 2m`.
///
/// With `Fe`/`Fo` the even/odd sub-spectra and `w = exp(-2πi k/N)`, the forward
/// gives `X[k] = Fe + w·Fo` and `conj(X[m−k]) = Fe − w·Fo`, so
/// `Fe = (X[k] + conj(X[m−k]))/2`, `Fo = conj(w)·(X[k] − conj(X[m−k]))/2`, and
/// `Z[k] = Fe + i·Fo`.
fn retangle<T: FloatScalar>(bins: &[Complex<T>], z: &mut [Complex<T>], n: usize) {
    let m = z.len();
    debug_assert_eq!(n, 2 * m);
    debug_assert_eq!(bins.len(), m + 1);

    let half = cast::<T>(0.5);
    for (k, slot) in z.iter_mut().enumerate() {
        let xf = bins[k];
        let xb = bins[m - k].conj();
        let fe = (xf + xb) * half;
        let d = (xf - xb) * half;
        let ang = cast::<T>(core::f64::consts::TAU * (k as f64) / (n as f64));
        let wc = Complex::new(ang.cos(), ang.sin()); // conj(w) = exp(+2πi k / N)
        let fo = d * wc;
        // Z = Fe + i·Fo
        *slot = Complex::new(fe.re - fo.im, fe.im + fo.re);
    }
}

/// Pack `n` reals into `n/2` complex samples `z[j] = x[2j] + i·x[2j+1]`.
#[inline]
fn pack<T: FloatScalar>(input: &[T], z: &mut [Complex<T>]) {
    for (slot, pair) in z.iter_mut().zip(input.chunks_exact(2)) {
        *slot = Complex::new(pair[0], pair[1]);
    }
}

/// Unpack `n/2` complex samples back into `n` reals (inverse of [`pack`]).
#[inline]
fn unpack<T: FloatScalar>(z: &[Complex<T>], output: &mut [T]) {
    for (pair, s) in output.chunks_exact_mut(2).zip(z) {
        pair[0] = s.re;
        pair[1] = s.im;
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
    pack(input, &mut z[..m]);
    radix2_inline(&mut z[..m], false);
    untangle(&z[..m], output, N);
}

/// Inverse real FFT (fixed size). `input` is the `N/2 + 1` complex bins;
/// `output` receives the `N` reconstructed real samples (normalized by `1/N`).
///
/// `N` must be a power of two with `2 ≤ N ≤ 4096`, and `input.len()` must equal
/// `N/2 + 1`. Runs a single length-`N/2` inverse FFT (half-size packing).
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

    let m = N / 2;
    let mut z = [Complex::new(T::zero(), T::zero()); N];
    retangle(input, &mut z[..m], N);
    radix2_inline(&mut z[..m], true);
    unpack(&z[..m], output);
}

#[cfg(feature = "alloc")]
pub use dyn_real::{DynRealFft, DynRealFftScratch};

#[cfg(feature = "alloc")]
mod dyn_real {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    use num_complex::Complex;

    use super::super::dynfft::{DynFft, DynFftScratch};
    use super::{pack, retangle, unpack, untangle};
    use crate::traits::FloatScalar;

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

    /// Work buffers for a [`DynRealFft`] plan (requires `alloc`).
    ///
    /// The real-FFT analogue of [`DynFftScratch`]: build one per worker with
    /// [`DynRealFft::make_scratch`] to run a shared plan through
    /// [`DynRealFft::forward_with`] / [`DynRealFft::inverse_with`].
    pub struct DynRealFftScratch<T: FloatScalar> {
        n: usize,
        /// Packed (even `n`: length `n/2`) or full (odd `n`: length `n`) complex
        /// work buffer.
        z: Vec<Complex<T>>,
        fft: DynFftScratch<T>,
    }

    impl<T: FloatScalar> DynRealFftScratch<T> {
        /// The real signal length this scratch was built for.
        #[inline]
        pub fn len(&self) -> usize {
            self.n
        }

        /// Always `false` — a scratch cannot be built for length zero.
        #[inline]
        pub fn is_empty(&self) -> bool {
            false
        }
    }

    /// Runtime-sized real FFT (requires `alloc`).
    ///
    /// Handles any signal length `n`. Even lengths use the half-size packing
    /// trick in both directions (a single length-`n/2` complex plan); odd
    /// lengths run a full length-`n` complex FFT and slice off / rebuild the
    /// non-redundant bins.
    ///
    /// Like [`DynFft`], the plan is read-only during a transform: use
    /// [`make_scratch`](DynRealFft::make_scratch) with
    /// [`forward_with`](DynRealFft::forward_with) /
    /// [`inverse_with`](DynRealFft::inverse_with) to share one plan across
    /// workers.
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
        /// Length-`n/2` plan for even `n`, length-`n` plan for odd `n`.
        plan: DynFft<T>,
        scratch: DynRealFftScratch<T>,
    }

    impl<T: FloatScalar> DynRealFft<T> {
        /// Build a plan for real signals of length `len`. Panics if `len == 0`.
        pub fn new(len: usize) -> Self {
            assert!(len > 0, "DynRealFft length must be non-zero");
            let plan_len = if len % 2 == 0 { len / 2 } else { len };
            let plan = DynFft::new(plan_len);
            let scratch = Self::scratch_for(&plan, len);
            Self {
                n: len,
                plan,
                scratch,
            }
        }

        fn scratch_for(plan: &DynFft<T>, n: usize) -> DynRealFftScratch<T> {
            DynRealFftScratch {
                n,
                z: vec![Complex::new(T::zero(), T::zero()); plan.len()],
                fft: plan.make_scratch(),
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

        /// Allocate a fresh scratch buffer for this plan.
        pub fn make_scratch(&self) -> DynRealFftScratch<T> {
            Self::scratch_for(&self.plan, self.n)
        }

        /// Forward real FFT. `input.len() == n`, `output.len() == n/2 + 1`.
        pub fn forward(&mut self, input: &[T], output: &mut [Complex<T>]) {
            let Self { n, plan, scratch } = self;
            Self::forward_impl(plan, *n, input, output, scratch);
        }

        /// Inverse real FFT. `input.len() == n/2 + 1`, `output.len() == n`.
        pub fn inverse(&mut self, input: &[Complex<T>], output: &mut [T]) {
            let Self { n, plan, scratch } = self;
            Self::inverse_impl(plan, *n, input, output, scratch);
        }

        /// Forward real FFT through a caller-owned scratch, leaving the plan
        /// shared. Panics on a length mismatch or a scratch built for another
        /// length.
        pub fn forward_with(
            &self,
            input: &[T],
            output: &mut [Complex<T>],
            scratch: &mut DynRealFftScratch<T>,
        ) {
            Self::forward_impl(&self.plan, self.n, input, output, scratch);
        }

        /// Inverse real FFT through a caller-owned scratch, leaving the plan
        /// shared. Panics on a length mismatch or a scratch built for another
        /// length.
        pub fn inverse_with(
            &self,
            input: &[Complex<T>],
            output: &mut [T],
            scratch: &mut DynRealFftScratch<T>,
        ) {
            Self::inverse_impl(&self.plan, self.n, input, output, scratch);
        }

        fn forward_impl(
            plan: &DynFft<T>,
            n: usize,
            input: &[T],
            output: &mut [Complex<T>],
            scratch: &mut DynRealFftScratch<T>,
        ) {
            assert_eq!(input.len(), n, "DynRealFft: input length must equal n");
            assert_eq!(
                output.len(),
                n / 2 + 1,
                "DynRealFft: output length must be n/2 + 1"
            );
            assert_eq!(
                scratch.n, n,
                "DynRealFft: scratch was built for a different length"
            );
            let DynRealFftScratch { z, fft, .. } = scratch;

            if n % 2 == 0 {
                pack(input, z);
                plan.forward_with(z, fft);
                untangle(z, output, n);
            } else {
                for (slot, &x) in z.iter_mut().zip(input) {
                    *slot = Complex::new(x, T::zero());
                }
                plan.forward_with(z, fft);
                output.copy_from_slice(&z[..n / 2 + 1]);
            }
        }

        fn inverse_impl(
            plan: &DynFft<T>,
            n: usize,
            input: &[Complex<T>],
            output: &mut [T],
            scratch: &mut DynRealFftScratch<T>,
        ) {
            assert_eq!(
                input.len(),
                n / 2 + 1,
                "DynRealFft: input length must be n/2 + 1"
            );
            assert_eq!(output.len(), n, "DynRealFft: output length must equal n");
            assert_eq!(
                scratch.n, n,
                "DynRealFft: scratch was built for a different length"
            );
            let DynRealFftScratch { z, fft, .. } = scratch;

            if n % 2 == 0 {
                retangle(input, z, n);
                plan.inverse_with(z, fft);
                unpack(z, output);
            } else {
                hermitian_fill(input, z);
                plan.inverse_with(z, fft);
                for (o, f) in output.iter_mut().zip(z.iter()) {
                    *o = f.re;
                }
            }
        }
    }
}
