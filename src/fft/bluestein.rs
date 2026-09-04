//! Bluestein's algorithm (chirp-z transform) for arbitrary transform lengths.
//!
//! Reduces a length-`n` DFT — for *any* `n`, including primes — to a linear
//! convolution evaluated with power-of-two FFTs, reusing the SIMD radix core in
//! [`super::soa`]. This is what lets [`DynFft`](super::DynFft) accept awkward
//! lengths without dedicated radix-3/5 butterflies.
//!
//! Using `jk = (j² + k² − (k−j)²)/2` and `w = exp(-πi/n)` (so `w^{2jk} =
//! exp(-2πi jk/n)`), the forward DFT becomes
//!
//! ```text
//! X[k] = chirp[k] · Σ_j (x[j]·chirp[j]) · g[k−j],   g[d] = conj(chirp[|d|])
//! ```
//!
//! where `chirp[j] = w^{j²}` and the filter `g` is the two-sided `conj(chirp)`.
//! The sum is a convolution, computed as `IFFT(FFT(a) · FFT(g))` at a
//! power-of-two length `m ≥ 2n − 1`.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use num_complex::Complex;

use super::cast;
use super::soa::{SoaPlan, SoaScratch};
use crate::traits::FloatScalar;

/// Precomputed Bluestein plan for a fixed length `n` (not a power of two).
/// Read-only during a transform; pair with a [`BluesteinScratch`].
pub(crate) struct Bluestein<T: FloatScalar> {
    n: usize,
    /// `chirp[j] = exp(-πi j² / n)`, `j = 0..n`.
    chirp: Vec<Complex<T>>,
    /// `FFT_m` of the two-sided filter `conj(chirp)`.
    filter_fft: Vec<Complex<T>>,
    /// Length-`m` power-of-two plan for the convolution.
    inner: SoaPlan<T>,
}

/// Work buffers for a [`Bluestein`] plan: the length-`m` convolution buffer
/// plus the inner SoA scratch.
pub(crate) struct BluesteinScratch<T: FloatScalar> {
    work: Vec<Complex<T>>,
    soa: SoaScratch<T>,
}

impl<T: FloatScalar> BluesteinScratch<T> {
    /// The padded convolution length `m` this scratch was built for.
    #[inline]
    pub(crate) fn padded_len(&self) -> usize {
        self.work.len()
    }
}

impl<T: FloatScalar> Bluestein<T> {
    /// Build the plan for length `n`. `n` need not be a power of two.
    pub(crate) fn new(n: usize) -> Self {
        debug_assert!(n >= 2);
        let m = (2 * n - 1).next_power_of_two();
        let inner = SoaPlan::new(m);

        // chirp[j] = exp(-πi j² / n). Reduce j² mod 2n before the divide to keep
        // the angle small and accurate for large n (j² grows quadratically).
        let mut chirp = Vec::with_capacity(n);
        for j in 0..n {
            let j2 = ((j as u128 * j as u128) % (2 * n as u128)) as f64;
            let angle = cast::<T>(-core::f64::consts::PI * j2 / n as f64);
            chirp.push(Complex::new(angle.cos(), angle.sin()));
        }

        // Two-sided filter b[k] = conj(chirp[|k|]) placed at indices 0 and, for
        // the negative lobe, m − j. b is symmetric because (−j)² = j².
        let zero = Complex::new(T::zero(), T::zero());
        let mut filter = vec![zero; m];
        filter[0] = chirp[0].conj();
        for j in 1..n {
            let v = chirp[j].conj();
            filter[j] = v;
            filter[m - j] = v;
        }
        let mut soa = inner.scratch();
        inner.transform(&mut filter, &mut soa, false);

        Self {
            n,
            chirp,
            filter_fft: filter,
            inner,
        }
    }

    /// Padded convolution length `m`.
    #[inline]
    pub(crate) fn padded_len(&self) -> usize {
        self.filter_fft.len()
    }

    /// Allocate a scratch buffer sized for this plan.
    pub(crate) fn scratch(&self) -> BluesteinScratch<T> {
        BluesteinScratch {
            work: vec![Complex::new(T::zero(), T::zero()); self.padded_len()],
            soa: self.inner.scratch(),
        }
    }

    /// Transform `buf` in place (`buf.len() == n`). `inverse` computes
    /// `conj(fft(conj(x))) / n`, with the conjugations folded into the chirp
    /// pre-/post-multiplies.
    pub(crate) fn transform(
        &self,
        buf: &mut [Complex<T>],
        scratch: &mut BluesteinScratch<T>,
        inverse: bool,
    ) {
        debug_assert_eq!(buf.len(), self.n);
        debug_assert_eq!(scratch.padded_len(), self.padded_len());
        let zero = Complex::new(T::zero(), T::zero());
        let BluesteinScratch { work: a, soa } = scratch;

        // a[j] = x[j] · chirp[j] (conj(x[j]) for the inverse), zero-padded to m.
        for ((slot, &x), &c) in a.iter_mut().zip(buf.iter()).zip(&self.chirp) {
            *slot = if inverse { x.conj() * c } else { x * c };
        }
        for slot in a.iter_mut().skip(self.n) {
            *slot = zero;
        }

        // conv = IFFT(FFT(a) · FFT(b))
        self.inner.transform(a, soa, false);
        for (ai, bi) in a.iter_mut().zip(&self.filter_fft) {
            *ai = *ai * *bi;
        }
        self.inner.transform(a, soa, true);

        // X[k] = chirp[k] · conv[k]; inverse: conj(·) / n.
        if inverse {
            let inv_n = T::one() / cast::<T>(self.n as f64);
            for ((out, &v), &c) in buf.iter_mut().zip(a.iter()).zip(&self.chirp) {
                let y = (v * c).conj();
                *out = Complex::new(y.re * inv_n, y.im * inv_n);
            }
        } else {
            for ((out, &v), &c) in buf.iter_mut().zip(a.iter()).zip(&self.chirp) {
                *out = v * c;
            }
        }
    }
}
