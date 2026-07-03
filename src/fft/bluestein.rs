//! Bluestein's algorithm (chirp-z transform) for arbitrary transform lengths.
//!
//! Reduces a length-`n` DFT — for *any* `n`, including primes — to a linear
//! convolution evaluated with power-of-two FFTs, reusing the radix core in
//! [`super::radix`]. This is what lets [`DynFft`](super::DynFft) accept awkward
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
use super::radix::radix2_inline;
use crate::traits::FloatScalar;

/// Precomputed Bluestein plan for a fixed length `n` (not a power of two).
pub(crate) struct Bluestein<T: FloatScalar> {
    n: usize,
    /// `chirp[j] = exp(-πi j² / n)`, `j = 0..n`.
    chirp: Vec<Complex<T>>,
    /// `FFT_m` of the two-sided filter `conj(chirp)`.
    filter_fft: Vec<Complex<T>>,
    /// Reusable length-`m` work buffer (avoids per-call allocation).
    scratch: Vec<Complex<T>>,
}

impl<T: FloatScalar> Bluestein<T> {
    /// Build the plan for length `n`. `n` need not be a power of two.
    pub(crate) fn new(n: usize) -> Self {
        debug_assert!(n >= 2);
        let m = (2 * n - 1).next_power_of_two();

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
        radix2_inline(&mut filter, false);

        Self {
            n,
            chirp,
            filter_fft: filter,
            scratch: vec![zero; m],
        }
    }

    /// Apply the forward DFT in place over `buf` (`buf.len() == n`).
    pub(crate) fn forward(&mut self, buf: &mut [Complex<T>]) {
        debug_assert_eq!(buf.len(), self.n);
        let zero = Complex::new(T::zero(), T::zero());

        // a[j] = x[j] · chirp[j], zero-padded to length m.
        let a = &mut self.scratch;
        for j in 0..self.n {
            a[j] = buf[j] * self.chirp[j];
        }
        for slot in a.iter_mut().skip(self.n) {
            *slot = zero;
        }

        // conv = IFFT(FFT(a) · FFT(b))
        radix2_inline(a, false);
        for (ai, bi) in a.iter_mut().zip(&self.filter_fft) {
            *ai = *ai * *bi;
        }
        radix2_inline(a, true);

        // X[k] = chirp[k] · conv[k]
        for k in 0..self.n {
            buf[k] = a[k] * self.chirp[k];
        }
    }
}
