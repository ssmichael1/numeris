//! Runtime-sized FFT planner ([`DynFft`], requires `alloc`).

use num_complex::Complex;

use super::bluestein::Bluestein;
use super::cast;
use super::soa::SoaPlan;
use crate::traits::FloatScalar;

/// Transform strategy chosen by the planner for a given length.
enum Plan<T: FloatScalar> {
    /// Length is a power of two — the deinterleaved SIMD radix core transforms
    /// in place directly.
    PowerOfTwo(SoaPlan<T>),
    /// Arbitrary/prime length — Bluestein reduces it to power-of-two FFTs.
    Bluestein(Bluestein<T>),
}

/// A cached FFT plan for a fixed runtime length.
///
/// Build once with [`DynFft::new`], then call [`forward`](DynFft::forward) /
/// [`inverse`](DynFft::inverse) repeatedly — twiddles and scratch are computed
/// once and reused. Power-of-two lengths transform directly with the radix core;
/// any other length (including primes) goes through Bluestein's algorithm.
///
/// This is the RustFFT-style planner shape without the autotuning. Requires the
/// `alloc` feature.
///
/// # Examples
///
/// ```
/// # use numeris::fft::DynFft;
/// # use numeris::Complex;
/// let mut plan = DynFft::<f64>::new(6); // not a power of two -> Bluestein
/// let mut buf: Vec<Complex<f64>> = (0..6).map(|i| Complex::new(i as f64, 0.0)).collect();
/// plan.forward(&mut buf);
/// plan.inverse(&mut buf);
/// assert!((buf[1].re - 1.0).abs() < 1e-10);
/// ```
pub struct DynFft<T: FloatScalar> {
    n: usize,
    plan: Plan<T>,
}

impl<T: FloatScalar> DynFft<T> {
    /// Build a plan for transforms of length `len`.
    ///
    /// Panics if `len == 0`.
    pub fn new(len: usize) -> Self {
        assert!(len > 0, "DynFft length must be non-zero");
        let plan = if len.is_power_of_two() {
            Plan::PowerOfTwo(SoaPlan::new(len))
        } else {
            Plan::Bluestein(Bluestein::new(len))
        };
        Self { n: len, plan }
    }

    /// The transform length this plan was built for.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the transform length is zero. Always `false` — a
    /// `DynFft` cannot be constructed with length zero (kept for lint parity).
    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// In-place forward FFT. Panics unless `buf.len() == self.len()`.
    ///
    /// Uses the `exp(-2πi k n / N)` sign convention.
    pub fn forward(&mut self, buf: &mut [Complex<T>]) {
        assert_eq!(
            buf.len(),
            self.n,
            "DynFft: buffer length must equal plan length"
        );
        match &mut self.plan {
            Plan::PowerOfTwo(soa) => soa.forward(buf),
            Plan::Bluestein(b) => b.forward(buf),
        }
    }

    /// In-place inverse FFT, normalized by `1/N`. Panics unless
    /// `buf.len() == self.len()`.
    ///
    /// Implemented via `ifft(x) = conj(fft(conj(x))) / N`, so both the
    /// power-of-two and Bluestein paths reuse the (accelerated) forward plan.
    pub fn inverse(&mut self, buf: &mut [Complex<T>]) {
        assert_eq!(
            buf.len(),
            self.n,
            "DynFft: buffer length must equal plan length"
        );
        for z in buf.iter_mut() {
            *z = z.conj();
        }
        self.forward(buf);
        let inv_n = T::one() / cast::<T>(self.n as f64);
        for z in buf.iter_mut() {
            let c = z.conj();
            *z = Complex::new(c.re * inv_n, c.im * inv_n);
        }
    }
}
