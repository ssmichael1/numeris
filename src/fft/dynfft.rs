//! Runtime-sized FFT planner ([`DynFft`], requires `alloc`).

use num_complex::Complex;

use super::bluestein::{Bluestein, BluesteinScratch};
use super::soa::{SoaPlan, SoaScratch};
use crate::traits::FloatScalar;

/// Transform strategy chosen by the planner for a given length.
enum Plan<T: FloatScalar> {
    /// Length is a power of two — the deinterleaved SIMD radix core transforms
    /// in place directly.
    PowerOfTwo(SoaPlan<T>),
    /// Arbitrary/prime length — Bluestein reduces it to power-of-two FFTs.
    Bluestein(Bluestein<T>),
}

enum Scratch<T: FloatScalar> {
    PowerOfTwo(SoaScratch<T>),
    Bluestein(BluesteinScratch<T>),
}

/// Work buffers for a [`DynFft`] plan (requires `alloc`).
///
/// A plan's twiddles are read-only during a transform; the scratch is the only
/// mutable state. [`DynFft::forward`] / [`DynFft::inverse`] use the scratch the
/// plan carries internally. To run one plan from several threads at once — or
/// to keep the plan behind a shared reference — build one scratch per worker
/// with [`DynFft::make_scratch`] and call [`DynFft::forward_with`] /
/// [`DynFft::inverse_with`]. A scratch is tied to the length it was built for;
/// passing it to a plan of a different length panics.
pub struct DynFftScratch<T: FloatScalar> {
    len: usize,
    inner: Scratch<T>,
}

impl<T: FloatScalar> DynFftScratch<T> {
    /// The transform length this scratch was built for.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always `false` — a scratch cannot be built for length zero.
    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// A cached FFT plan for a fixed runtime length.
///
/// Build once with [`DynFft::new`], then call [`forward`](DynFft::forward) /
/// [`inverse`](DynFft::inverse) repeatedly — twiddles and scratch are computed
/// once and reused. Power-of-two lengths transform directly with the SIMD radix
/// core; any other length (including primes) goes through Bluestein's algorithm
/// (whose inner power-of-two FFTs use the same SIMD core).
///
/// The plan is immutable during a transform, so it can be shared across threads:
/// give each worker its own [`DynFftScratch`] and use
/// [`forward_with`](DynFft::forward_with) / [`inverse_with`](DynFft::inverse_with).
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
///
/// Batched transforms through a shared plan:
///
/// ```
/// # use numeris::fft::DynFft;
/// # use numeris::Complex;
/// let plan = DynFft::<f64>::new(8);
/// let mut scratch = plan.make_scratch();
/// let mut frames = vec![vec![Complex::new(1.0, 0.0); 8]; 4];
/// for frame in &mut frames {
///     plan.forward_with(frame, &mut scratch); // `&plan`, not `&mut plan`
/// }
/// assert!((frames[0][0].re - 8.0).abs() < 1e-12);
/// ```
pub struct DynFft<T: FloatScalar> {
    n: usize,
    plan: Plan<T>,
    scratch: DynFftScratch<T>,
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
        let scratch = Self::scratch_for(&plan, len);
        Self {
            n: len,
            plan,
            scratch,
        }
    }

    fn scratch_for(plan: &Plan<T>, len: usize) -> DynFftScratch<T> {
        let inner = match plan {
            Plan::PowerOfTwo(soa) => Scratch::PowerOfTwo(soa.scratch()),
            Plan::Bluestein(b) => Scratch::Bluestein(b.scratch()),
        };
        DynFftScratch { len, inner }
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

    /// Allocate a fresh scratch buffer for this plan, for use with
    /// [`forward_with`](DynFft::forward_with) / [`inverse_with`](DynFft::inverse_with).
    pub fn make_scratch(&self) -> DynFftScratch<T> {
        Self::scratch_for(&self.plan, self.n)
    }

    /// In-place forward FFT. Panics unless `buf.len() == self.len()`.
    ///
    /// Uses the `exp(-2πi k n / N)` sign convention.
    pub fn forward(&mut self, buf: &mut [Complex<T>]) {
        let Self { plan, scratch, n } = self;
        Self::transform(plan, *n, buf, scratch, false);
    }

    /// In-place inverse FFT, normalized by `1/N`. Panics unless
    /// `buf.len() == self.len()`.
    ///
    /// Implemented via `ifft(x) = conj(fft(conj(x))) / N`, with the
    /// conjugations folded into the data copies, so both the power-of-two and
    /// Bluestein paths reuse the (accelerated) forward plan at no extra pass.
    pub fn inverse(&mut self, buf: &mut [Complex<T>]) {
        let Self { plan, scratch, n } = self;
        Self::transform(plan, *n, buf, scratch, true);
    }

    /// Forward FFT through a caller-owned scratch, leaving the plan shared.
    /// Panics unless `buf.len() == self.len()` and `scratch` was built for this
    /// length.
    pub fn forward_with(&self, buf: &mut [Complex<T>], scratch: &mut DynFftScratch<T>) {
        Self::transform(&self.plan, self.n, buf, scratch, false);
    }

    /// Inverse FFT (normalized by `1/N`) through a caller-owned scratch. Panics
    /// unless `buf.len() == self.len()` and `scratch` was built for this length.
    pub fn inverse_with(&self, buf: &mut [Complex<T>], scratch: &mut DynFftScratch<T>) {
        Self::transform(&self.plan, self.n, buf, scratch, true);
    }

    fn transform(
        plan: &Plan<T>,
        n: usize,
        buf: &mut [Complex<T>],
        scratch: &mut DynFftScratch<T>,
        inverse: bool,
    ) {
        assert_eq!(buf.len(), n, "DynFft: buffer length must equal plan length");
        assert_eq!(
            scratch.len, n,
            "DynFft: scratch was built for a different transform length"
        );
        match (plan, &mut scratch.inner) {
            (Plan::PowerOfTwo(soa), Scratch::PowerOfTwo(s)) => soa.transform(buf, s, inverse),
            (Plan::Bluestein(b), Scratch::Bluestein(s)) => b.transform(buf, s, inverse),
            // A scratch of the right length always has the matching kind: the
            // kind is a pure function of the length (power of two or not).
            _ => unreachable!("DynFft: scratch kind does not match plan kind"),
        }
    }
}
