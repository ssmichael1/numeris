use crate::traits::FloatScalar;

/// A single second-order section (biquad) filter using Direct Form II Transposed.
///
/// Transfer function:
/// ```text
/// H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)
/// ```
///
/// The denominator is stored normalized so `a[0] = 1`.
#[derive(Debug, Clone, Copy)]
pub struct Biquad<T> {
    b: [T; 3],
    a: [T; 3], // [1, a1, a2]
    z: [T; 2], // DFII-T state
}

impl<T: FloatScalar> Biquad<T> {
    /// Create a new biquad from numerator `b` and denominator `a` coefficients.
    ///
    /// Normalizes by `a[0]` so the stored `a[0]` is always 1.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::control::Biquad;
    ///
    /// let bq = Biquad::new([1.0, 2.0, 1.0], [1.0, -0.5, 0.1]);
    /// let (b, a) = bq.coefficients();
    /// assert_eq!(a[0], 1.0);
    /// ```
    pub fn new(b: [T; 3], a: [T; 3]) -> Self {
        let a0 = a[0];
        Self {
            b: [b[0] / a0, b[1] / a0, b[2] / a0],
            a: [T::one(), a[1] / a0, a[2] / a0],
            z: [T::zero(); 2],
        }
    }

    /// Identity (passthrough) filter: output equals input.
    pub fn passthrough() -> Self {
        Self {
            b: [T::one(), T::zero(), T::zero()],
            a: [T::one(), T::zero(), T::zero()],
            z: [T::zero(); 2],
        }
    }

    /// Process a single input sample, returning the filtered output.
    ///
    /// Uses Direct Form II Transposed for numerical stability.
    #[inline]
    pub fn tick(&mut self, x: T) -> T {
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x - self.a[1] * y + self.z[1];
        self.z[1] = self.b[2] * x - self.a[2] * y;
        y
    }

    /// Reset internal state to zero.
    pub fn reset(&mut self) {
        self.z = [T::zero(); 2];
    }

    /// Process a slice of input samples into an output slice.
    ///
    /// # Panics
    ///
    /// Panics if `output.len() < input.len()`.
    pub fn process(&mut self, input: &[T], output: &mut [T]) {
        assert!(output.len() >= input.len());
        for (i, &x) in input.iter().enumerate() {
            output[i] = self.tick(x);
        }
    }

    /// Process a slice of samples in-place.
    pub fn process_inplace(&mut self, data: &mut [T]) {
        for sample in data.iter_mut() {
            *sample = self.tick(*sample);
        }
    }

    /// Return the `(b, a)` coefficient arrays.
    pub fn coefficients(&self) -> ([T; 3], [T; 3]) {
        (self.b, self.a)
    }
}

/// A cascade of `N` biquad (second-order) sections.
///
/// Filter order is `2*N` for all-second-order sections, or `2*N - 1` when the
/// last section is a degenerate first-order biquad (`b2 = a2 = 0`).
///
/// # Example
///
/// ```
/// use numeris::control::{butterworth_lowpass, BiquadCascade};
///
/// // 4th-order Butterworth → 2 biquad sections
/// let mut lpf: BiquadCascade<f64, 2> = butterworth_lowpass(4, 1000.0, 8000.0).unwrap();
/// let y = lpf.tick(1.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BiquadCascade<T, const N: usize> {
    pub sections: [Biquad<T>; N],
}

impl<T: FloatScalar, const N: usize> BiquadCascade<T, N> {
    /// Process a single input sample through all sections in series.
    #[inline]
    pub fn tick(&mut self, x: T) -> T {
        let mut y = x;
        for section in self.sections.iter_mut() {
            y = section.tick(y);
        }
        y
    }

    /// Reset all sections' internal state to zero.
    pub fn reset(&mut self) {
        for section in self.sections.iter_mut() {
            section.reset();
        }
    }

    /// Process a slice of input samples into an output slice.
    ///
    /// # Panics
    ///
    /// Panics if `output.len() < input.len()`.
    pub fn process(&mut self, input: &[T], output: &mut [T]) {
        assert!(output.len() >= input.len());
        for (i, &x) in input.iter().enumerate() {
            output[i] = self.tick(x);
        }
    }

    /// Process a slice of samples in-place.
    pub fn process_inplace(&mut self, data: &mut [T]) {
        for sample in data.iter_mut() {
            *sample = self.tick(*sample);
        }
    }

    /// Actual filter order (detects degenerate first-order last section).
    pub fn order(&self) -> usize {
        if N == 0 {
            return 0;
        }
        let last = &self.sections[N - 1];
        let (b, a) = last.coefficients();
        // Degenerate first-order: b2 == 0 and a2 == 0
        if b[2] == T::zero() && a[2] == T::zero() {
            2 * N - 1
        } else {
            2 * N
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Bilinear transform helpers (pub(super) — used by butterworth/chebyshev)
// ─────────────────────────────────────────────────────────────────────

/// Build a lowpass biquad from a conjugate analog pole pair `(σ ± jω)`.
///
/// `wa` is the pre-warped analog cutoff, `c = 2·fs`.
pub(super) fn bilinear_lp_pair<T: FloatScalar>(sigma: T, omega: T, wa: T, c: T) -> Biquad<T> {
    let two = T::one() + T::one();
    let d = c * c - two * sigma * c + sigma * sigma + omega * omega;
    let wa2 = wa * wa;
    let b0 = wa2 / d;
    let b1 = two * wa2 / d;
    let b2 = b0;
    let a1 = two * (sigma * sigma + omega * omega - c * c) / d;
    let a2 = (c * c + two * sigma * c + sigma * sigma + omega * omega) / d;
    Biquad::new([b0, b1, b2], [T::one(), a1, a2])
}

/// Build a highpass biquad from a conjugate analog pole pair `(σ ± jω)`.
pub(super) fn bilinear_hp_pair<T: FloatScalar>(sigma: T, omega: T, _wa: T, c: T) -> Biquad<T> {
    let two = T::one() + T::one();
    let c2 = c * c;
    let d = c2 - two * sigma * c + sigma * sigma + omega * omega;
    let b0 = c2 / d;
    let b1 = -(two * c2) / d;
    let b2 = b0;
    let a1 = two * (sigma * sigma + omega * omega - c2) / d;
    let a2 = (c2 + two * sigma * c + sigma * sigma + omega * omega) / d;
    Biquad::new([b0, b1, b2], [T::one(), a1, a2])
}

/// Build a lowpass biquad from a single real analog pole `σ` (odd-order case).
pub(super) fn bilinear_lp_real<T: FloatScalar>(sigma: T, wa: T, c: T) -> Biquad<T> {
    // sigma is negative for stable poles; wa > 0
    // LP: b0 = wa/(c - sigma), b1 = b0, b2 = 0
    //     a1 = (sigma + c) is wrong — let's derive carefully:
    // H_a(s) = wa / (s - sigma)   [sigma < 0]
    // Bilinear: s = c·(z-1)/(z+1)
    // H(z) = wa / (c·(z-1)/(z+1) - sigma)
    //       = wa·(z+1) / (c·(z-1) - sigma·(z+1))
    //       = wa·(z+1) / ((c - sigma)z + (-c - sigma))
    // Divide by (c - sigma):
    //   b0 = wa/(c - sigma), b1 = wa/(c - sigma), b2 = 0
    //   a0 = 1, a1 = (-c - sigma)/(c - sigma), a2 = 0
    let denom = c - sigma;
    let b0 = wa / denom;
    Biquad::new([b0, b0, T::zero()], [T::one(), (-c - sigma) / denom, T::zero()])
}

/// Build a highpass biquad from a single real analog pole `σ` (odd-order case).
pub(super) fn bilinear_hp_real<T: FloatScalar>(sigma: T, _wa: T, c: T) -> Biquad<T> {
    // H_a(s) = s / (s - sigma)   [sigma < 0]
    // Bilinear: s = c·(z-1)/(z+1)
    // H(z) = c·(z-1)/(z+1) / (c·(z-1)/(z+1) - sigma)
    //       = c·(z-1) / (c·(z-1) - sigma·(z+1))
    //       = c·(z-1) / ((c-sigma)z + (-c-sigma))
    // Divide by (c - sigma):
    //   b0 = c/(c - sigma), b1 = -c/(c - sigma), b2 = 0
    //   a0 = 1, a1 = (-c - sigma)/(c - sigma), a2 = 0
    let denom = c - sigma;
    let b0 = c / denom;
    Biquad::new(
        [b0, -b0, T::zero()],
        [T::one(), (-c - sigma) / denom, T::zero()],
    )
}
