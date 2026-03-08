//! Simple xoshiro256++ pseudo-random number generator.
//!
//! No-std compatible, no heap allocation. Suitable for Monte Carlo
//! simulation and random sampling, NOT for cryptography.

use crate::FloatScalar;

/// xoshiro256++ pseudo-random number generator.
///
/// Fast, high-quality PRNG with 256-bit state and a period of 2^256 - 1.
/// State is seeded from a single `u64` via SplitMix64.
///
/// # Example
///
/// ```
/// use numeris::stats::Rng;
///
/// let mut rng = Rng::new(42);
/// let x = rng.next_f64();
/// assert!(x >= 0.0 && x < 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

/// SplitMix64 — used only to seed xoshiro from a single u64.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

impl Rng {
    /// Create a new PRNG from a 64-bit seed.
    ///
    /// Uses SplitMix64 to expand the seed into the 4-word xoshiro state.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::stats::Rng;
    ///
    /// let mut rng = Rng::new(12345);
    /// let a = rng.next_u64();
    /// let b = rng.next_u64();
    /// assert_ne!(a, b);
    /// ```
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let s0 = splitmix64(&mut sm);
        let s1 = splitmix64(&mut sm);
        let s2 = splitmix64(&mut sm);
        let s3 = splitmix64(&mut sm);
        Self { s: [s0, s1, s2, s3] }
    }

    /// Generate the next `u64` in the sequence.
    pub fn next_u64(&mut self) -> u64 {
        // xoshiro256++ result
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        result
    }

    /// Uniform `f64` in [0, 1).
    ///
    /// Uses the upper 53 bits of `next_u64` to fill the mantissa exactly.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0_f64 / (1u64 << 53) as f64)
    }

    /// Uniform `f32` in [0, 1).
    ///
    /// Uses the upper 24 bits of `next_u64` to fill the mantissa exactly.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0_f32 / (1u64 << 24) as f32)
    }

    /// Sample a standard normal (mean 0, std dev 1) as `f64` via Box-Muller transform.
    pub(crate) fn next_normal_f64(&mut self) -> f64 {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > 0.0 {
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * core::f64::consts::PI * u2;
                return r * theta.cos();
            }
        }
    }

    /// Uniform float in [0, 1), generic over `FloatScalar`.
    ///
    /// Uses `next_f64` internally and converts via `T::from`.
    pub(crate) fn next_float<T: FloatScalar>(&mut self) -> T {
        T::from(self.next_f64()).unwrap()
    }

    /// Standard normal, generic over `FloatScalar`.
    pub(crate) fn next_normal<T: FloatScalar>(&mut self) -> T {
        T::from(self.next_normal_f64()).unwrap()
    }

    /// Gamma variate with shape `alpha` and rate 1 (scale 1), generic.
    ///
    /// Uses Marsaglia & Tsang's method for shape >= 1.
    /// For shape < 1: sample Gamma(alpha+1, 1) * U^(1/alpha).
    pub(crate) fn next_gamma<T: FloatScalar>(&mut self, alpha: T) -> T {
        let one = T::one();

        if alpha < one {
            // Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
            let g = self.next_gamma(alpha + one);
            let u: T = self.next_float();
            return g * u.powf(one / alpha);
        }

        // Marsaglia & Tsang for alpha >= 1
        let three = T::from(3.0).unwrap();
        let nine = T::from(9.0).unwrap();
        let d = alpha - one / three;
        let c = one / (nine * d).sqrt();

        loop {
            let x: T = self.next_normal();
            let v = one + c * x;
            if v <= T::zero() {
                continue;
            }
            let v = v * v * v;
            let u: T = self.next_float();
            let x2 = x * x;
            // Squeeze test
            if u < one - T::from(0.0331).unwrap() * x2 * x2 {
                return d * v;
            }
            if u.ln() < T::from(0.5).unwrap() * x2 + d * (one - v + v.ln()) {
                return d * v;
            }
        }
    }
}
