use crate::FloatScalar;
use super::{DiscreteDistribution, StatsError};

/// Bernoulli distribution with success probability p.
///
/// P(X = 1) = p, P(X = 0) = 1 − p.
///
/// # Example
///
/// ```
/// use numeris::stats::{Bernoulli, DiscreteDistribution};
///
/// let b = Bernoulli::new(0.3_f64).unwrap();
/// assert!((b.pmf(1) - 0.3).abs() < 1e-14);
/// assert!((b.pmf(0) - 0.7).abs() < 1e-14);
/// assert!((b.mean() - 0.3).abs() < 1e-14);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Bernoulli<T> {
    p: T,
}

impl<T: FloatScalar> Bernoulli<T> {
    /// Create a Bernoulli distribution with success probability `p`.
    /// Requires `0 ≤ p ≤ 1`.
    pub fn new(p: T) -> Result<Self, StatsError> {
        if p < T::zero() || p > T::one() {
            return Err(StatsError::InvalidParameter);
        }
        Ok(Self { p })
    }
}

impl<T: FloatScalar> Bernoulli<T> {
    /// Draw a random sample (0 or 1) from this distribution.
    pub fn sample(&self, rng: &mut super::Rng) -> u64 {
        if rng.next_float::<T>() < self.p { 1 } else { 0 }
    }

    /// Fill a fixed-size array with independent samples.
    pub fn sample_array<const K: usize>(&self, rng: &mut super::Rng) -> [u64; K] {
        let mut out = [0u64; K];
        for v in out.iter_mut() {
            *v = self.sample(rng);
        }
        out
    }
}

impl<T: FloatScalar> DiscreteDistribution<T> for Bernoulli<T> {
    fn pmf(&self, k: u64) -> T {
        match k {
            0 => T::one() - self.p,
            1 => self.p,
            _ => T::zero(),
        }
    }

    fn ln_pmf(&self, k: u64) -> T {
        match k {
            0 => (T::one() - self.p).ln(),
            1 => self.p.ln(),
            _ => T::neg_infinity(),
        }
    }

    fn cdf(&self, k: u64) -> T {
        if k == 0 {
            T::one() - self.p
        } else {
            T::one()
        }
    }

    fn quantile(&self, p: T) -> u64 {
        // Support is {0, 1}. CDF(0) = 1 - self.p, CDF(1) = 1.
        if p <= T::zero() || self.cdf(0) >= p {
            0
        } else {
            1
        }
    }

    fn mean(&self) -> T {
        self.p
    }

    fn variance(&self) -> T {
        self.p * (T::one() - self.p)
    }
}
