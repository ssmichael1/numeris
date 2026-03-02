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

    fn mean(&self) -> T {
        self.p
    }

    fn variance(&self) -> T {
        self.p * (T::one() - self.p)
    }
}
