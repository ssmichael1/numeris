use crate::FloatScalar;
use crate::special::{lgamma, gamma_inc_upper};
use super::{DiscreteDistribution, StatsError};

/// Poisson distribution with rate λ.
///
/// P(X = k) = λ^k e^{−λ} / k! for k = 0, 1, 2, …
///
/// # Example
///
/// ```
/// use numeris::stats::{Poisson, DiscreteDistribution};
///
/// let p = Poisson::new(3.0_f64).unwrap();
/// assert!((p.mean() - 3.0).abs() < 1e-14);
/// assert!((p.variance() - 3.0).abs() < 1e-14);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Poisson<T> {
    lambda: T,
}

impl<T: FloatScalar> Poisson<T> {
    /// Create a Poisson distribution with rate `lambda`. Requires `lambda > 0`.
    pub fn new(lambda: T) -> Result<Self, StatsError> {
        if lambda <= T::zero() {
            return Err(StatsError::InvalidParameter);
        }
        Ok(Self { lambda })
    }
}

impl<T: FloatScalar> DiscreteDistribution<T> for Poisson<T> {
    fn pmf(&self, k: u64) -> T {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: u64) -> T {
        let one = T::one();
        let kf = T::from(k).unwrap();
        kf * self.lambda.ln() - self.lambda - lgamma(kf + one)
    }

    fn cdf(&self, k: u64) -> T {
        // P(X ≤ k) = Q(k+1, λ) = gamma_inc_upper(k+1, λ)
        let a = T::from(k + 1).unwrap();
        gamma_inc_upper(a, self.lambda).unwrap_or(T::nan())
    }

    fn mean(&self) -> T {
        self.lambda
    }

    fn variance(&self) -> T {
        self.lambda
    }
}
