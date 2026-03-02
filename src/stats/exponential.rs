use crate::FloatScalar;
use super::{ContinuousDistribution, StatsError};

/// Exponential distribution with rate λ.
///
/// f(x) = λ exp(−λx) for x ≥ 0.
///
/// # Example
///
/// ```
/// use numeris::stats::{Exponential, ContinuousDistribution};
///
/// let e = Exponential::new(2.0_f64).unwrap();
/// assert!((e.mean() - 0.5).abs() < 1e-14);
/// assert!((e.cdf(0.0)).abs() < 1e-14);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Exponential<T> {
    lambda: T,
}

impl<T: FloatScalar> Exponential<T> {
    /// Create an exponential distribution with rate `lambda`. Requires `lambda > 0`.
    pub fn new(lambda: T) -> Result<Self, StatsError> {
        if lambda <= T::zero() {
            return Err(StatsError::InvalidParameter);
        }
        Ok(Self { lambda })
    }
}

impl<T: FloatScalar> ContinuousDistribution<T> for Exponential<T> {
    fn pdf(&self, x: T) -> T {
        if x < T::zero() {
            T::zero()
        } else {
            self.lambda * (-self.lambda * x).exp()
        }
    }

    fn ln_pdf(&self, x: T) -> T {
        if x < T::zero() {
            T::neg_infinity()
        } else {
            self.lambda.ln() - self.lambda * x
        }
    }

    fn cdf(&self, x: T) -> T {
        if x <= T::zero() {
            T::zero()
        } else {
            T::one() - (-self.lambda * x).exp()
        }
    }

    fn quantile(&self, p: T) -> T {
        -(T::one() - p).ln() / self.lambda
    }

    fn mean(&self) -> T {
        T::one() / self.lambda
    }

    fn variance(&self) -> T {
        T::one() / (self.lambda * self.lambda)
    }
}
