use crate::FloatScalar;
use super::{ContinuousDistribution, StatsError};

/// Continuous uniform distribution on [a, b].
///
/// # Example
///
/// ```
/// use numeris::stats::{Uniform, ContinuousDistribution};
///
/// let u = Uniform::new(0.0_f64, 1.0).unwrap();
/// assert!((u.pdf(0.5) - 1.0).abs() < 1e-14);
/// assert!((u.cdf(0.5) - 0.5).abs() < 1e-14);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Uniform<T> {
    a: T,
    b: T,
}

impl<T: FloatScalar> Uniform<T> {
    /// Create a uniform distribution on [a, b]. Requires `a < b`.
    pub fn new(a: T, b: T) -> Result<Self, StatsError> {
        if a >= b {
            return Err(StatsError::InvalidParameter);
        }
        Ok(Self { a, b })
    }
}

impl<T: FloatScalar> ContinuousDistribution<T> for Uniform<T> {
    fn pdf(&self, x: T) -> T {
        if x >= self.a && x <= self.b {
            T::one() / (self.b - self.a)
        } else {
            T::zero()
        }
    }

    fn ln_pdf(&self, x: T) -> T {
        if x >= self.a && x <= self.b {
            -((self.b - self.a).ln())
        } else {
            T::neg_infinity()
        }
    }

    fn cdf(&self, x: T) -> T {
        if x <= self.a {
            T::zero()
        } else if x >= self.b {
            T::one()
        } else {
            (x - self.a) / (self.b - self.a)
        }
    }

    fn quantile(&self, p: T) -> T {
        self.a + p * (self.b - self.a)
    }

    fn mean(&self) -> T {
        let two = T::one() + T::one();
        (self.a + self.b) / two
    }

    fn variance(&self) -> T {
        let twelve = T::from(12.0).unwrap();
        let d = self.b - self.a;
        d * d / twelve
    }
}
