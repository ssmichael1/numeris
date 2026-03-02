use crate::FloatScalar;
use crate::special::{erf, erfc};
use super::{ContinuousDistribution, StatsError, normal_quantile_standard};

/// Normal (Gaussian) distribution N(μ, σ²).
///
/// # Example
///
/// ```
/// use numeris::stats::{Normal, ContinuousDistribution};
///
/// let n = Normal::new(0.0_f64, 1.0).unwrap();
/// assert!((n.cdf(0.0) - 0.5).abs() < 1e-14);
/// assert!((n.quantile(0.975) - 1.96).abs() < 0.01);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Normal<T> {
    mu: T,
    sigma: T,
}

impl<T: FloatScalar> Normal<T> {
    /// Create a normal distribution with mean `mu` and standard deviation `sigma`.
    ///
    /// Requires `sigma > 0`.
    pub fn new(mu: T, sigma: T) -> Result<Self, StatsError> {
        if sigma <= T::zero() {
            return Err(StatsError::InvalidParameter);
        }
        Ok(Self { mu, sigma })
    }
}

impl<T: FloatScalar> ContinuousDistribution<T> for Normal<T> {
    fn pdf(&self, x: T) -> T {
        let two = T::one() + T::one();
        let pi = T::from(core::f64::consts::PI).unwrap();
        let z = (x - self.mu) / self.sigma;
        (-(z * z) / two).exp() / (self.sigma * (two * pi).sqrt())
    }

    fn ln_pdf(&self, x: T) -> T {
        let two = T::one() + T::one();
        let pi = T::from(core::f64::consts::PI).unwrap();
        let z = (x - self.mu) / self.sigma;
        -self.sigma.ln() - (two * pi).ln() / two - z * z / two
    }

    fn cdf(&self, x: T) -> T {
        let half = T::from(0.5).unwrap();
        let sqrt2 = T::from(core::f64::consts::SQRT_2).unwrap();
        let z = (x - self.mu) / (self.sigma * sqrt2);
        if z >= T::zero() {
            half * (T::one() + erf(z))
        } else {
            half * erfc(-z)
        }
    }

    fn quantile(&self, p: T) -> T {
        self.mu + self.sigma * normal_quantile_standard(p)
    }

    fn mean(&self) -> T {
        self.mu
    }

    fn variance(&self) -> T {
        self.sigma * self.sigma
    }
}
