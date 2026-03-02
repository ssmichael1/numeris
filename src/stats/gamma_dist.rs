use crate::FloatScalar;
use crate::special::{lgamma, gamma_inc};
use super::{ContinuousDistribution, StatsError, quantile_newton, normal_quantile_standard};

/// Gamma distribution with shape α and rate β.
///
/// f(x) = β^α x^{α−1} e^{−βx} / Γ(α) for x > 0.
///
/// The scale parameter is θ = 1/β.
///
/// # Example
///
/// ```
/// use numeris::stats::{Gamma, ContinuousDistribution};
///
/// let g = Gamma::new(2.0_f64, 1.0).unwrap();
/// assert!((g.mean() - 2.0).abs() < 1e-14);
/// assert!((g.variance() - 2.0).abs() < 1e-14);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Gamma<T> {
    shape: T, // α > 0
    rate: T,  // β > 0
}

impl<T: FloatScalar> Gamma<T> {
    /// Create a Gamma distribution with `shape` α and `rate` β.
    /// Requires both > 0. Scale = 1/rate.
    pub fn new(shape: T, rate: T) -> Result<Self, StatsError> {
        if shape <= T::zero() || rate <= T::zero() {
            return Err(StatsError::InvalidParameter);
        }
        Ok(Self { shape, rate })
    }
}

impl<T: FloatScalar> ContinuousDistribution<T> for Gamma<T> {
    fn pdf(&self, x: T) -> T {
        if x < T::zero() {
            return T::zero();
        }
        if x == T::zero() {
            let one = T::one();
            if self.shape == one {
                return self.rate; // Exponential special case
            } else if self.shape > one {
                return T::zero(); // mode is interior
            } else {
                return T::infinity(); // shape < 1: density blows up at 0
            }
        }
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        if x < T::zero() {
            return T::neg_infinity();
        }
        if x == T::zero() {
            return self.pdf(x).ln();
        }
        let one = T::one();
        self.shape * self.rate.ln() - lgamma(self.shape)
            + (self.shape - one) * x.ln()
            - self.rate * x
    }

    fn cdf(&self, x: T) -> T {
        if x <= T::zero() {
            return T::zero();
        }
        gamma_inc(self.shape, self.rate * x).unwrap_or(T::nan())
    }

    fn quantile(&self, p: T) -> T {
        let mean = self.mean();
        let std = self.variance().sqrt();
        // Wilson-Hilferty initial guess for shape >= 1
        let x0 = if self.shape >= T::one() {
            let nine = T::from(9.0).unwrap();
            let z = normal_quantile_standard(p);
            let v = T::one() - T::one() / (nine * self.shape)
                + z / (nine * self.shape).sqrt();
            let wh = self.shape / self.rate * v * v * v;
            if wh > T::zero() { wh } else { mean }
        } else {
            mean
        };
        let hi = mean + T::from(40.0).unwrap() * std;
        quantile_newton(|x| self.cdf(x), |x| self.pdf(x), p, x0, T::zero(), hi)
    }

    fn mean(&self) -> T {
        self.shape / self.rate
    }

    fn variance(&self) -> T {
        self.shape / (self.rate * self.rate)
    }
}
