use crate::FloatScalar;
use crate::special::{lgamma, gamma_inc};
use super::{ContinuousDistribution, StatsError, quantile_newton, normal_quantile_standard};

/// Chi-squared distribution with k degrees of freedom.
///
/// Special case of Gamma(k/2, 1/2).
///
/// # Example
///
/// ```
/// use numeris::stats::{ChiSquared, ContinuousDistribution};
///
/// let chi2 = ChiSquared::new(3.0_f64).unwrap();
/// assert!((chi2.mean() - 3.0).abs() < 1e-14);
/// assert!((chi2.variance() - 6.0).abs() < 1e-14);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ChiSquared<T> {
    k: T, // degrees of freedom
}

impl<T: FloatScalar> ChiSquared<T> {
    /// Create a chi-squared distribution with `k` degrees of freedom. Requires `k > 0`.
    pub fn new(k: T) -> Result<Self, StatsError> {
        if k <= T::zero() {
            return Err(StatsError::InvalidParameter);
        }
        Ok(Self { k })
    }
}

impl<T: FloatScalar> ContinuousDistribution<T> for ChiSquared<T> {
    fn pdf(&self, x: T) -> T {
        if x <= T::zero() {
            return T::zero();
        }
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        if x <= T::zero() {
            return T::neg_infinity();
        }
        let one = T::one();
        let two = one + one;
        let half_k = self.k / two;
        (half_k - one) * x.ln() - x / two - half_k * two.ln() - lgamma(half_k)
    }

    fn cdf(&self, x: T) -> T {
        if x <= T::zero() {
            return T::zero();
        }
        let two = T::one() + T::one();
        gamma_inc(self.k / two, x / two).unwrap_or(T::nan())
    }

    fn quantile(&self, p: T) -> T {
        let two = T::one() + T::one();
        let nine = T::from(9.0).unwrap();
        // Wilson-Hilferty approximation
        let z = normal_quantile_standard(p);
        let v = T::one() - two / (nine * self.k) + z * (two / (nine * self.k)).sqrt();
        let x0 = if v > T::zero() { self.k * v * v * v } else { self.mean() };
        let hi = self.mean() + T::from(40.0).unwrap() * self.variance().sqrt();
        quantile_newton(|x| self.cdf(x), |x| self.pdf(x), p, x0, T::zero(), hi)
    }

    fn mean(&self) -> T {
        self.k
    }

    fn variance(&self) -> T {
        let two = T::one() + T::one();
        two * self.k
    }
}
