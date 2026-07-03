use super::{ContinuousDistribution, Gamma, StatsError};
use crate::FloatScalar;

/// Chi-squared distribution with k degrees of freedom.
///
/// Special case of Gamma(k/2, 1/2), to which every method delegates.
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
    inner: Gamma<T>, // Gamma(k/2, 1/2)
}

impl<T: FloatScalar> ChiSquared<T> {
    /// Create a chi-squared distribution with `k` degrees of freedom. Requires `k > 0`.
    pub fn new(k: T) -> Result<Self, StatsError> {
        if k <= T::zero() {
            return Err(StatsError::InvalidParameter);
        }
        let two = T::one() + T::one();
        // Chi-squared(k) = Gamma(shape = k/2, rate = 1/2).
        let inner = Gamma::new(k / two, T::one() / two)?;
        Ok(Self { inner })
    }

    /// Draw a random sample from this distribution.
    ///
    /// Chi-squared(k) = Gamma(k/2, 1/2).
    pub fn sample(&self, rng: &mut super::Rng) -> T {
        self.inner.sample(rng)
    }

    impl_sample_array!(T, T::zero());
}

impl<T: FloatScalar> ContinuousDistribution<T> for ChiSquared<T> {
    fn pdf(&self, x: T) -> T {
        self.inner.pdf(x)
    }

    fn ln_pdf(&self, x: T) -> T {
        self.inner.ln_pdf(x)
    }

    fn cdf(&self, x: T) -> T {
        self.inner.cdf(x)
    }

    fn quantile(&self, p: T) -> T {
        self.inner.quantile(p)
    }

    fn mean(&self) -> T {
        self.inner.mean()
    }

    fn variance(&self) -> T {
        self.inner.variance()
    }
}
