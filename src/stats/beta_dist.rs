use super::{quantile_newton, ContinuousDistribution, StatsError};
use crate::special::{betainc, lbeta};
use crate::FloatScalar;

/// Beta distribution with shape parameters α and β on [0, 1].
///
/// f(x) = x^{α−1} (1−x)^{β−1} / B(α, β) for 0 ≤ x ≤ 1.
///
/// # Example
///
/// ```
/// use numeris::stats::{Beta, ContinuousDistribution};
///
/// let b = Beta::new(2.0_f64, 5.0).unwrap();
/// assert!((b.mean() - 2.0/7.0).abs() < 1e-14);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Beta<T> {
    alpha: T,
    beta: T,
    ln_beta_fn: T, // lbeta(α, β), cached at construction
}

impl<T: FloatScalar> Beta<T> {
    /// Create a Beta distribution with shape parameters `alpha` and `beta`.
    /// Requires both > 0.
    pub fn new(alpha: T, beta: T) -> Result<Self, StatsError> {
        if alpha <= T::zero() || beta <= T::zero() {
            return Err(StatsError::InvalidParameter);
        }
        let ln_beta_fn = lbeta(alpha, beta);
        Ok(Self {
            alpha,
            beta,
            ln_beta_fn,
        })
    }
}

impl<T: FloatScalar> Beta<T> {
    /// Draw a random sample from this distribution.
    ///
    /// Samples X ~ Gamma(alpha, 1), Y ~ Gamma(beta, 1), returns X / (X + Y).
    pub fn sample(&self, rng: &mut super::Rng) -> T {
        let x = rng.next_gamma(self.alpha);
        let y = rng.next_gamma(self.beta);
        x / (x + y)
    }

    impl_sample_array!(T, T::zero());
}

impl<T: FloatScalar> ContinuousDistribution<T> for Beta<T> {
    fn pdf(&self, x: T) -> T {
        if x < T::zero() || x > T::one() {
            return T::zero();
        }
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        if x < T::zero() || x > T::one() {
            return T::neg_infinity();
        }
        let one = T::one();
        (self.alpha - one) * x.ln() + (self.beta - one) * (one - x).ln() - self.ln_beta_fn
    }

    fn cdf(&self, x: T) -> T {
        if x <= T::zero() {
            return T::zero();
        }
        if x >= T::one() {
            return T::one();
        }
        betainc(self.alpha, self.beta, x).unwrap_or(T::nan())
    }

    fn quantile(&self, p: T) -> T {
        let x0 = self.mean();
        let eps = T::epsilon();
        quantile_newton(|x| self.cdf(x), |x| self.pdf(x), p, x0, eps, T::one() - eps)
    }

    fn mean(&self) -> T {
        self.alpha / (self.alpha + self.beta)
    }

    fn variance(&self) -> T {
        let ab = self.alpha + self.beta;
        self.alpha * self.beta / (ab * ab * (ab + T::one()))
    }
}
