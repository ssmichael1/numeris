use crate::FloatScalar;
use crate::special::{lgamma, betainc};
use super::{DiscreteDistribution, StatsError};

/// Binomial distribution B(n, p).
///
/// P(X = k) = C(n,k) p^k (1−p)^{n−k} for k = 0, …, n.
///
/// # Example
///
/// ```
/// use numeris::stats::{Binomial, DiscreteDistribution};
///
/// let b = Binomial::new(10, 0.5_f64).unwrap();
/// assert!((b.mean() - 5.0).abs() < 1e-14);
/// assert!((b.variance() - 2.5).abs() < 1e-14);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Binomial<T> {
    n: u64,
    p: T,
}

impl<T: FloatScalar> Binomial<T> {
    /// Create a binomial distribution with `n` trials and success probability `p`.
    /// Requires `0 ≤ p ≤ 1`.
    pub fn new(n: u64, p: T) -> Result<Self, StatsError> {
        if p < T::zero() || p > T::one() {
            return Err(StatsError::InvalidParameter);
        }
        Ok(Self { n, p })
    }
}

impl<T: FloatScalar> DiscreteDistribution<T> for Binomial<T> {
    fn pmf(&self, k: u64) -> T {
        if k > self.n {
            return T::zero();
        }
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: u64) -> T {
        if k > self.n {
            return T::neg_infinity();
        }
        let one = T::one();
        let nf = T::from(self.n).unwrap();
        let kf = T::from(k).unwrap();
        lgamma(nf + one) - lgamma(kf + one) - lgamma(nf - kf + one)
            + kf * self.p.ln()
            + (nf - kf) * (one - self.p).ln()
    }

    fn cdf(&self, k: u64) -> T {
        if k >= self.n {
            return T::one();
        }
        let one = T::one();
        // P(X ≤ k) = I_{1-p}(n-k, k+1)
        let a = T::from(self.n - k).unwrap();
        let b = T::from(k + 1).unwrap();
        betainc(a, b, one - self.p).unwrap_or(T::nan())
    }

    fn mean(&self) -> T {
        T::from(self.n).unwrap() * self.p
    }

    fn variance(&self) -> T {
        T::from(self.n).unwrap() * self.p * (T::one() - self.p)
    }
}
