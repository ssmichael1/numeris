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

impl<T: FloatScalar> Poisson<T> {
    /// Draw a random sample from this distribution.
    ///
    /// For small lambda (< 30), uses Knuth's algorithm.
    /// For large lambda, uses a normal approximation.
    pub fn sample(&self, rng: &mut super::Rng) -> u64 {
        let thirty = T::from(30.0).unwrap();
        if self.lambda < thirty {
            // Knuth's algorithm
            let l = (-self.lambda).exp();
            let mut k = 0u64;
            let mut p = T::one();
            loop {
                k += 1;
                p = p * rng.next_float::<T>();
                if p <= l {
                    return k - 1;
                }
            }
        } else {
            // Normal approximation for large lambda
            let z: T = rng.next_normal();
            let x = self.lambda + self.lambda.sqrt() * z;
            if x < T::zero() {
                0
            } else {
                x.round().to_u64().unwrap_or(0)
            }
        }
    }

    /// Fill a fixed-size array with independent samples.
    pub fn sample_array<const K: usize>(&self, rng: &mut super::Rng) -> [u64; K] {
        let mut out = [0u64; K];
        for v in out.iter_mut() {
            *v = self.sample(rng);
        }
        out
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

    fn quantile(&self, p: T) -> u64 {
        if p <= T::zero() {
            return 0;
        }
        if p >= T::one() {
            // Use lambda + 8·sqrt(lambda) + 1 as a conservative upper bound and search down.
            let k_hi = (self.lambda + T::from(8.0).unwrap() * self.lambda.sqrt() + T::one())
                .to_u64()
                .unwrap_or(u64::MAX / 2);
            return super::discrete_quantile_search(|k| self.cdf(k), p, k_hi);
        }
        // Normal approximation: k0 ≈ lambda + sqrt(lambda)·z_p
        let z = super::normal_quantile_standard(p);
        let k0 = (self.lambda + self.lambda.sqrt() * z)
            .max(T::zero())
            .floor()
            .to_u64()
            .unwrap_or(0);
        super::discrete_quantile_search(|k| self.cdf(k), p, k0)
    }

    fn mean(&self) -> T {
        self.lambda
    }

    fn variance(&self) -> T {
        self.lambda
    }
}
