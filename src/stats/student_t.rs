use crate::FloatScalar;
use crate::special::{lgamma, betainc};
use super::{ContinuousDistribution, StatsError, quantile_newton, normal_quantile_standard};

/// Student's t-distribution with ν degrees of freedom.
///
/// # Example
///
/// ```
/// use numeris::stats::{StudentT, ContinuousDistribution};
///
/// let t = StudentT::new(10.0_f64).unwrap();
/// assert!((t.mean()).abs() < 1e-14);
/// assert!((t.variance() - 10.0/8.0).abs() < 1e-14);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct StudentT<T> {
    df: T, // ν > 0
}

impl<T: FloatScalar> StudentT<T> {
    /// Create a Student's t-distribution with `df` degrees of freedom. Requires `df > 0`.
    pub fn new(df: T) -> Result<Self, StatsError> {
        if df <= T::zero() {
            return Err(StatsError::InvalidParameter);
        }
        Ok(Self { df })
    }
}

impl<T: FloatScalar> ContinuousDistribution<T> for StudentT<T> {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let one = T::one();
        let two = one + one;
        let half = one / two;
        let pi = T::from(core::f64::consts::PI).unwrap();
        let v = self.df;
        lgamma((v + one) * half) - lgamma(v * half)
            - half * (v * pi).ln()
            - (v + one) * half * (one + x * x / v).ln()
    }

    fn cdf(&self, x: T) -> T {
        let one = T::one();
        let two = one + one;
        let half = one / two;
        let t = self.df / (self.df + x * x);
        let ib = betainc(self.df * half, half, t).unwrap_or(T::nan());
        if x >= T::zero() {
            one - half * ib
        } else {
            half * ib
        }
    }

    fn quantile(&self, p: T) -> T {
        let two = T::one() + T::one();
        // Initial guess from normal quantile, scaled by std dev
        let z = normal_quantile_standard(p);
        let x0 = if self.df > two {
            z * (self.df / (self.df - two)).sqrt()
        } else {
            z
        };
        let bound = T::from(1e6).unwrap();
        quantile_newton(|x| self.cdf(x), |x| self.pdf(x), p, x0, -bound, bound)
    }

    fn mean(&self) -> T {
        if self.df > T::one() {
            T::zero()
        } else {
            T::nan()
        }
    }

    fn variance(&self) -> T {
        let one = T::one();
        let two = one + one;
        if self.df > two {
            self.df / (self.df - two)
        } else if self.df > one {
            T::infinity()
        } else {
            T::nan()
        }
    }
}
