//! Statistical distributions: continuous and discrete.
//!
//! Each distribution provides [`ContinuousDistribution`] or [`DiscreteDistribution`]
//! trait implementations for a consistent API across all distributions.
//!
//! # Continuous distributions
//!
//! | Distribution | Parameters | Support |
//! |---|---|---|
//! | [`Normal`] | mean μ, std dev σ | (−∞, ∞) |
//! | [`Uniform`] | lower a, upper b | [a, b] |
//! | [`Exponential`] | rate λ | [0, ∞) |
//! | [`Gamma`] | shape α, rate β | (0, ∞) |
//! | [`Beta`] | shape α, shape β | [0, 1] |
//! | [`ChiSquared`] | degrees of freedom k | [0, ∞) |
//! | [`StudentT`] | degrees of freedom ν | (−∞, ∞) |
//!
//! # Discrete distributions
//!
//! | Distribution | Parameters | Support |
//! |---|---|---|
//! | [`Bernoulli`] | probability p | {0, 1} |
//! | [`Binomial`] | trials n, probability p | {0, …, n} |
//! | [`Poisson`] | rate λ | {0, 1, 2, …} |
//!
//! # Example
//!
//! ```
//! use numeris::stats::{Normal, ContinuousDistribution};
//!
//! let n = Normal::new(0.0_f64, 1.0).unwrap();
//! assert!((n.cdf(0.0) - 0.5).abs() < 1e-14);
//! assert!((n.mean()).abs() < 1e-14);
//! ```

mod normal;
mod uniform;
mod exponential;
mod gamma_dist;
mod beta_dist;
mod chi_squared;
mod student_t;
mod bernoulli;
mod binomial;
mod poisson;

#[cfg(test)]
mod tests;

pub use normal::Normal;
pub use uniform::Uniform;
pub use exponential::Exponential;
pub use gamma_dist::Gamma;
pub use beta_dist::Beta;
pub use chi_squared::ChiSquared;
pub use student_t::StudentT;
pub use bernoulli::Bernoulli;
pub use binomial::Binomial;
pub use poisson::Poisson;

use crate::traits::FloatScalar;

/// Errors from distribution construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StatsError {
    /// A parameter is out of its valid range.
    InvalidParameter,
}

impl core::fmt::Display for StatsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StatsError::InvalidParameter => {
                write!(f, "distribution parameter out of valid range")
            }
        }
    }
}

/// Trait for continuous probability distributions.
pub trait ContinuousDistribution<T> {
    /// Probability density function.
    fn pdf(&self, x: T) -> T;
    /// Natural log of the probability density function.
    fn ln_pdf(&self, x: T) -> T;
    /// Cumulative distribution function P(X ≤ x).
    fn cdf(&self, x: T) -> T;
    /// Quantile function (inverse CDF). Returns x such that P(X ≤ x) = p.
    fn quantile(&self, p: T) -> T;
    /// Expected value E\[X\].
    fn mean(&self) -> T;
    /// Variance Var(X).
    fn variance(&self) -> T;
}

/// Trait for discrete probability distributions.
pub trait DiscreteDistribution<T> {
    /// Probability mass function P(X = k).
    fn pmf(&self, k: u64) -> T;
    /// Natural log of the probability mass function.
    fn ln_pmf(&self, k: u64) -> T;
    /// Cumulative distribution function P(X ≤ k).
    fn cdf(&self, k: u64) -> T;
    /// Expected value E\[X\].
    fn mean(&self) -> T;
    /// Variance Var(X).
    fn variance(&self) -> T;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Standard normal quantile via Acklam's rational approximation.
/// Relative error < 1.15e-9. Input: p ∈ (0, 1).
pub(crate) fn normal_quantile_standard<T: FloatScalar>(p: T) -> T {
    let one = T::one();
    let half = T::from(0.5).unwrap();
    let two = T::from(2.0).unwrap();

    let p_low = T::from(0.02425).unwrap();
    let p_high = one - p_low;

    // Coefficients — central region
    let a1 = T::from(-3.969683028665376e+01).unwrap();
    let a2 = T::from(2.209460984245205e+02).unwrap();
    let a3 = T::from(-2.759285104469687e+02).unwrap();
    let a4 = T::from(1.383577518672690e+02).unwrap();
    let a5 = T::from(-3.066479806614716e+01).unwrap();
    let a6 = T::from(2.506628277459239e+00).unwrap();

    let b1 = T::from(-5.447609879822406e+01).unwrap();
    let b2 = T::from(1.615858368580409e+02).unwrap();
    let b3 = T::from(-1.556989798598866e+02).unwrap();
    let b4 = T::from(6.680131188771972e+01).unwrap();
    let b5 = T::from(-1.328068155288572e+01).unwrap();

    // Coefficients — tail regions
    let c1 = T::from(-7.784894002430293e-03).unwrap();
    let c2 = T::from(-3.223964580411365e-01).unwrap();
    let c3 = T::from(-2.400758277161838e+00).unwrap();
    let c4 = T::from(-2.549732539343734e+00).unwrap();
    let c5 = T::from(4.374664141464968e+00).unwrap();
    let c6 = T::from(2.938163982698783e+00).unwrap();

    let d1 = T::from(7.784695709041462e-03).unwrap();
    let d2 = T::from(3.224671290700398e-01).unwrap();
    let d3 = T::from(2.445134137142996e+00).unwrap();
    let d4 = T::from(3.754408661907416e+00).unwrap();

    if p < p_low {
        // Lower tail
        let q = (-two * p.ln()).sqrt();
        (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + one)
    } else if p <= p_high {
        // Central region
        let q = p - half;
        let r = q * q;
        (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
            / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + one)
    } else {
        // Upper tail — symmetry
        let q = (-two * (one - p).ln()).sqrt();
        -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + one)
    }
}

/// Newton-Raphson with bisection fallback for quantile computation.
pub(crate) fn quantile_newton<T: FloatScalar>(
    cdf_fn: impl Fn(T) -> T,
    pdf_fn: impl Fn(T) -> T,
    p: T,
    x0: T,
    mut lo: T,
    mut hi: T,
) -> T {
    if p <= T::zero() {
        return lo;
    }
    if p >= T::one() {
        return hi;
    }

    let two = T::one() + T::one();
    let tol = T::epsilon() * T::from(1000.0).unwrap();
    let mut x = x0.max(lo).min(hi);

    for _ in 0..100 {
        let f = cdf_fn(x) - p;
        if f.abs() < tol {
            return x;
        }
        if f < T::zero() {
            lo = x;
        } else {
            hi = x;
        }
        let fprime = pdf_fn(x);
        if fprime > T::epsilon() {
            let x_new = x - f / fprime;
            if x_new > lo && x_new < hi {
                x = x_new;
            } else {
                x = (lo + hi) / two;
            }
        } else {
            x = (lo + hi) / two;
        }
    }
    x
}
