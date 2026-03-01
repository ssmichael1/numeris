//! Beta function and log-beta via lgamma.

use crate::FloatScalar;
use super::gamma_fn::lgamma;

/// Beta function B(a, b) = Γ(a)·Γ(b) / Γ(a+b).
///
/// Computed as `exp(lbeta(a, b))` to avoid overflow for large arguments.
///
/// # Example
///
/// ```
/// use numeris::special::beta;
///
/// // B(1, 1) = 1
/// assert!((beta(1.0_f64, 1.0) - 1.0).abs() < 1e-14);
///
/// // B(2, 3) = 1/12
/// assert!((beta(2.0_f64, 3.0) - 1.0 / 12.0).abs() < 1e-14);
/// ```
pub fn beta<T: FloatScalar>(a: T, b: T) -> T {
    lbeta(a, b).exp()
}

/// Natural logarithm of the beta function, ln B(a, b).
///
/// Computed as `lgamma(a) + lgamma(b) − lgamma(a+b)`.
///
/// # Example
///
/// ```
/// use numeris::special::lbeta;
///
/// // ln B(1, 1) = 0
/// assert!(lbeta(1.0_f64, 1.0).abs() < 1e-14);
/// ```
pub fn lbeta<T: FloatScalar>(a: T, b: T) -> T {
    lgamma(a) + lgamma(b) - lgamma(a + b)
}
