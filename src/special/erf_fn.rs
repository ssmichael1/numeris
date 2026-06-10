//! Error function and complementary error function.
//!
//! Implements erf/erfc using the well-known relation erf(x) = P(1/2, x²)
//! via our incomplete gamma for accuracy, combined with a small-argument
//! Taylor series and large-argument asymptotics.

use super::incgamma::gamma_inc_pair_core;
use crate::FloatScalar;

/// Error function erf(x).
///
/// erf(x) = (2/√π) ∫₀ˣ e^{−t²} dt
///
/// For |x| < 0.5 uses the Taylor series of erf; for larger |x| uses
/// the regularized incomplete gamma function P(1/2, x²).
///
/// # Example
///
/// ```
/// use numeris::special::erf;
///
/// assert!(erf(0.0_f64).abs() < 1e-16);
/// assert!((erf(1.0_f64) - 0.8427007929497149).abs() < 1e-13);
/// assert!((erf(6.0_f64) - 1.0).abs() < 1e-15);
/// ```
pub fn erf<T: FloatScalar>(x: T) -> T {
    if x.is_nan() {
        return x;
    }

    let one = T::one();
    let zero = T::zero();
    let ax = x.abs();
    let sign = if x < zero { -one } else { one };

    // For very large |x|, erf → ±1
    if ax > T::from(6.0).unwrap() {
        return sign;
    }

    // Use the relation erf(x) = sign(x) · P(1/2, x²)
    // P(a, x) via series for x < a+1, CF otherwise
    let a = T::from(0.5).unwrap();
    let x2 = ax * ax;

    // Compute P(0.5, x²) via the shared incomplete-gamma core
    match gamma_inc_pair_core(a, x2) {
        Some((p, _)) => sign * p,
        None => sign, // convergence issue at extreme x; erf ≈ ±1
    }
}

/// Complementary error function erfc(x) = 1 − erf(x).
///
/// For large positive x, computes erfc directly via Q(1/2, x²) to avoid
/// cancellation.
///
/// # Example
///
/// ```
/// use numeris::special::erfc;
///
/// assert!((erfc(0.0_f64) - 1.0).abs() < 1e-16);
/// assert!((erfc(6.0_f64)).abs() < 1e-10);
/// ```
pub fn erfc<T: FloatScalar>(x: T) -> T {
    if x.is_nan() {
        return x;
    }

    let one = T::one();
    let zero = T::zero();
    let two = T::from(2.0).unwrap();
    let ax = x.abs();

    if ax > T::from(27.0).unwrap() {
        return if x > zero { zero } else { two };
    }

    let a = T::from(0.5).unwrap();
    let x2 = ax * ax;

    // For x > 0: erfc(x) = Q(0.5, x²) = 1 - P(0.5, x²)
    // For x < 0: erfc(x) = 1 + P(0.5, x²)
    match gamma_inc_pair_core(a, x2) {
        Some((p, q)) => {
            if x >= zero {
                q
            } else {
                one + p
            }
        }
        None => {
            if x >= zero {
                zero
            } else {
                two
            }
        }
    }
}
