//! Gamma and log-gamma functions via Lanczos approximation.

use crate::FloatScalar;
use super::{LANCZOS_G, lanczos_sum};

/// Factorial lookup table for small positive integers: FACTORIAL[n] = n!
/// Valid for n = 0..=20 (20! < 2^64, fits in f64 exactly up to 18!).
const FACTORIAL: [f64; 21] = [
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5040.0,
    40320.0,
    362880.0,
    3628800.0,
    39916800.0,
    479001600.0,
    6227020800.0,
    87178291200.0,
    1307674368000.0,
    20922789888000.0,
    355687428096000.0,
    6402373705728000.0,
    121645100408832000.0,
    2432902008176640000.0,
];

/// Gamma function Γ(x).
///
/// Uses the Lanczos approximation (g = 7, n = 9) for general arguments
/// and the reflection formula for x < 0.5. Returns infinity at
/// non-positive integer poles (0, −1, −2, …) and NaN for NaN input.
///
/// # Example
///
/// ```
/// use numeris::special::gamma;
///
/// // Γ(5) = 4! = 24
/// assert!((gamma(5.0_f64) - 24.0).abs() < 1e-10);
///
/// // Γ(0.5) = √π
/// let sqrt_pi = core::f64::consts::PI.sqrt();
/// assert!((gamma(0.5_f64) - sqrt_pi).abs() < 1e-14);
/// ```
pub fn gamma<T: FloatScalar>(x: T) -> T {
    let zero = T::zero();
    let one = T::one();
    let half = T::from(0.5).unwrap();

    // NaN passthrough
    if x.is_nan() {
        return x;
    }

    // Positive integer fast path (factorial lookup)
    if x > zero && x == x.floor() {
        if let Some(n) = num_traits::cast::<T, u64>(x) {
            if n >= 1 && n <= 21 {
                // Γ(n) = (n-1)!
                return T::from(FACTORIAL[(n - 1) as usize]).unwrap();
            }
        }
    }

    // Non-positive integers: poles → +infinity
    if x <= zero && x == x.floor() {
        return T::infinity();
    }

    // Reflection formula for x < 0.5
    if x < half {
        let pi = T::from(core::f64::consts::PI).unwrap();
        let sin_pi_x = (pi * x).sin();
        if sin_pi_x == zero {
            return T::infinity();
        }
        return pi / (sin_pi_x * gamma(one - x));
    }

    // Lanczos approximation for x >= 0.5
    let z = x - one;
    let g = T::from(LANCZOS_G).unwrap();
    let t = z + g + half;
    let sqrt_2pi = T::from(core::f64::consts::TAU.sqrt()).unwrap();

    sqrt_2pi * t.powf(z + half) * (-t).exp() * lanczos_sum(z)
}

/// Natural logarithm of the gamma function, ln Γ(x).
///
/// Uses the Lanczos approximation in log space to avoid overflow for large
/// arguments. For x < 0.5, uses the reflection formula in log space.
/// Returns infinity at non-positive integer poles and NaN for NaN input.
///
/// # Example
///
/// ```
/// use numeris::special::lgamma;
///
/// // ln Γ(1) = 0
/// assert!(lgamma(1.0_f64).abs() < 1e-14);
///
/// // ln Γ(100) — large argument, no overflow
/// let val = lgamma(100.0_f64);
/// assert!((val - 359.1342053695754).abs() < 1e-8);
/// ```
pub fn lgamma<T: FloatScalar>(x: T) -> T {
    let zero = T::zero();
    let one = T::one();
    let half = T::from(0.5).unwrap();

    // NaN passthrough
    if x.is_nan() {
        return x;
    }

    // Non-positive integers: poles → +infinity
    if x <= zero && x == x.floor() {
        return T::infinity();
    }

    // Reflection formula in log space for x < 0.5
    if x < half {
        let pi = T::from(core::f64::consts::PI).unwrap();
        let sin_pi_x = (pi * x).sin().abs();
        if sin_pi_x == zero {
            return T::infinity();
        }
        return pi.ln() - sin_pi_x.ln() - lgamma(one - x);
    }

    // Lanczos in log space for x >= 0.5
    let z = x - one;
    let g = T::from(LANCZOS_G).unwrap();
    let t = z + g + half;
    let ln_sqrt_2pi = T::from(0.5 * core::f64::consts::TAU.ln()).unwrap();

    ln_sqrt_2pi + (z + half) * t.ln() - t + lanczos_sum(z).ln()
}
