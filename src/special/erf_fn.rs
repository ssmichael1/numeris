//! Error function and complementary error function.
//!
//! Implements erf/erfc using the well-known relation erf(x) = P(1/2, x²)
//! via our incomplete gamma for accuracy, combined with a small-argument
//! Taylor series and large-argument asymptotics.

use crate::FloatScalar;
use super::gamma_fn::lgamma;

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

    // Compute P(0.5, x²) using the same series/CF as gamma_inc
    // but inline here to avoid Result overhead
    match inc_gamma_p(a, x2) {
        Some(p) => sign * p,
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
    match inc_gamma_pair(a, x2) {
        Some((p, q)) => {
            if x >= zero {
                q
            } else {
                one + p
            }
        }
        None => {
            if x >= zero { zero } else { two }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: regularized incomplete gamma P(a, x) and Q(a, x)
// Duplicates the logic from incgamma.rs but returns Option instead of Result
// and avoids circular dependency issues with re-exporting.
// ---------------------------------------------------------------------------

const MAX_ITER: usize = 200;

/// Compute P(a, x) only.
fn inc_gamma_p<T: FloatScalar>(a: T, x: T) -> Option<T> {
    let (p, _) = inc_gamma_pair(a, x)?;
    Some(p)
}

/// Compute both (P, Q).
fn inc_gamma_pair<T: FloatScalar>(a: T, x: T) -> Option<(T, T)> {
    let zero = T::zero();
    let one = T::one();

    if x == zero {
        return Some((zero, one));
    }

    // Log prefactor: exp(-x + a·ln(x) - lgamma(a))
    let log_pf = -x + a * x.ln() - lgamma(a);
    let pf = log_pf.exp();

    if x < a + one {
        let p = series_p(a, x, pf)?;
        Some((p, one - p))
    } else {
        let q = cf_q(a, x, pf)?;
        Some((one - q, q))
    }
}

fn series_p<T: FloatScalar>(a: T, x: T, pf: T) -> Option<T> {
    let one = T::one();
    let eps = T::epsilon();
    let mut term = one / a;
    let mut sum = term;
    let mut ap = a;
    for _ in 0..MAX_ITER {
        ap = ap + one;
        term = term * x / ap;
        sum = sum + term;
        if term.abs() < sum.abs() * eps {
            return Some(pf * sum);
        }
    }
    None
}

fn cf_q<T: FloatScalar>(a: T, x: T, pf: T) -> Option<T> {
    let one = T::one();
    let eps = T::epsilon();
    let tiny = T::from(1e-30).unwrap();

    let b0 = x + one - a;
    let mut f = if b0.abs() < tiny { tiny } else { b0 };
    let mut c = f;
    let mut d = T::zero();

    for n in 1..=MAX_ITER {
        let nf = T::from(n).unwrap();
        let an = nf * (a - nf);
        let bn = x + T::from(2 * n + 1).unwrap() - a;

        d = bn + an * d;
        if d.abs() < tiny { d = tiny; }
        d = one / d;

        c = bn + an / c;
        if c.abs() < tiny { c = tiny; }

        let delta = c * d;
        f = f * delta;

        if (delta - one).abs() < eps {
            return Some(pf * f.recip());
        }
    }
    None
}
