//! Regularized incomplete gamma functions P(a,x) and Q(a,x).

use crate::FloatScalar;
use super::SpecialError;
use super::gamma_fn::lgamma;

/// Maximum iterations for series / continued fraction.
const MAX_ITER: usize = 200;

/// Regularized lower incomplete gamma function P(a, x).
///
/// P(a, x) = γ(a, x) / Γ(a), where γ(a, x) = ∫₀ˣ t^{a−1} e^{−t} dt.
///
/// Requires a > 0 and x ≥ 0.
///
/// # Example
///
/// ```
/// use numeris::special::gamma_inc;
///
/// // P(a, 0) = 0 for any a > 0
/// assert!((gamma_inc(2.0_f64, 0.0).unwrap()).abs() < 1e-15);
///
/// // P(1, x) = 1 − e^{−x}
/// let x = 1.5_f64;
/// let expected = 1.0 - (-x).exp();
/// assert!((gamma_inc(1.0, x).unwrap() - expected).abs() < 1e-14);
/// ```
pub fn gamma_inc<T: FloatScalar>(a: T, x: T) -> Result<T, SpecialError> {
    let (p, _q) = gamma_inc_pair(a, x)?;
    Ok(p)
}

/// Regularized upper incomplete gamma function Q(a, x) = 1 − P(a, x).
///
/// Q(a, x) = Γ(a, x) / Γ(a), where Γ(a, x) = ∫ₓ^∞ t^{a−1} e^{−t} dt.
///
/// Requires a > 0 and x ≥ 0.
///
/// # Example
///
/// ```
/// use numeris::special::gamma_inc_upper;
///
/// // Q(a, 0) = 1 for any a > 0
/// assert!((gamma_inc_upper(2.0_f64, 0.0).unwrap() - 1.0).abs() < 1e-15);
/// ```
pub fn gamma_inc_upper<T: FloatScalar>(a: T, x: T) -> Result<T, SpecialError> {
    let (_p, q) = gamma_inc_pair(a, x)?;
    Ok(q)
}

/// Compute both P(a, x) and Q(a, x) = 1 − P(a, x) simultaneously.
///
/// Uses series expansion when x < a + 1, continued fraction otherwise.
/// This avoids cancellation when computing the complement.
fn gamma_inc_pair<T: FloatScalar>(a: T, x: T) -> Result<(T, T), SpecialError> {
    let zero = T::zero();
    let one = T::one();

    // Domain checks
    if a <= zero || x < zero {
        return Err(SpecialError::DomainError);
    }

    // Trivial case
    if x == zero {
        return Ok((zero, one));
    }

    // Log prefactor: exp(-x + a·ln(x) - lgamma(a))
    let log_prefactor = -x + a * x.ln() - lgamma(a);
    let prefactor = log_prefactor.exp();

    if x < a + one {
        // Series expansion for P(a, x)
        let p = series_p(a, x, prefactor)?;
        Ok((p, one - p))
    } else {
        // Continued fraction for Q(a, x) via Lentz's method
        let q = cf_q(a, x, prefactor)?;
        Ok((one - q, q))
    }
}

/// Series expansion for P(a, x):
/// P(a, x) = prefactor · Σ_{n=0}^∞ x^n / (a·(a+1)·…·(a+n))
fn series_p<T: FloatScalar>(a: T, x: T, prefactor: T) -> Result<T, SpecialError> {
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
            return Ok(prefactor * sum);
        }
    }
    Err(SpecialError::ConvergenceFailure)
}

/// Lentz continued fraction for Q(a, x):
/// Q(a, x) = prefactor · 1/(x + 1−a − 1·(1−a)/(x+3−a− 2·(2−a)/(x+5−a−…)))
///
/// Using the modified Lentz algorithm (Thompson & Barnett).
fn cf_q<T: FloatScalar>(a: T, x: T, prefactor: T) -> Result<T, SpecialError> {
    let one = T::one();
    let eps = T::epsilon();
    let tiny = T::from(1e-30).unwrap();

    // CF: Q(a,x) = prefactor / (x + 1 - a + K_{n=1}^∞ a_n/b_n)
    // where a_n = n(n-a), b_n = x + 2n + 1 - a
    // Lentz method: f = b0, C = b0, D = 0
    let b0 = x + one - a;
    let mut f = if b0.abs() < tiny { tiny } else { b0 };
    let mut c = f;
    let mut d = T::zero();

    for n in 1..=MAX_ITER {
        let nf = T::from(n).unwrap();
        let an = nf * (a - nf);                      // a_n = n*(a-n)
        let bn = x + T::from(2 * n + 1).unwrap() - a; // b_n = x + 2n + 1 - a

        d = bn + an * d;
        if d.abs() < tiny {
            d = tiny;
        }
        d = one / d;

        c = bn + an / c;
        if c.abs() < tiny {
            c = tiny;
        }

        let delta = c * d;
        f = f * delta;

        if (delta - one).abs() < eps {
            return Ok(prefactor * f.recip());
        }
    }
    Err(SpecialError::ConvergenceFailure)
}
