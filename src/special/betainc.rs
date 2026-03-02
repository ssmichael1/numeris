//! Regularized incomplete beta function I_x(a, b).

use crate::FloatScalar;
use super::SpecialError;
use super::beta_fn::lbeta;

/// Maximum iterations for continued fraction.
const MAX_ITER: usize = 200;

/// Regularized incomplete beta function I_x(a, b).
///
/// I_x(a, b) = B(x; a, b) / B(a, b) where B(x; a, b) = ∫₀ˣ t^{a−1}(1−t)^{b−1} dt.
///
/// Domain: a > 0, b > 0, 0 ≤ x ≤ 1.
///
/// # Example
///
/// ```
/// use numeris::special::betainc;
///
/// // I_0(a, b) = 0 and I_1(a, b) = 1
/// assert!(betainc(2.0_f64, 3.0, 0.0).unwrap().abs() < 1e-15);
/// assert!((betainc(2.0_f64, 3.0, 1.0).unwrap() - 1.0).abs() < 1e-15);
///
/// // I_{0.5}(1, 1) = 0.5 (uniform distribution)
/// assert!((betainc(1.0_f64, 1.0, 0.5).unwrap() - 0.5).abs() < 1e-14);
/// ```
pub fn betainc<T: FloatScalar>(a: T, b: T, x: T) -> Result<T, SpecialError> {
    let zero = T::zero();
    let one = T::one();

    if a <= zero || b <= zero {
        return Err(SpecialError::DomainError);
    }
    if x < zero || x > one {
        return Err(SpecialError::DomainError);
    }
    if x == zero {
        return Ok(zero);
    }
    if x == one {
        return Ok(one);
    }

    // Use symmetry for better CF convergence:
    // When x > (a+1)/(a+b+2), compute I_{1-x}(b, a) instead
    let two = one + one;
    if x > (a + one) / (a + b + two) {
        Ok(one - betainc_cf(b, a, one - x)?)
    } else {
        betainc_cf(a, b, x)
    }
}

/// Evaluate I_x(a,b) via continued fraction (modified Lentz's method).
///
/// CF from DLMF 8.17.22 / Numerical Recipes:
/// I_x(a,b) = x^a (1-x)^b / (a·B(a,b)) · 1/cf
///
/// where cf is evaluated by the modified Lentz algorithm.
fn betainc_cf<T: FloatScalar>(a: T, b: T, x: T) -> Result<T, SpecialError> {
    let one = T::one();
    let two = one + one;
    let eps = T::epsilon();
    let tiny = T::from(1e-30).unwrap();

    // Log-prefactor for numerical stability
    let ln_prefix = a * x.ln() + b * (one - x).ln() - lbeta(a, b);
    let prefix = ln_prefix.exp() / a;

    // Modified Lentz's method
    // Initialize with first odd term: d1 = -(a+b)*x/(a+1)
    let qab = a + b;
    let qap = a + one;
    let qam = a - one;

    // Lentz variables: f accumulates the result, c and d are independent chains
    let mut c = one;
    let mut d = one - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = one / d;
    let mut f = d;

    for m in 1..=MAX_ITER {
        let fm = T::from(m).unwrap();
        let m2 = two * fm;

        // Even step: a_{2m} = m(b-m)x / ((a+2m-1)(a+2m))
        let aa_even = fm * (b - fm) * x / ((qam + m2) * (a + m2));

        d = one + aa_even * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = one + aa_even / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = one / d;
        f = f * d * c;

        // Odd step: a_{2m+1} = -(a+m)(a+b+m)x / ((a+2m)(a+2m+1))
        let aa_odd = -((a + fm) * (qab + fm) * x) / ((a + m2) * (qap + m2));

        d = one + aa_odd * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = one + aa_odd / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = one / d;
        let delta = d * c;
        f = f * delta;

        if (delta - one).abs() < eps {
            return Ok(prefix * f);
        }
    }

    Err(SpecialError::ConvergenceFailure)
}
