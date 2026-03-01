//! Digamma (psi) function via recurrence and asymptotic expansion.

use crate::FloatScalar;

/// Bernoulli-number-derived coefficients for the asymptotic expansion
/// of digamma: ψ(x) ≈ ln(x) − 1/(2x) − Σ B_{2k}/(2k · x^{2k}).
/// These are B_{2k}/(2k) for k = 1..7.
/// B2=1/6, B4=-1/30, B6=1/42, B8=-1/30, B10=5/66, B12=-691/2730, B14=7/6
const DIGAMMA_ASYMP: [f64; 7] = [
    1.0 / 12.0,               // B2/2 = (1/6)/2
    -1.0 / 120.0,             // B4/4 = (-1/30)/4
    1.0 / 252.0,              // B6/6 = (1/42)/6
    -1.0 / 240.0,             // B8/8 = (-1/30)/8
    1.0 / 132.0,              // B10/10 = (5/66)/10
    -691.0 / 32760.0,         // B12/12 = (-691/2730)/12
    1.0 / 12.0,               // B14/14 = (7/6)/14 = 1/12
];

/// Digamma function ψ(x) = d/dx ln Γ(x).
///
/// Uses the recurrence relation ψ(x+1) = ψ(x) + 1/x to shift x into the
/// asymptotic region (x ≥ 6), then applies a 7-term asymptotic expansion
/// in 1/x². For negative x, uses the reflection formula
/// ψ(1−x) = ψ(x) + π·cot(πx).
///
/// Returns NaN at non-positive integer poles (0, −1, −2, …).
///
/// # Example
///
/// ```
/// use numeris::special::digamma;
///
/// // ψ(1) = −γ (Euler-Mascheroni constant)
/// let euler_mascheroni = 0.5772156649015329_f64;
/// assert!((digamma(1.0_f64) - (-euler_mascheroni)).abs() < 1e-12);
/// ```
pub fn digamma<T: FloatScalar>(x: T) -> T {
    let zero = T::zero();
    let one = T::one();

    // NaN passthrough
    if x.is_nan() {
        return x;
    }

    // Non-positive integers: poles → NaN
    if x <= zero && x == x.floor() {
        return T::nan();
    }

    // Reflection formula for x < 0: ψ(x) = ψ(1-x) + π·cot(πx)
    // Rearranged: ψ(x) = ψ(1-x) - π/tan(πx)
    if x < zero {
        let pi = T::from(core::f64::consts::PI).unwrap();
        let tan_pi_x = (pi * x).tan();
        return digamma(one - x) - pi / tan_pi_x;
    }

    // Recurrence: shift x up until x >= 6 for asymptotic accuracy
    let mut result = zero;
    let mut xx = x;
    let threshold = T::from(6.0).unwrap();
    while xx < threshold {
        result = result - one / xx;
        xx = xx + one;
    }

    // Asymptotic expansion: ψ(x) ≈ ln(x) - 1/(2x) - Σ coeff_k / x^{2k}
    let half = T::from(0.5).unwrap();
    result = result + xx.ln() - half / xx;

    let inv_x2 = one / (xx * xx);
    let mut term = inv_x2;
    for &c in &DIGAMMA_ASYMP {
        let ci = T::from(c).unwrap();
        result = result - ci * term;
        term = term * inv_x2;
    }

    result
}
