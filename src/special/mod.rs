//! Special mathematical functions.
//!
//! Provides gamma, digamma, beta, incomplete gamma, and error functions.
//! All functions are generic over [`FloatScalar`] (f32/f64), no-std compatible,
//! and stack-only.
//!
//! # Functions
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`gamma`] | Gamma function Γ(x) |
//! | [`lgamma`] | Log-gamma ln Γ(x) |
//! | [`digamma`] | Digamma ψ(x) = d/dx ln Γ(x) |
//! | [`beta`] | Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b) |
//! | [`lbeta`] | Log-beta ln B(a,b) |
//! | [`gamma_inc`] | Regularized lower incomplete gamma P(a,x) |
//! | [`gamma_inc_upper`] | Regularized upper incomplete gamma Q(a,x) = 1−P(a,x) |
//! | [`erf`] | Error function |
//! | [`erfc`] | Complementary error function 1−erf(x) |
//!
//! # Example
//!
//! ```
//! use numeris::special::{gamma, lgamma, beta, erf};
//!
//! // Γ(5) = 4! = 24
//! assert!((gamma(5.0_f64) - 24.0).abs() < 1e-12);
//!
//! // ln Γ(1) = 0
//! assert!(lgamma(1.0_f64).abs() < 1e-14);
//!
//! // B(a,b) = B(b,a)
//! assert!((beta(2.0_f64, 3.0) - beta(3.0, 2.0)).abs() < 1e-14);
//!
//! // erf(0) = 0
//! assert!(erf(0.0_f64).abs() < 1e-16);
//! ```

use core::fmt;

use crate::FloatScalar;

mod gamma_fn;
mod digamma_fn;
mod beta_fn;
mod incgamma;
mod erf_fn;

#[cfg(test)]
mod tests;

pub use gamma_fn::{gamma, lgamma};
pub use digamma_fn::digamma;
pub use beta_fn::{beta, lbeta};
pub use incgamma::{gamma_inc, gamma_inc_upper};
pub use erf_fn::{erf, erfc};

/// Errors from special function evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialError {
    /// Series or continued fraction did not converge within the iteration limit.
    ConvergenceFailure,
    /// Input outside the function's domain (e.g. a ≤ 0 or x < 0 for incomplete gamma).
    DomainError,
}

impl fmt::Display for SpecialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConvergenceFailure => write!(f, "series/continued fraction did not converge"),
            Self::DomainError => write!(f, "input outside function domain"),
        }
    }
}

// ---------------------------------------------------------------------------
// Lanczos approximation constants (g = 7, n = 9)
// Coefficients from Paul Godfrey / Boost / CPython.
// ---------------------------------------------------------------------------

/// Lanczos parameter g.
pub(crate) const LANCZOS_G: f64 = 7.0;

/// Lanczos series coefficients (n = 9).
pub(crate) const LANCZOS_COEFFS: [f64; 9] = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
];

/// Evaluate the Lanczos series Ag(z) = c0 + c1/(z+1) + c2/(z+2) + ...
#[inline]
pub(crate) fn lanczos_sum<T: FloatScalar>(z: T) -> T {
    let mut sum = T::from(LANCZOS_COEFFS[0]).unwrap();
    for (i, &c) in LANCZOS_COEFFS[1..].iter().enumerate() {
        let ci = T::from(c).unwrap();
        let denom = z + T::from(i + 1).unwrap();
        sum = sum + ci / denom;
    }
    sum
}
