//! Optimization: root finding, unconstrained minimization, nonlinear least squares.
//!
//! All algorithms are no-alloc compatible, using fixed-size stack-allocated
//! matrices from [`crate::matrix`]. Requires [`FloatScalar`] bound (real-valued only).
//!
//! # Root finding
//!
//! - [`brent`] — Brent's method (bisection + secant + inverse quadratic interpolation)
//! - [`newton_1d`] — Newton's method with user-supplied derivative
//!
//! # Unconstrained minimization
//!
//! - [`minimize_bfgs`] — BFGS quasi-Newton with Armijo line search
//!
//! # Nonlinear least squares
//!
//! - [`least_squares_gn`] — Gauss-Newton (QR-based)
//! - [`least_squares_lm`] — Levenberg-Marquardt (damped normal equations)
//!
//! # Finite differences
//!
//! - [`finite_difference_gradient`] — forward-difference gradient approximation
//! - [`finite_difference_jacobian`] — forward-difference Jacobian approximation

mod bfgs;
mod gauss_newton;
mod jacobian;
mod levenberg_marquardt;
pub(crate) mod line_search;
mod root;

#[cfg(test)]
mod tests;

pub use bfgs::{minimize_bfgs, BfgsSettings};
pub use gauss_newton::{least_squares_gn, GaussNewtonSettings};
pub use jacobian::{finite_difference_gradient, finite_difference_jacobian};
pub use levenberg_marquardt::{least_squares_lm, LmSettings};
pub use root::{brent, newton_1d, RootSettings};

use crate::matrix::vector::Vector;
use crate::traits::FloatScalar;

/// Errors from optimization algorithms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimError {
    /// Maximum number of iterations exceeded.
    MaxIterations,
    /// Bracket endpoints do not have opposite signs.
    BracketInvalid,
    /// Encountered a singular or near-singular matrix.
    Singular,
    /// A computed value was NaN or infinity.
    NotFinite,
    /// Line search failed to find a sufficient decrease.
    LineSearchFailed,
}

impl core::fmt::Display for OptimError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OptimError::MaxIterations => write!(f, "maximum iterations exceeded"),
            OptimError::BracketInvalid => write!(f, "bracket endpoints must have opposite signs"),
            OptimError::Singular => write!(f, "singular or near-singular matrix"),
            OptimError::NotFinite => write!(f, "computed value is NaN or infinity"),
            OptimError::LineSearchFailed => write!(f, "line search failed"),
        }
    }
}

/// Result of a scalar root-finding algorithm.
#[derive(Debug, Clone, Copy)]
pub struct RootResult<T> {
    /// Approximate root.
    pub x: T,
    /// Function value at the root: `f(x)`.
    pub fx: T,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Number of function evaluations.
    pub evals: usize,
}

/// Result of an unconstrained minimization algorithm.
#[derive(Debug, Clone, Copy)]
pub struct MinimizeResult<T: FloatScalar, const N: usize> {
    /// Approximate minimizer.
    pub x: Vector<T, N>,
    /// Function value at the minimizer: `f(x)`.
    pub fx: T,
    /// Gradient norm at the minimizer.
    pub grad_norm: T,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Number of function evaluations.
    pub f_evals: usize,
    /// Number of gradient evaluations.
    pub grad_evals: usize,
}

/// Result of a nonlinear least-squares algorithm.
#[derive(Debug, Clone, Copy)]
pub struct LeastSquaresResult<T: FloatScalar, const N: usize> {
    /// Approximate minimizer of `0.5 * ||r(x)||^2`.
    pub x: Vector<T, N>,
    /// Final cost: `0.5 * ||r(x)||^2`.
    pub cost: T,
    /// Gradient norm: `||J^T r||`.
    pub grad_norm: T,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Number of residual evaluations.
    pub r_evals: usize,
    /// Number of Jacobian evaluations.
    pub j_evals: usize,
}
