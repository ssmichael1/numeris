//! ODE integration — fixed-step, adaptive, and stiff solvers.
//!
//! # Fixed-step
//!
//! [`rk4_step`] and [`rk4`] provide classic 4th-order Runge-Kutta integration.
//! Fully no-alloc: k-values are local variables on the stack.
//!
//! # Adaptive explicit solvers
//!
//! All adaptive solvers implement the [`RKAdaptive`] trait with Butcher tableau
//! constants. The [`integrate`](RKAdaptive::integrate) method uses a PID step-size
//! controller (Söderlind & Wang 2006) with embedded error estimation.
//!
//! | Solver               | Stages | Order | FSAL | Interpolant |
//! |----------------------|--------|-------|------|-------------|
//! | [`RKF45`]            |      6 | 5(4)  | no   | —           |
//! | [`RKTS54`]           |      7 | 5(4)  | yes  | 4th degree  |
//! | [`RKV65`]            |     10 | 6(5)  | no   | 6th degree  |
//! | [`RKV87`]            |     17 | 8(7)  | no   | 7th degree  |
//! | [`RKV98`]            |     21 | 9(8)  | no   | 8th degree  |
//! | [`RKV98NoInterp`]    |     16 | 9(8)  | no   | —           |
//! | [`RKV98Efficient`]   |     26 | 9(8)  | no   | 9th degree  |
//!
//! # Rosenbrock (stiff) solvers
//!
//! For stiff ODEs, the [`Rosenbrock`] trait provides linearly-implicit methods
//! that solve linear systems involving the Jacobian instead of nonlinear Newton
//! iterations. Use [`RODAS4`] for most stiff problems.
//!
//! | Solver               | Stages | Order | L-stable |
//! |----------------------|--------|-------|----------|
//! | [`RODAS4`]           |      6 | 4(3)  | yes      |
//!
//! # Example
//!
//! ```
//! use numeris::ode::{RKAdaptive, RKTS54, AdaptiveSettings};
//! use numeris::Vector;
//!
//! // Harmonic oscillator: y'' = -y  →  [y, y'] with dy/dt = [y', -y]
//! let y0 = Vector::from_array([1.0_f64, 0.0]); // cos(0), sin(0)
//! let tau = 2.0 * std::f64::consts::PI;
//! let settings = AdaptiveSettings::default();
//! let sol = RKTS54::integrate(
//!     0.0, tau, &y0,
//!     |_t, y| Vector::from_array([y[1], -y[0]]),
//!     &settings,
//! ).unwrap();
//! assert!((sol.y[0] - 1.0).abs() < 1e-6); // cos(2π) ≈ 1
//! assert!((sol.y[1]).abs() < 1e-6);        // sin(2π) ≈ 0
//! ```

mod rk4;
mod adaptive;
mod rkf45;
mod rkts54;
mod rkv65;
mod rkv87;
mod rkv98;
mod rkv98_nointerp;
mod rkv98_efficient;
mod rosenbrock;
mod rodas4;

use core::fmt;
use crate::traits::FloatScalar;
use crate::matrix::vector::Vector;

#[cfg(test)]
mod tests;

pub use rk4::{rk4_step, rk4};
pub use adaptive::{RKAdaptive, AdaptiveSettings};
pub use rkf45::RKF45;
pub use rkts54::RKTS54;
pub use rkv65::RKV65;
pub use rkv87::RKV87;
pub use rkv98::RKV98;
pub use rkv98_nointerp::RKV98NoInterp;
pub use rkv98_efficient::RKV98Efficient;
pub use rosenbrock::Rosenbrock;
pub use rodas4::RODAS4;

/// Errors from ODE integration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OdeError {
    /// Step error or state became non-finite (NaN / Inf).
    StepNotFinite,
    /// Exceeded maximum number of steps (adaptive only).
    MaxStepsExceeded,
    /// Interpolation requested but no dense output stored.
    NoDenseOutput,
    /// Interpolation point outside solution bounds.
    InterpOutOfBounds,
    /// Solver does not support interpolation.
    InterpNotImplemented,
    /// Jacobian matrix is singular (Rosenbrock solvers only).
    SingularJacobian,
}

impl fmt::Display for OdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StepNotFinite => write!(f, "step error is not finite"),
            Self::MaxStepsExceeded => write!(f, "maximum number of steps exceeded"),
            Self::NoDenseOutput => write!(f, "no dense output in solution"),
            Self::InterpOutOfBounds => write!(f, "interpolation point out of bounds"),
            Self::InterpNotImplemented => write!(f, "interpolation not implemented for this solver"),
            Self::SingularJacobian => write!(f, "Jacobian matrix is singular"),
        }
    }
}

/// Result of an adaptive integration.
pub struct Solution<T: FloatScalar, const S: usize> {
    /// Final independent variable value.
    pub t: T,
    /// Final state vector.
    pub y: Vector<T, S>,
    /// Total derivative evaluations.
    pub evals: usize,
    /// Accepted steps.
    pub accepted: usize,
    /// Rejected steps.
    pub rejected: usize,
    /// Dense output data (requires `std` feature).
    #[cfg(feature = "std")]
    pub dense: Option<DenseOutput<T, S>>,
}

/// Stored data for dense interpolation between accepted steps.
#[cfg(feature = "std")]
pub struct DenseOutput<T: FloatScalar, const S: usize> {
    /// Independent variable at start of each accepted step.
    pub t: Vec<T>,
    /// Step size of each accepted step.
    pub h: Vec<T>,
    /// All k-stages at each accepted step.
    pub stages: Vec<Vec<Vector<T, S>>>,
    /// State at start of each accepted step.
    pub y: Vec<Vector<T, S>>,
}
