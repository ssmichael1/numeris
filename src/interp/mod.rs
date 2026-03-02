//! Interpolation: linear, Hermite, barycentric Lagrange, natural cubic spline,
//! and bilinear (2D).
//!
//! All interpolants are constructed from sorted knots and support evaluation at
//! arbitrary points. Out-of-bounds queries extrapolate using the nearest boundary
//! segment. Each method provides both a fixed-size (const-generic, stack-allocated,
//! no-std) and a dynamic (`Dyn*`, requires `alloc`) variant.
//!
//! # Examples
//!
//! ```
//! use numeris::interp::LinearInterp;
//!
//! let xs = [0.0_f64, 1.0, 2.0, 3.0];
//! let ys = [0.0, 1.0, 0.0, 1.0];
//! let interp = LinearInterp::new(xs, ys).unwrap();
//! assert!((interp.eval(0.5) - 0.5).abs() < 1e-14);
//! ```

mod bilinear;
mod hermite;
mod lagrange;
mod linear;
mod spline;

#[cfg(test)]
mod tests;

pub use bilinear::BilinearInterp;
pub use hermite::HermiteInterp;
pub use lagrange::LagrangeInterp;
pub use linear::LinearInterp;
pub use spline::CubicSpline;

#[cfg(feature = "alloc")]
pub use bilinear::DynBilinearInterp;
#[cfg(feature = "alloc")]
pub use hermite::DynHermiteInterp;
#[cfg(feature = "alloc")]
pub use lagrange::DynLagrangeInterp;
#[cfg(feature = "alloc")]
pub use linear::DynLinearInterp;
#[cfg(feature = "alloc")]
pub use spline::DynCubicSpline;

use crate::traits::FloatScalar;

/// Errors from interpolant construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpError {
    /// Not enough data points for the interpolation method.
    TooFewPoints,
    /// The `xs` array is not strictly increasing.
    NotSorted,
    /// `xs` and `ys` have different lengths (dynamic variants only).
    LengthMismatch,
}

impl core::fmt::Display for InterpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InterpError::TooFewPoints => write!(f, "not enough data points for interpolation"),
            InterpError::NotSorted => write!(f, "x values must be strictly increasing"),
            InterpError::LengthMismatch => write!(f, "xs and ys must have the same length"),
        }
    }
}

/// Validate that a slice is strictly increasing.
fn validate_sorted<T: FloatScalar>(xs: &[T]) -> Result<(), InterpError> {
    for i in 1..xs.len() {
        if xs[i] <= xs[i - 1] {
            return Err(InterpError::NotSorted);
        }
    }
    Ok(())
}

/// Binary search for the interval containing `x` in a sorted slice.
///
/// Returns index `i` such that `xs[i] <= x < xs[i+1]`, clamped to
/// `[0, xs.len() - 2]` for extrapolation beyond boundaries.
fn find_interval<T: FloatScalar>(xs: &[T], x: T) -> usize {
    debug_assert!(xs.len() >= 2);
    let n = xs.len();
    // Clamp to valid segment range
    if x <= xs[0] {
        return 0;
    }
    if x >= xs[n - 1] {
        return n - 2;
    }
    // Binary search
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if x < xs[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    lo
}
