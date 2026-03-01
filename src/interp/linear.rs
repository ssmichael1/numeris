use crate::traits::FloatScalar;

use super::{InterpError, find_interval, validate_sorted};

/// Piecewise linear interpolant (fixed-size, stack-allocated).
///
/// Requires at least 2 points. Evaluations outside the knot range extrapolate
/// linearly from the nearest boundary segment.
///
/// # Example
///
/// ```
/// use numeris::interp::LinearInterp;
///
/// let interp = LinearInterp::new(
///     [0.0_f64, 1.0, 2.0],
///     [0.0, 2.0, 1.0],
/// ).unwrap();
/// assert!((interp.eval(0.5) - 1.0).abs() < 1e-14);
/// // derivative = slope of the segment
/// let (val, dval) = interp.eval_derivative(0.5);
/// assert!((dval - 2.0).abs() < 1e-14);
/// ```
#[derive(Debug, Clone)]
pub struct LinearInterp<T, const N: usize> {
    xs: [T; N],
    ys: [T; N],
}

impl<T: FloatScalar, const N: usize> LinearInterp<T, N> {
    /// Construct a linear interpolant from sorted knots.
    ///
    /// Returns `InterpError::TooFewPoints` if `N < 2`,
    /// `InterpError::NotSorted` if `xs` is not strictly increasing.
    pub fn new(xs: [T; N], ys: [T; N]) -> Result<Self, InterpError> {
        if N < 2 {
            return Err(InterpError::TooFewPoints);
        }
        validate_sorted(&xs)?;
        Ok(Self { xs, ys })
    }

    /// Evaluate the interpolant at `x`.
    pub fn eval(&self, x: T) -> T {
        let i = find_interval(&self.xs, x);
        let h = self.xs[i + 1] - self.xs[i];
        let t = (x - self.xs[i]) / h;
        self.ys[i] + t * (self.ys[i + 1] - self.ys[i])
    }

    /// Evaluate the interpolant and its derivative at `x`.
    ///
    /// Returns `(value, derivative)`. The derivative is the slope of the segment.
    pub fn eval_derivative(&self, x: T) -> (T, T) {
        let i = find_interval(&self.xs, x);
        let h = self.xs[i + 1] - self.xs[i];
        let t = (x - self.xs[i]) / h;
        let dy = self.ys[i + 1] - self.ys[i];
        (self.ys[i] + t * dy, dy / h)
    }

    /// The knot x-values.
    pub fn xs(&self) -> &[T; N] {
        &self.xs
    }

    /// The knot y-values.
    pub fn ys(&self) -> &[T; N] {
        &self.ys
    }
}

// ---------- Dynamic variant ----------

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Piecewise linear interpolant (heap-allocated, runtime-sized).
///
/// Dynamic counterpart of [`LinearInterp`]. Requires at least 2 points.
///
/// # Example
///
/// ```
/// use numeris::interp::DynLinearInterp;
///
/// let interp = DynLinearInterp::new(
///     vec![0.0_f64, 1.0, 2.0],
///     vec![0.0, 2.0, 1.0],
/// ).unwrap();
/// assert!((interp.eval(0.5) - 1.0).abs() < 1e-14);
/// ```
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct DynLinearInterp<T> {
    xs: Vec<T>,
    ys: Vec<T>,
}

#[cfg(feature = "alloc")]
impl<T: FloatScalar> DynLinearInterp<T> {
    /// Construct a linear interpolant from sorted knots.
    pub fn new(xs: Vec<T>, ys: Vec<T>) -> Result<Self, InterpError> {
        if xs.len() != ys.len() {
            return Err(InterpError::LengthMismatch);
        }
        if xs.len() < 2 {
            return Err(InterpError::TooFewPoints);
        }
        validate_sorted(&xs)?;
        Ok(Self { xs, ys })
    }

    /// Evaluate the interpolant at `x`.
    pub fn eval(&self, x: T) -> T {
        let i = find_interval(&self.xs, x);
        let h = self.xs[i + 1] - self.xs[i];
        let t = (x - self.xs[i]) / h;
        self.ys[i] + t * (self.ys[i + 1] - self.ys[i])
    }

    /// Evaluate the interpolant and its derivative at `x`.
    pub fn eval_derivative(&self, x: T) -> (T, T) {
        let i = find_interval(&self.xs, x);
        let h = self.xs[i + 1] - self.xs[i];
        let t = (x - self.xs[i]) / h;
        let dy = self.ys[i + 1] - self.ys[i];
        (self.ys[i] + t * dy, dy / h)
    }

    /// The knot x-values.
    pub fn xs(&self) -> &[T] {
        &self.xs
    }

    /// The knot y-values.
    pub fn ys(&self) -> &[T] {
        &self.ys
    }
}
