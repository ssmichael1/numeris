use crate::traits::FloatScalar;

use super::{InterpError, find_interval, validate_sorted};

/// Cubic Hermite interpolant (fixed-size, stack-allocated).
///
/// Uses user-supplied derivatives at each knot. Each segment is a cubic
/// polynomial matching both value and first derivative at its endpoints.
/// Requires at least 2 points.
///
/// # Example
///
/// ```
/// use numeris::interp::HermiteInterp;
///
/// // Interpolate sin(x) with known derivatives cos(x)
/// let xs = [0.0_f64, 1.0, 2.0];
/// let ys = [0.0_f64.sin(), 1.0_f64.sin(), 2.0_f64.sin()];
/// let dys = [0.0_f64.cos(), 1.0_f64.cos(), 2.0_f64.cos()];
/// let interp = HermiteInterp::new(xs, ys, dys).unwrap();
/// let mid = 0.5;
/// assert!((interp.eval(mid) - mid.sin()).abs() < 0.002);
/// ```
#[derive(Debug, Clone)]
pub struct HermiteInterp<T, const N: usize> {
    xs: [T; N],
    ys: [T; N],
    dys: [T; N],
}

impl<T: FloatScalar, const N: usize> HermiteInterp<T, N> {
    /// Construct a Hermite interpolant from knots and derivatives.
    pub fn new(xs: [T; N], ys: [T; N], dys: [T; N]) -> Result<Self, InterpError> {
        if N < 2 {
            return Err(InterpError::TooFewPoints);
        }
        validate_sorted(&xs)?;
        Ok(Self { xs, ys, dys })
    }

    /// Evaluate the interpolant at `x`.
    pub fn eval(&self, x: T) -> T {
        let i = find_interval(&self.xs, x);
        let h = self.xs[i + 1] - self.xs[i];
        let t = (x - self.xs[i]) / h;
        let t2 = t * t;
        let t3 = t2 * t;
        let two = T::one() + T::one();
        let three = two + T::one();

        // Hermite basis: h00 = 2t³ - 3t² + 1, h10 = t³ - 2t² + t
        //                h01 = -2t³ + 3t², h11 = t³ - t²
        let h00 = two * t3 - three * t2 + T::one();
        let h10 = t3 - two * t2 + t;
        let h01 = (T::zero() - two) * t3 + three * t2;
        let h11 = t3 - t2;

        h00 * self.ys[i] + h10 * h * self.dys[i] + h01 * self.ys[i + 1] + h11 * h * self.dys[i + 1]
    }

    /// Evaluate the interpolant and its derivative at `x`.
    pub fn eval_derivative(&self, x: T) -> (T, T) {
        let i = find_interval(&self.xs, x);
        let h = self.xs[i + 1] - self.xs[i];
        let t = (x - self.xs[i]) / h;
        let t2 = t * t;
        let t3 = t2 * t;
        let two = T::one() + T::one();
        let three = two + T::one();
        let six = three + three;

        let h00 = two * t3 - three * t2 + T::one();
        let h10 = t3 - two * t2 + t;
        let h01 = (T::zero() - two) * t3 + three * t2;
        let h11 = t3 - t2;

        let val =
            h00 * self.ys[i] + h10 * h * self.dys[i] + h01 * self.ys[i + 1] + h11 * h * self.dys[i + 1];

        // d/dx = (1/h) d/dt of the basis
        // h00' = 6t² - 6t, h10' = 3t² - 4t + 1
        // h01' = -6t² + 6t, h11' = 3t² - 2t
        let dh00 = six * t2 - six * t;
        let dh10 = three * t2 - (two + two) * t + T::one();
        let dh01 = (T::zero() - six) * t2 + six * t;
        let dh11 = three * t2 - two * t;

        let dval = (dh00 * self.ys[i] + dh10 * h * self.dys[i] + dh01 * self.ys[i + 1]
            + dh11 * h * self.dys[i + 1])
            / h;

        (val, dval)
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

/// Cubic Hermite interpolant (heap-allocated, runtime-sized).
///
/// Dynamic counterpart of [`HermiteInterp`]. Requires at least 2 points.
///
/// # Example
///
/// ```
/// use numeris::interp::DynHermiteInterp;
///
/// let interp = DynHermiteInterp::new(
///     vec![0.0_f64, 1.0, 2.0],
///     vec![0.0, 1.0, 0.0],
///     vec![1.0, 0.0, -1.0],
/// ).unwrap();
/// assert!((interp.eval(0.5) - 0.625).abs() < 1e-14);
/// ```
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct DynHermiteInterp<T> {
    xs: Vec<T>,
    ys: Vec<T>,
    dys: Vec<T>,
}

#[cfg(feature = "alloc")]
impl<T: FloatScalar> DynHermiteInterp<T> {
    /// Construct a Hermite interpolant from knots and derivatives.
    pub fn new(xs: Vec<T>, ys: Vec<T>, dys: Vec<T>) -> Result<Self, InterpError> {
        if xs.len() != ys.len() || xs.len() != dys.len() {
            return Err(InterpError::LengthMismatch);
        }
        if xs.len() < 2 {
            return Err(InterpError::TooFewPoints);
        }
        validate_sorted(&xs)?;
        Ok(Self { xs, ys, dys })
    }

    /// Evaluate the interpolant at `x`.
    pub fn eval(&self, x: T) -> T {
        let i = find_interval(&self.xs, x);
        let h = self.xs[i + 1] - self.xs[i];
        let t = (x - self.xs[i]) / h;
        let t2 = t * t;
        let t3 = t2 * t;
        let two = T::one() + T::one();
        let three = two + T::one();

        let h00 = two * t3 - three * t2 + T::one();
        let h10 = t3 - two * t2 + t;
        let h01 = (T::zero() - two) * t3 + three * t2;
        let h11 = t3 - t2;

        h00 * self.ys[i] + h10 * h * self.dys[i] + h01 * self.ys[i + 1] + h11 * h * self.dys[i + 1]
    }

    /// Evaluate the interpolant and its derivative at `x`.
    pub fn eval_derivative(&self, x: T) -> (T, T) {
        let i = find_interval(&self.xs, x);
        let h = self.xs[i + 1] - self.xs[i];
        let t = (x - self.xs[i]) / h;
        let t2 = t * t;
        let t3 = t2 * t;
        let two = T::one() + T::one();
        let three = two + T::one();
        let six = three + three;

        let h00 = two * t3 - three * t2 + T::one();
        let h10 = t3 - two * t2 + t;
        let h01 = (T::zero() - two) * t3 + three * t2;
        let h11 = t3 - t2;

        let val =
            h00 * self.ys[i] + h10 * h * self.dys[i] + h01 * self.ys[i + 1] + h11 * h * self.dys[i + 1];

        let dh00 = six * t2 - six * t;
        let dh10 = three * t2 - (two + two) * t + T::one();
        let dh01 = (T::zero() - six) * t2 + six * t;
        let dh11 = three * t2 - two * t;

        let dval = (dh00 * self.ys[i] + dh10 * h * self.dys[i] + dh01 * self.ys[i + 1]
            + dh11 * h * self.dys[i + 1])
            / h;

        (val, dval)
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
