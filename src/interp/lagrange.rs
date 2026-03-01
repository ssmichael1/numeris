use crate::traits::FloatScalar;

use super::{InterpError, validate_sorted};

/// Barycentric Lagrange interpolant (fixed-size, stack-allocated).
///
/// Precomputes barycentric weights in O(N²), then evaluates in O(N).
/// Numerically stable for moderate N. Requires at least 2 points.
///
/// # Example
///
/// ```
/// use numeris::interp::LagrangeInterp;
///
/// // Interpolate x² through 3 points — exact for polynomials of degree ≤ 2
/// let xs = [0.0_f64, 1.0, 2.0];
/// let ys = [0.0, 1.0, 4.0]; // y = x²
/// let interp = LagrangeInterp::new(xs, ys).unwrap();
/// assert!((interp.eval(1.5) - 2.25).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct LagrangeInterp<T, const N: usize> {
    xs: [T; N],
    ys: [T; N],
    ws: [T; N], // barycentric weights
}

impl<T: FloatScalar, const N: usize> LagrangeInterp<T, N> {
    /// Construct a barycentric Lagrange interpolant.
    pub fn new(xs: [T; N], ys: [T; N]) -> Result<Self, InterpError> {
        if N < 2 {
            return Err(InterpError::TooFewPoints);
        }
        validate_sorted(&xs)?;

        // Compute barycentric weights: w_j = 1 / ∏_{k≠j} (x_j - x_k)
        let mut ws = [T::one(); N];
        for j in 0..N {
            for k in 0..N {
                if k != j {
                    ws[j] = ws[j] / (xs[j] - xs[k]);
                }
            }
        }

        Ok(Self { xs, ys, ws })
    }

    /// Evaluate the interpolant at `x`.
    ///
    /// Uses the barycentric formula in O(N):
    /// `L(x) = (Σ w_j·y_j/(x-x_j)) / (Σ w_j/(x-x_j))`
    pub fn eval(&self, x: T) -> T {
        barycentric_eval(&self.xs, &self.ys, &self.ws, x)
    }

    /// Evaluate the interpolant and its derivative at `x`.
    pub fn eval_derivative(&self, x: T) -> (T, T) {
        barycentric_eval_deriv(&self.xs, &self.ys, &self.ws, x)
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

/// Barycentric Lagrange interpolant (heap-allocated, runtime-sized).
///
/// Dynamic counterpart of [`LagrangeInterp`]. Requires at least 2 points.
///
/// # Example
///
/// ```
/// use numeris::interp::DynLagrangeInterp;
///
/// let interp = DynLagrangeInterp::new(
///     vec![0.0_f64, 1.0, 2.0],
///     vec![0.0, 1.0, 4.0],
/// ).unwrap();
/// assert!((interp.eval(1.5) - 2.25).abs() < 1e-12);
/// ```
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct DynLagrangeInterp<T> {
    xs: Vec<T>,
    ys: Vec<T>,
    ws: Vec<T>,
}

#[cfg(feature = "alloc")]
impl<T: FloatScalar> DynLagrangeInterp<T> {
    /// Construct a barycentric Lagrange interpolant.
    pub fn new(xs: Vec<T>, ys: Vec<T>) -> Result<Self, InterpError> {
        if xs.len() != ys.len() {
            return Err(InterpError::LengthMismatch);
        }
        if xs.len() < 2 {
            return Err(InterpError::TooFewPoints);
        }
        validate_sorted(&xs)?;

        let n = xs.len();
        let mut ws = alloc::vec![T::one(); n];
        for j in 0..n {
            for k in 0..n {
                if k != j {
                    ws[j] = ws[j] / (xs[j] - xs[k]);
                }
            }
        }

        Ok(Self { xs, ys, ws })
    }

    /// Evaluate the interpolant at `x`.
    pub fn eval(&self, x: T) -> T {
        barycentric_eval(&self.xs, &self.ys, &self.ws, x)
    }

    /// Evaluate the interpolant and its derivative at `x`.
    pub fn eval_derivative(&self, x: T) -> (T, T) {
        barycentric_eval_deriv(&self.xs, &self.ys, &self.ws, x)
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

// ---------- Shared helpers ----------

/// Barycentric evaluation: L(x) = (Σ w_j·y_j/(x-x_j)) / (Σ w_j/(x-x_j))
fn barycentric_eval<T: FloatScalar>(xs: &[T], ys: &[T], ws: &[T], x: T) -> T {
    let eps = T::epsilon() * T::from(1e3).unwrap_or(T::epsilon());
    let mut numer = T::zero();
    let mut denom = T::zero();

    for j in 0..xs.len() {
        let diff = x - xs[j];
        if diff.abs() < eps {
            return ys[j];
        }
        let term = ws[j] / diff;
        numer = numer + term * ys[j];
        denom = denom + term;
    }
    numer / denom
}

/// Barycentric evaluation with derivative via quotient rule.
///
/// L(x) = N/D where N = Σ w_j·y_j/(x-x_j), D = Σ w_j/(x-x_j)
/// L'(x) = (N'·D - N·D') / D² where N' = -Σ w_j·y_j/(x-x_j)², D' = -Σ w_j/(x-x_j)²
fn barycentric_eval_deriv<T: FloatScalar>(xs: &[T], ys: &[T], ws: &[T], x: T) -> (T, T) {
    let eps = T::epsilon() * T::from(1e3).unwrap_or(T::epsilon());

    // Check if x coincides with a knot
    for j in 0..xs.len() {
        let diff = x - xs[j];
        if diff.abs() < eps {
            // Value is exactly y_j; compute derivative via l'Hôpital on barycentric form.
            // L'(x_j) = -Σ_{k≠j} (w_k / w_j) · (y_j - y_k) / (x_j - x_k)
            let mut deriv = T::zero();
            for k in 0..xs.len() {
                if k != j {
                    deriv = deriv - ws[k] / ws[j] * (ys[j] - ys[k]) / (xs[j] - xs[k]);
                }
            }
            return (ys[j], deriv);
        }
    }

    let mut n = T::zero();
    let mut d = T::zero();
    let mut np = T::zero();
    let mut dp = T::zero();

    for j in 0..xs.len() {
        let diff = x - xs[j];
        let inv = T::one() / diff;
        let inv2 = inv * inv;
        let wj = ws[j];
        let term = wj * inv;
        n = n + term * ys[j];
        d = d + term;
        np = np - wj * inv2 * ys[j];
        dp = dp - wj * inv2;
    }

    let val = n / d;
    let dval = (np * d - n * dp) / (d * d);
    (val, dval)
}
