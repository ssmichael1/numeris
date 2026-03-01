use crate::traits::FloatScalar;

use super::{InterpError, find_interval, validate_sorted};

/// Natural cubic spline interpolant (fixed-size, stack-allocated).
///
/// Uses natural boundary conditions (S''(x₀) = S''(x_{N-1}) = 0). The tridiagonal
/// system for second derivatives is solved via the Thomas algorithm in O(N).
/// Requires at least 3 points.
///
/// Each segment stores coefficients `[a, b, c, d]` for:
/// `S_i(x) = a + b·(x - x_i) + c·(x - x_i)² + d·(x - x_i)³`
///
/// # Example
///
/// ```
/// use numeris::interp::CubicSpline;
///
/// let xs = [0.0_f64, 1.0, 2.0, 3.0];
/// let ys = [0.0, 1.0, 0.0, 1.0];
/// let spline = CubicSpline::new(xs, ys).unwrap();
///
/// // Passes through knots exactly
/// assert!((spline.eval(0.0) - 0.0).abs() < 1e-14);
/// assert!((spline.eval(1.0) - 1.0).abs() < 1e-14);
/// assert!((spline.eval(2.0) - 0.0).abs() < 1e-14);
/// ```
#[derive(Debug, Clone)]
pub struct CubicSpline<T, const N: usize> {
    xs: [T; N],
    // Per-segment coefficients [a, b, c, d]. Only indices 0..N-1 used.
    // We store N entries (not N-1) to avoid unstable generic_const_exprs.
    coeffs: [[T; 4]; N],
}

impl<T: FloatScalar, const N: usize> CubicSpline<T, N> {
    /// Construct a natural cubic spline from sorted knots.
    ///
    /// Returns `InterpError::TooFewPoints` if `N < 3`,
    /// `InterpError::NotSorted` if `xs` is not strictly increasing.
    pub fn new(xs: [T; N], ys: [T; N]) -> Result<Self, InterpError> {
        if N < 3 {
            return Err(InterpError::TooFewPoints);
        }
        validate_sorted(&xs)?;

        let two = T::one() + T::one();
        let three = two + T::one();
        let six = three + three;

        // Thomas algorithm for natural cubic spline.
        // Tridiagonal system for second derivatives m_i (interior points i=1..N-2).
        // Natural BCs: m_0 = m_{N-1} = 0.
        //
        // For interior point i:
        //   h_{i-1}·m_{i-1} + 2(h_{i-1}+h_i)·m_i + h_i·m_{i+1} = 6·(δ_i - δ_{i-1})
        // where h_i = x_{i+1} - x_i, δ_i = (y_{i+1} - y_i) / h_i

        let n = N;
        // h[i] = x[i+1] - x[i], delta[i] = (y[i+1] - y[i]) / h[i]
        let mut h = [T::zero(); N];
        let mut delta = [T::zero(); N];
        for i in 0..n - 1 {
            h[i] = xs[i + 1] - xs[i];
            delta[i] = (ys[i + 1] - ys[i]) / h[i];
        }

        // Solve tridiagonal system for m[1..n-2]
        // Use m array for the solution, m[0] = m[n-1] = 0
        let mut m = [T::zero(); N];

        if n > 3 {
            // Forward sweep arrays (use coeffs temporarily)
            let mut cp = [T::zero(); N]; // modified upper diagonal
            let mut dp = [T::zero(); N]; // modified RHS

            // Row i=1 (first interior point)
            let diag = two * (h[0] + h[1]);
            let rhs = six * (delta[1] - delta[0]);
            cp[1] = h[1] / diag;
            dp[1] = rhs / diag;

            // Forward sweep i=2..n-2
            for i in 2..n - 1 {
                let diag_i = two * (h[i - 1] + h[i]) - h[i - 1] * cp[i - 1];
                let rhs_i = six * (delta[i] - delta[i - 1]) - h[i - 1] * dp[i - 1];
                if i < n - 1 {
                    cp[i] = h[i] / diag_i;
                }
                dp[i] = rhs_i / diag_i;
            }

            // Back substitution
            m[n - 2] = dp[n - 2];
            for i in (1..n - 2).rev() {
                m[i] = dp[i] - cp[i] * m[i + 1];
            }
        } else {
            // n == 3: single interior point, direct solve
            let diag = two * (h[0] + h[1]);
            m[1] = six * (delta[1] - delta[0]) / diag;
        }

        // Compute per-segment coefficients
        let mut coeffs = [[T::zero(); 4]; N];
        for i in 0..n - 1 {
            let a = ys[i];
            let b = delta[i] - h[i] * (two * m[i] + m[i + 1]) / six;
            let c = m[i] / two;
            let d = (m[i + 1] - m[i]) / (six * h[i]);
            coeffs[i] = [a, b, c, d];
        }

        Ok(Self { xs, coeffs })
    }

    /// Evaluate the spline at `x`.
    pub fn eval(&self, x: T) -> T {
        let i = find_interval(&self.xs, x);
        let dx = x - self.xs[i];
        let [a, b, c, d] = self.coeffs[i];
        // Horner form: a + dx·(b + dx·(c + dx·d))
        a + dx * (b + dx * (c + dx * d))
    }

    /// Evaluate the spline and its derivative at `x`.
    pub fn eval_derivative(&self, x: T) -> (T, T) {
        let i = find_interval(&self.xs, x);
        let dx = x - self.xs[i];
        let [a, b, c, d] = self.coeffs[i];
        let two = T::one() + T::one();
        let three = two + T::one();
        let val = a + dx * (b + dx * (c + dx * d));
        let dval = b + dx * (two * c + three * d * dx);
        (val, dval)
    }

    /// The knot x-values.
    pub fn xs(&self) -> &[T; N] {
        &self.xs
    }
}

// ---------- Dynamic variant ----------

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Natural cubic spline interpolant (heap-allocated, runtime-sized).
///
/// Dynamic counterpart of [`CubicSpline`]. Requires at least 3 points.
///
/// # Example
///
/// ```
/// use numeris::interp::DynCubicSpline;
///
/// let xs = vec![0.0_f64, 1.0, 2.0, 3.0];
/// let ys = vec![0.0, 1.0, 0.0, 1.0];
/// let spline = DynCubicSpline::new(xs, ys).unwrap();
/// assert!((spline.eval(1.0) - 1.0).abs() < 1e-14);
/// ```
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct DynCubicSpline<T> {
    xs: Vec<T>,
    coeffs: Vec<[T; 4]>,
}

#[cfg(feature = "alloc")]
impl<T: FloatScalar> DynCubicSpline<T> {
    /// Construct a natural cubic spline from sorted knots.
    pub fn new(xs: Vec<T>, ys: Vec<T>) -> Result<Self, InterpError> {
        if xs.len() != ys.len() {
            return Err(InterpError::LengthMismatch);
        }
        if xs.len() < 3 {
            return Err(InterpError::TooFewPoints);
        }
        validate_sorted(&xs)?;

        let n = xs.len();
        let two = T::one() + T::one();
        let three = two + T::one();
        let six = three + three;

        let mut h = alloc::vec![T::zero(); n];
        let mut delta = alloc::vec![T::zero(); n];
        for i in 0..n - 1 {
            h[i] = xs[i + 1] - xs[i];
            delta[i] = (ys[i + 1] - ys[i]) / h[i];
        }

        let mut m = alloc::vec![T::zero(); n];

        if n > 3 {
            let mut cp = alloc::vec![T::zero(); n];
            let mut dp = alloc::vec![T::zero(); n];

            let diag = two * (h[0] + h[1]);
            let rhs = six * (delta[1] - delta[0]);
            cp[1] = h[1] / diag;
            dp[1] = rhs / diag;

            for i in 2..n - 1 {
                let diag_i = two * (h[i - 1] + h[i]) - h[i - 1] * cp[i - 1];
                let rhs_i = six * (delta[i] - delta[i - 1]) - h[i - 1] * dp[i - 1];
                if i < n - 1 {
                    cp[i] = h[i] / diag_i;
                }
                dp[i] = rhs_i / diag_i;
            }

            m[n - 2] = dp[n - 2];
            for i in (1..n - 2).rev() {
                m[i] = dp[i] - cp[i] * m[i + 1];
            }
        } else {
            let diag = two * (h[0] + h[1]);
            m[1] = six * (delta[1] - delta[0]) / diag;
        }

        let mut coeffs = alloc::vec![[T::zero(); 4]; n - 1];
        for i in 0..n - 1 {
            let a = ys[i];
            let b = delta[i] - h[i] * (two * m[i] + m[i + 1]) / six;
            let c = m[i] / two;
            let d = (m[i + 1] - m[i]) / (six * h[i]);
            coeffs[i] = [a, b, c, d];
        }

        Ok(Self { xs, coeffs })
    }

    /// Evaluate the spline at `x`.
    pub fn eval(&self, x: T) -> T {
        let i = find_interval(&self.xs, x);
        let dx = x - self.xs[i];
        let [a, b, c, d] = self.coeffs[i];
        a + dx * (b + dx * (c + dx * d))
    }

    /// Evaluate the spline and its derivative at `x`.
    pub fn eval_derivative(&self, x: T) -> (T, T) {
        let i = find_interval(&self.xs, x);
        let dx = x - self.xs[i];
        let [a, b, c, d] = self.coeffs[i];
        let two = T::one() + T::one();
        let three = two + T::one();
        let val = a + dx * (b + dx * (c + dx * d));
        let dval = b + dx * (two * c + three * d * dx);
        (val, dval)
    }

    /// The knot x-values.
    pub fn xs(&self) -> &[T] {
        &self.xs
    }
}
