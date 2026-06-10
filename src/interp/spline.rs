use crate::traits::FloatScalar;

use super::{find_interval, validate_sorted, InterpError};

/// Evaluate the segment cubic `a + dx·(b + dx·(c + dx·d))` (Horner form).
#[inline]
fn cubic_eval<T: FloatScalar>(dx: T, [a, b, c, d]: [T; 4]) -> T {
    a + dx * (b + dx * (c + dx * d))
}

/// Evaluate the segment cubic and its derivative at `dx`.
#[inline]
fn cubic_eval_with_derivative<T: FloatScalar>(dx: T, coeffs: [T; 4]) -> (T, T) {
    let [_, b, c, d] = coeffs;
    let two = T::one() + T::one();
    let three = two + T::one();
    let val = cubic_eval(dx, coeffs);
    let dval = b + dx * (two * c + three * d * dx);
    (val, dval)
}

/// Solve the natural-spline tridiagonal system (Thomas algorithm) and fill
/// per-segment Horner coefficients `[a, b, c, d]`.
///
/// Shared by the fixed-size and dynamic constructors. `h`, `delta`, `cp`, and
/// `m` are caller-provided scratch of length `n = xs.len()`; `m` must be
/// zero-initialized (the natural boundary conditions `m₀ = m_{n−1} = 0` are
/// never written) and doubles as the modified RHS during the forward sweep,
/// holding the knot second derivatives afterwards. `coeffs` receives the
/// `n − 1` segment coefficient rows.
fn natural_spline_coeffs<T: FloatScalar>(
    xs: &[T],
    ys: &[T],
    h: &mut [T],
    delta: &mut [T],
    cp: &mut [T],
    m: &mut [T],
    coeffs: &mut [[T; 4]],
) -> Result<(), InterpError> {
    let n = xs.len();
    let two = T::one() + T::one();
    let three = two + T::one();
    let six = three + three;

    // h[i] = x[i+1] - x[i], delta[i] = (y[i+1] - y[i]) / h[i]
    for i in 0..n - 1 {
        h[i] = xs[i + 1] - xs[i];
        delta[i] = (ys[i + 1] - ys[i]) / h[i];
    }

    // Tridiagonal system for second derivatives m_i (interior points i=1..n-2):
    //   h_{i-1}·m_{i-1} + 2(h_{i-1}+h_i)·m_i + h_i·m_{i+1} = 6·(δ_i - δ_{i-1})
    if n > 3 {
        // Row i=1 (first interior point)
        let diag = two * (h[0] + h[1]);
        if diag.abs() < T::epsilon() {
            return Err(InterpError::IllConditioned);
        }
        cp[1] = h[1] / diag;
        m[1] = six * (delta[1] - delta[0]) / diag;

        // Forward sweep i=2..n-2
        for i in 2..n - 1 {
            let diag_i = two * (h[i - 1] + h[i]) - h[i - 1] * cp[i - 1];
            if diag_i.abs() < T::epsilon() {
                return Err(InterpError::IllConditioned);
            }
            let rhs_i = six * (delta[i] - delta[i - 1]) - h[i - 1] * m[i - 1];
            cp[i] = h[i] / diag_i;
            m[i] = rhs_i / diag_i;
        }

        // Back substitution (in place: m currently holds the modified RHS)
        for i in (1..n - 2).rev() {
            m[i] = m[i] - cp[i] * m[i + 1];
        }
    } else {
        // n == 3: single interior point, direct solve
        let diag = two * (h[0] + h[1]);
        if diag.abs() < T::epsilon() {
            return Err(InterpError::IllConditioned);
        }
        m[1] = six * (delta[1] - delta[0]) / diag;
    }

    // Per-segment coefficients
    for i in 0..n - 1 {
        coeffs[i] = [
            ys[i],
            delta[i] - h[i] * (two * m[i] + m[i + 1]) / six,
            m[i] / two,
            (m[i + 1] - m[i]) / (six * h[i]),
        ];
    }

    Ok(())
}

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

        let mut h = [T::zero(); N];
        let mut delta = [T::zero(); N];
        let mut cp = [T::zero(); N];
        let mut m = [T::zero(); N];
        let mut coeffs = [[T::zero(); 4]; N];
        natural_spline_coeffs(
            &xs,
            &ys,
            &mut h,
            &mut delta,
            &mut cp,
            &mut m,
            &mut coeffs[..N - 1],
        )?;

        Ok(Self { xs, coeffs })
    }

    /// Evaluate the spline at `x`.
    pub fn eval(&self, x: T) -> T {
        let i = find_interval(&self.xs, x);
        cubic_eval(x - self.xs[i], self.coeffs[i])
    }

    /// Evaluate the spline and its derivative at `x`.
    pub fn eval_derivative(&self, x: T) -> (T, T) {
        let i = find_interval(&self.xs, x);
        cubic_eval_with_derivative(x - self.xs[i], self.coeffs[i])
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
        let mut h = alloc::vec![T::zero(); n];
        let mut delta = alloc::vec![T::zero(); n];
        let mut cp = alloc::vec![T::zero(); n];
        let mut m = alloc::vec![T::zero(); n];
        let mut coeffs = alloc::vec![[T::zero(); 4]; n - 1];
        natural_spline_coeffs(&xs, &ys, &mut h, &mut delta, &mut cp, &mut m, &mut coeffs)?;

        Ok(Self { xs, coeffs })
    }

    /// Evaluate the spline at `x`.
    pub fn eval(&self, x: T) -> T {
        let i = find_interval(&self.xs, x);
        cubic_eval(x - self.xs[i], self.coeffs[i])
    }

    /// Evaluate the spline and its derivative at `x`.
    pub fn eval_derivative(&self, x: T) -> (T, T) {
        let i = find_interval(&self.xs, x);
        cubic_eval_with_derivative(x - self.xs[i], self.coeffs[i])
    }

    /// The knot x-values.
    pub fn xs(&self) -> &[T] {
        &self.xs
    }
}
