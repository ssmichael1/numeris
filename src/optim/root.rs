use crate::traits::FloatScalar;

use super::{OptimError, RootResult};

/// Settings for scalar root-finding algorithms.
#[derive(Debug, Clone, Copy)]
pub struct RootSettings<T> {
    /// Convergence tolerance on the bracket width `|b - a|`.
    pub x_tol: T,
    /// Convergence tolerance on the function value `|f(x)|`.
    pub f_tol: T,
    /// Maximum number of iterations.
    pub max_iter: usize,
}

impl Default for RootSettings<f64> {
    fn default() -> Self {
        Self {
            x_tol: 1e-12,
            f_tol: 1e-12,
            max_iter: 100,
        }
    }
}

impl Default for RootSettings<f32> {
    fn default() -> Self {
        Self {
            x_tol: 1e-6,
            f_tol: 1e-6,
            max_iter: 100,
        }
    }
}

/// Brent's method for bracketed root finding.
///
/// Combines bisection, secant, and inverse quadratic interpolation for
/// superlinear convergence while guaranteeing the bracket shrinks every step.
///
/// # Arguments
///
/// * `f` — function whose root is sought
/// * `a`, `b` — bracket endpoints; must satisfy `f(a) * f(b) < 0`
/// * `settings` — convergence tolerances and iteration limit
///
/// # Errors
///
/// Returns [`OptimError::BracketInvalid`] if `f(a)` and `f(b)` have the same sign.
/// Returns [`OptimError::MaxIterations`] if convergence is not achieved.
///
/// # Example
///
/// ```
/// use numeris::optim::{brent, RootSettings};
///
/// // Find √2 as root of x² - 2
/// let r = brent(|x| x * x - 2.0, 0.0, 2.0, &RootSettings::default()).unwrap();
/// assert!((r.x - core::f64::consts::SQRT_2).abs() < 1e-12);
/// ```
pub fn brent<T: FloatScalar>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    settings: &RootSettings<T>,
) -> Result<RootResult<T>, OptimError> {
    let mut a = a;
    let mut b = b;
    let mut fa = f(a);
    let mut fb = f(b);
    let mut evals = 2usize;

    // Check bracket validity
    if (fa > T::zero()) == (fb > T::zero()) {
        return Err(OptimError::BracketInvalid);
    }

    // Ensure |f(a)| >= |f(b)| so b is the best approximation
    if fa.abs() < fb.abs() {
        core::mem::swap(&mut a, &mut b);
        core::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut mflag = true;

    for iter in 0..settings.max_iter {
        // Convergence checks
        if fb.abs() < settings.f_tol || (b - a).abs() < settings.x_tol {
            return Ok(RootResult {
                x: b,
                fx: fb,
                iterations: iter,
                evals,
            });
        }

        let mut s;

        if fa != fc && fb != fc {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        // Conditions for rejecting interpolation and using bisection instead
        let three_quarter = (T::from(3).unwrap() * a + b) / T::from(4).unwrap();
        let cond1 = if three_quarter < b {
            s < three_quarter || s > b
        } else {
            s > three_quarter || s < b
        };
        let cond2 = mflag && (s - b).abs() >= (b - c).abs() / T::from(2).unwrap();
        let cond3 = !mflag && (s - b).abs() >= (c - d).abs() / T::from(2).unwrap();
        let cond4 = mflag && (b - c).abs() < settings.x_tol;
        let cond5 = !mflag && (c - d).abs() < settings.x_tol;

        if cond1 || cond2 || cond3 || cond4 || cond5 {
            // Bisection
            s = (a + b) / T::from(2).unwrap();
            mflag = true;
        } else {
            mflag = false;
        }

        let fs = f(s);
        evals += 1;

        d = c;
        c = b;
        fc = fb;

        if (fa > T::zero()) != (fs > T::zero()) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        // Keep |f(a)| >= |f(b)|
        if fa.abs() < fb.abs() {
            core::mem::swap(&mut a, &mut b);
            core::mem::swap(&mut fa, &mut fb);
        }

    }

    Err(OptimError::MaxIterations)
}

/// Newton's method for scalar root finding.
///
/// Uses `x_{n+1} = x_n - f(x_n) / f'(x_n)` with user-supplied derivative.
///
/// # Arguments
///
/// * `f` — function whose root is sought
/// * `df` — derivative of `f`
/// * `x0` — initial guess
/// * `settings` — convergence tolerances and iteration limit
///
/// # Errors
///
/// Returns [`OptimError::Singular`] if the derivative is near zero.
/// Returns [`OptimError::MaxIterations`] if convergence is not achieved.
///
/// # Example
///
/// ```
/// use numeris::optim::{newton_1d, RootSettings};
///
/// // Find √2 as root of x² - 2
/// let r = newton_1d(
///     |x| x * x - 2.0,
///     |x| 2.0 * x,
///     1.0,
///     &RootSettings::default(),
/// ).unwrap();
/// assert!((r.x - core::f64::consts::SQRT_2).abs() < 1e-12);
/// ```
pub fn newton_1d<T: FloatScalar>(
    mut f: impl FnMut(T) -> T,
    mut df: impl FnMut(T) -> T,
    x0: T,
    settings: &RootSettings<T>,
) -> Result<RootResult<T>, OptimError> {
    let mut x = x0;
    let mut fx = f(x);
    let mut evals = 1usize;

    for iter in 0..settings.max_iter {
        if fx.abs() < settings.f_tol {
            return Ok(RootResult {
                x,
                fx,
                iterations: iter,
                evals,
            });
        }

        let dfx = df(x);
        evals += 1;

        if dfx.abs() < T::epsilon() {
            return Err(OptimError::Singular);
        }

        let x_new = x - fx / dfx;

        if (x_new - x).abs() < settings.x_tol {
            fx = f(x_new);
            evals += 1;
            return Ok(RootResult {
                x: x_new,
                fx,
                iterations: iter + 1,
                evals,
            });
        }

        x = x_new;
        fx = f(x);
        evals += 1;
    }

    Err(OptimError::MaxIterations)
}
