use crate::matrix::vector::Vector;
use crate::traits::FloatScalar;
use crate::Matrix;

use super::line_search::backtracking_armijo;
use super::{MinimizeResult, OptimError};

/// Settings for BFGS quasi-Newton minimization.
#[derive(Debug, Clone, Copy)]
pub struct BfgsSettings<T> {
    /// Convergence tolerance on gradient norm.
    pub grad_tol: T,
    /// Convergence tolerance on relative function change.
    pub f_tol: T,
    /// Convergence tolerance on relative step size.
    pub x_tol: T,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Armijo condition parameter (sufficient decrease).
    pub armijo_c1: T,
    /// Backtracking contraction factor.
    pub armijo_rho: T,
    /// Maximum line search iterations.
    pub max_ls_iter: usize,
}

impl Default for BfgsSettings<f64> {
    fn default() -> Self {
        Self {
            grad_tol: 1e-8,
            f_tol: 1e-12,
            x_tol: 1e-12,
            max_iter: 200,
            armijo_c1: 1e-4,
            armijo_rho: 0.5,
            max_ls_iter: 40,
        }
    }
}

impl Default for BfgsSettings<f32> {
    fn default() -> Self {
        Self {
            grad_tol: 1e-4,
            f_tol: 1e-6,
            x_tol: 1e-6,
            max_iter: 200,
            armijo_c1: 1e-4,
            armijo_rho: 0.5,
            max_ls_iter: 40,
        }
    }
}

/// Minimize a function using the BFGS quasi-Newton method.
///
/// Maintains an inverse Hessian approximation `H` on the stack (N×N matrix).
/// Uses Armijo backtracking line search for step length selection.
///
/// # Arguments
///
/// * `f` — objective function `f: R^N → R`
/// * `grad` — gradient `∇f: R^N → R^N`
/// * `x0` — initial guess
/// * `settings` — convergence tolerances and algorithm parameters
///
/// # Errors
///
/// Returns [`OptimError::MaxIterations`] if convergence is not achieved.
/// Returns [`OptimError::LineSearchFailed`] if the line search fails.
///
/// # Example
///
/// ```
/// use numeris::optim::{minimize_bfgs, BfgsSettings};
/// use numeris::Vector;
///
/// // Minimize f(x) = (x0-1)^2 + (x1-2)^2
/// let r = minimize_bfgs(
///     |x: &Vector<f64, 2>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2),
///     |x: &Vector<f64, 2>| Vector::from_array([2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)]),
///     &Vector::from_array([0.0, 0.0]),
///     &BfgsSettings::default(),
/// ).unwrap();
/// assert!((r.x[0] - 1.0).abs() < 1e-6);
/// assert!((r.x[1] - 2.0).abs() < 1e-6);
/// ```
pub fn minimize_bfgs<T: FloatScalar, const N: usize>(
    mut f: impl FnMut(&Vector<T, N>) -> T,
    mut grad: impl FnMut(&Vector<T, N>) -> Vector<T, N>,
    x0: &Vector<T, N>,
    settings: &BfgsSettings<T>,
) -> Result<MinimizeResult<T, N>, OptimError> {
    let mut x = *x0;
    let mut fx = f(&x);
    let mut g = grad(&x);
    let mut f_evals = 1usize;
    let mut grad_evals = 1usize;

    // Inverse Hessian approximation, initialized to identity
    let mut h: Matrix<T, N, N> = Matrix::eye();

    for iter in 0..settings.max_iter {
        let g_norm = g.norm();

        // Gradient norm convergence
        if g_norm < settings.grad_tol {
            return Ok(MinimizeResult {
                x,
                fx,
                grad_norm: g_norm,
                iterations: iter,
                f_evals,
                grad_evals,
            });
        }

        // Search direction: p = -H * g
        let p = -h.vecmul(&g);

        // Check descent direction; reset H if not
        let grad_dot_p = g.dot(&p);
        let (p, h_reset) = if grad_dot_p >= T::zero() {
            // Not a descent direction, reset H to identity
            (-g, true)
        } else {
            (p, false)
        };

        let grad_dot_p = if h_reset { -g.dot(&g) } else { grad_dot_p };

        // Armijo line search
        let (alpha, f_new, ls_evals) = backtracking_armijo(
            fx,
            grad_dot_p,
            &x,
            &p,
            &mut f,
            settings.armijo_c1,
            settings.armijo_rho,
            settings.max_ls_iter,
        )?;
        f_evals += ls_evals;

        let s = p * alpha; // step
        let x_new = x + s;
        let g_new = grad(&x_new);
        grad_evals += 1;

        let y = g_new - g; // gradient change
        let ys = y.dot(&s);

        // Relative step size convergence
        if s.norm() < settings.x_tol * (T::one() + x.norm()) {
            return Ok(MinimizeResult {
                x: x_new,
                fx: f_new,
                grad_norm: g_new.norm(),
                iterations: iter + 1,
                f_evals,
                grad_evals,
            });
        }

        // Relative function change convergence
        if (fx - f_new).abs() < settings.f_tol * (T::one() + fx.abs()) {
            return Ok(MinimizeResult {
                x: x_new,
                fx: f_new,
                grad_norm: g_new.norm(),
                iterations: iter + 1,
                f_evals,
                grad_evals,
            });
        }

        // BFGS update of inverse Hessian
        if h_reset {
            h = Matrix::eye();
        }

        if ys > T::epsilon() {
            let rho = T::one() / ys;

            // H += (1 + rho * y^T * H * y) * rho * s * s^T - rho * (H*y*s^T + s*y^T*H)
            let hy = h.vecmul(&y);
            let yhy = y.dot(&hy);

            // Factor: (1 + rho * y^T H y) * rho
            let factor = (T::one() + rho * yhy) * rho;

            // Update: H = H + factor * s⊗s - rho * (Hy⊗s + s⊗(H^T y))
            // Since H is symmetric, H^T y = H y = hy, and y^T H = hy^T
            for i in 0..N {
                for j in 0..N {
                    h[(i, j)] = h[(i, j)] + factor * s[i] * s[j]
                        - rho * (hy[i] * s[j] + s[i] * hy[j]);
                }
            }
        }

        x = x_new;
        fx = f_new;
        g = g_new;
    }

    Err(OptimError::MaxIterations)
}
