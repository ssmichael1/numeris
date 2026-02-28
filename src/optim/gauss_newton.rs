use crate::matrix::vector::Vector;
use crate::traits::FloatScalar;
use crate::Matrix;

use super::{LeastSquaresResult, OptimError};

/// Settings for Gauss-Newton least-squares optimization.
#[derive(Debug, Clone, Copy)]
pub struct GaussNewtonSettings<T> {
    /// Convergence tolerance on gradient norm `||J^T r||`.
    pub grad_tol: T,
    /// Convergence tolerance on relative step size.
    pub x_tol: T,
    /// Convergence tolerance on relative cost change.
    pub f_tol: T,
    /// Maximum number of iterations.
    pub max_iter: usize,
}

impl Default for GaussNewtonSettings<f64> {
    fn default() -> Self {
        Self {
            grad_tol: 1e-8,
            x_tol: 1e-12,
            f_tol: 1e-12,
            max_iter: 100,
        }
    }
}

impl Default for GaussNewtonSettings<f32> {
    fn default() -> Self {
        Self {
            grad_tol: 1e-4,
            x_tol: 1e-6,
            f_tol: 1e-6,
            max_iter: 100,
        }
    }
}

/// Solve a nonlinear least-squares problem using Gauss-Newton.
///
/// Minimizes `0.5 * ||r(x)||^2` where `r: R^N → R^M` is the residual function
/// and `J: R^N → R^{M×N}` is its Jacobian. Requires `M >= N`.
///
/// Each iteration solves the linear least-squares subproblem via QR decomposition
/// of the Jacobian, avoiding the numerically inferior normal equations `J^T J`.
///
/// # Arguments
///
/// * `residual` — residual function `r(x)` returning an M-vector
/// * `jacobian` — Jacobian `J(x)` returning an M×N matrix
/// * `x0` — initial guess (N-vector)
/// * `settings` — convergence tolerances and iteration limit
///
/// # Errors
///
/// Returns [`OptimError::Singular`] if the Jacobian is rank-deficient.
/// Returns [`OptimError::MaxIterations`] if convergence is not achieved.
///
/// # Example
///
/// ```
/// use numeris::optim::{least_squares_gn, GaussNewtonSettings};
/// use numeris::{Matrix, Vector};
///
/// // Linear least squares: A*x = b, residual r(x) = A*x - b
/// let a = Matrix::new([[1.0_f64, 1.0], [1.0, 2.0], [1.0, 3.0]]);
/// let b = Vector::from_array([1.0, 2.0, 3.0]);
/// let r = least_squares_gn(
///     |x: &Vector<f64, 2>| a.vecmul(x) - b,
///     |_: &Vector<f64, 2>| a,
///     &Vector::from_array([0.0, 0.0]),
///     &GaussNewtonSettings::default(),
/// ).unwrap();
/// // For a linear problem, GN converges in 1 step
/// assert!(r.iterations <= 2);
/// ```
pub fn least_squares_gn<T: FloatScalar, const M: usize, const N: usize>(
    mut residual: impl FnMut(&Vector<T, N>) -> Vector<T, M>,
    mut jacobian: impl FnMut(&Vector<T, N>) -> Matrix<T, M, N>,
    x0: &Vector<T, N>,
    settings: &GaussNewtonSettings<T>,
) -> Result<LeastSquaresResult<T, N>, OptimError> {
    let mut x = *x0;
    let mut r = residual(&x);
    let mut r_evals = 1usize;
    let mut j_evals = 0usize;

    let half = T::one() / (T::one() + T::one());
    let mut cost = r.dot(&r) * half;

    for iter in 0..settings.max_iter {
        let j = jacobian(&x);
        j_evals += 1;

        // Gradient: g = J^T * r
        let g = j.transpose().vecmul(&r);
        let g_norm = g.norm();

        if g_norm < settings.grad_tol {
            return Ok(LeastSquaresResult {
                x,
                cost,
                grad_norm: g_norm,
                iterations: iter,
                r_evals,
                j_evals,
            });
        }

        // Solve J * delta = -r via QR
        let neg_r = -r;
        let qr = j.qr().map_err(|_| OptimError::Singular)?;
        let delta = qr.solve(&neg_r);

        let x_new = x + delta;
        r = residual(&x_new);
        r_evals += 1;
        let cost_new = r.dot(&r) * half;

        // Relative step size convergence
        if delta.norm() < settings.x_tol * (T::one() + x.norm()) {
            return Ok(LeastSquaresResult {
                x: x_new,
                cost: cost_new,
                grad_norm: g_norm,
                iterations: iter + 1,
                r_evals,
                j_evals,
            });
        }

        // Relative cost change convergence
        if (cost - cost_new).abs() < settings.f_tol * (T::one() + cost.abs()) {
            return Ok(LeastSquaresResult {
                x: x_new,
                cost: cost_new,
                grad_norm: g_norm,
                iterations: iter + 1,
                r_evals,
                j_evals,
            });
        }

        x = x_new;
        cost = cost_new;
    }

    Err(OptimError::MaxIterations)
}
