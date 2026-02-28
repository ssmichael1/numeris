use crate::linalg::cholesky::cholesky_in_place;
use crate::linalg::lu::{lu_in_place, lu_solve};
use crate::matrix::vector::Vector;
use crate::traits::FloatScalar;
use crate::Matrix;

use super::{LeastSquaresResult, OptimError};

/// Settings for Levenberg-Marquardt least-squares optimization.
#[derive(Debug, Clone, Copy)]
pub struct LmSettings<T> {
    /// Convergence tolerance on gradient norm `||J^T r||`.
    pub grad_tol: T,
    /// Convergence tolerance on relative step size.
    pub x_tol: T,
    /// Convergence tolerance on relative cost change.
    pub f_tol: T,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Initial damping parameter.
    pub mu_init: T,
    /// Factor to increase damping on rejected step.
    pub mu_increase: T,
    /// Factor to decrease damping on accepted step.
    pub mu_decrease: T,
}

impl Default for LmSettings<f64> {
    fn default() -> Self {
        Self {
            grad_tol: 1e-8,
            x_tol: 1e-12,
            f_tol: 1e-12,
            max_iter: 100,
            mu_init: 1e-3,
            mu_increase: 10.0,
            mu_decrease: 0.1,
        }
    }
}

impl Default for LmSettings<f32> {
    fn default() -> Self {
        Self {
            grad_tol: 1e-4,
            x_tol: 1e-6,
            f_tol: 1e-6,
            max_iter: 100,
            mu_init: 1e-3,
            mu_increase: 10.0,
            mu_decrease: 0.1,
        }
    }
}

/// Solve a nonlinear least-squares problem using Levenberg-Marquardt.
///
/// Minimizes `0.5 * ||r(x)||^2` using damped normal equations:
/// `(J^T J + μ I) δ = -J^T r`
///
/// Interpolates between Gauss-Newton (small μ) and gradient descent (large μ)
/// based on a trust-region gain ratio.
///
/// # Arguments
///
/// * `residual` — residual function `r(x)` returning an M-vector
/// * `jacobian` — Jacobian `J(x)` returning an M×N matrix
/// * `x0` — initial guess (N-vector)
/// * `settings` — convergence tolerances and algorithm parameters
///
/// # Errors
///
/// Returns [`OptimError::Singular`] if the damped system is singular (should not happen for μ > 0).
/// Returns [`OptimError::MaxIterations`] if convergence is not achieved.
///
/// # Example
///
/// ```
/// use numeris::optim::{least_squares_lm, LmSettings};
/// use numeris::{Matrix, Vector};
///
/// // Fit y = a * exp(b * x) to data points
/// let t = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
/// let y = [2.0, 2.7, 3.65, 4.95, 6.7];
///
/// let r = least_squares_lm(
///     |x: &Vector<f64, 2>| {
///         let mut res = Vector::<f64, 5>::zeros();
///         for i in 0..5 {
///             res[i] = x[0] * (x[1] * t[i]).exp() - y[i];
///         }
///         res
///     },
///     |x: &Vector<f64, 2>| {
///         let mut j = Matrix::<f64, 5, 2>::zeros();
///         for i in 0..5 {
///             let e = (x[1] * t[i]).exp();
///             j[(i, 0)] = e;
///             j[(i, 1)] = x[0] * t[i] * e;
///         }
///         j
///     },
///     &Vector::from_array([1.0, 0.1]),
///     &LmSettings::default(),
/// ).unwrap();
/// assert!(r.cost < 0.1);
/// ```
pub fn least_squares_lm<T: FloatScalar, const M: usize, const N: usize>(
    mut residual: impl FnMut(&Vector<T, N>) -> Vector<T, M>,
    mut jacobian: impl FnMut(&Vector<T, N>) -> Matrix<T, M, N>,
    x0: &Vector<T, N>,
    settings: &LmSettings<T>,
) -> Result<LeastSquaresResult<T, N>, OptimError> {
    let mut x = *x0;
    let mut r = residual(&x);
    let mut r_evals = 1usize;
    let mut j_evals = 0usize;

    let half = T::one() / (T::one() + T::one());
    let mut cost = r.dot(&r) * half;
    let mut mu = settings.mu_init;

    for iter in 0..settings.max_iter {
        let j = jacobian(&x);
        j_evals += 1;

        // J^T J
        let jtj = j.transpose() * j;
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

        // Solve (J^T J + mu * I) * delta = -g
        let neg_g = -g;
        let delta = solve_damped(&jtj, mu, &neg_g)?;

        // Trust-region gain ratio
        let x_new = x + delta;
        let r_new = residual(&x_new);
        r_evals += 1;
        let cost_new = r_new.dot(&r_new) * half;

        // Predicted reduction: delta^T (mu * delta - g)
        let predicted = delta.dot(&(delta * mu - g));

        if predicted > T::zero() {
            let actual = cost - cost_new;
            let gain_ratio = actual / predicted;

            if gain_ratio > T::zero() {
                // Accept step
                x = x_new;
                r = r_new;
                cost = cost_new;
                mu = mu * settings.mu_decrease;

                // Relative step size convergence
                if delta.norm() < settings.x_tol * (T::one() + x.norm()) {
                    return Ok(LeastSquaresResult {
                        x,
                        cost,
                        grad_norm: g_norm,
                        iterations: iter + 1,
                        r_evals,
                        j_evals,
                    });
                }

                // Relative cost change convergence
                if actual.abs() < settings.f_tol * (T::one() + cost.abs()) {
                    return Ok(LeastSquaresResult {
                        x,
                        cost,
                        grad_norm: g_norm,
                        iterations: iter + 1,
                        r_evals,
                        j_evals,
                    });
                }
            } else {
                // Reject step, increase damping
                mu = mu * settings.mu_increase;
            }
        } else {
            // Reject step, increase damping
            mu = mu * settings.mu_increase;
        }
    }

    Err(OptimError::MaxIterations)
}

/// Solve `(A + mu * I) * x = b` via Cholesky, falling back to LU.
fn solve_damped<T: FloatScalar, const N: usize>(
    a: &Matrix<T, N, N>,
    mu: T,
    b: &Vector<T, N>,
) -> Result<Vector<T, N>, OptimError> {
    // Form A + mu * I
    let mut damped = *a;
    for i in 0..N {
        damped[(i, i)] = damped[(i, i)] + mu;
    }

    // Try Cholesky first (SPD for mu > 0 and A = J^T J)
    let mut chol = damped;
    if cholesky_in_place(&mut chol).is_ok() {
        // Forward substitution: L y = b
        let mut y = [T::zero(); N];
        let mut b_flat = [T::zero(); N];
        for i in 0..N {
            b_flat[i] = b[i];
        }
        crate::linalg::cholesky::forward_substitute(&chol, &b_flat, &mut y);

        // Back substitution: L^T x = y
        let mut x_flat = [T::zero(); N];
        crate::linalg::cholesky::back_substitute_lt(&chol, &y, &mut x_flat);

        return Ok(Vector::from_array(x_flat));
    }

    // Fall back to LU
    let mut lu = damped;
    let mut perm = [0usize; N];
    lu_in_place(&mut lu, &mut perm).map_err(|_| OptimError::Singular)?;

    let mut b_flat = [T::zero(); N];
    for i in 0..N {
        b_flat[i] = b[i];
    }
    let mut x_flat = [T::zero(); N];
    lu_solve(&lu, &perm, &b_flat, &mut x_flat);

    Ok(Vector::from_array(x_flat))
}
