//! Dynamically-sized variants of the optimization routines.
//!
//! Mirrors the fixed-size API but uses [`DynVector`] / [`DynMatrix`] so the
//! parameter and residual dimensions are chosen at runtime. Requires the
//! `alloc` feature.

use crate::dynmatrix::{DynMatrix, DynVector};
use crate::traits::FloatScalar;

use super::line_search::backtracking_armijo_dyn;
use super::{BfgsSettings, GaussNewtonSettings, LmSettings, OptimError};

// ── Result types ────────────────────────────────────────────────────

/// Result of a dynamically-sized unconstrained minimization algorithm.
#[derive(Debug, Clone)]
pub struct MinimizeResultDyn<T: FloatScalar> {
    /// Approximate minimizer.
    pub x: DynVector<T>,
    /// Function value at the minimizer: `f(x)`.
    pub fx: T,
    /// Gradient norm at the minimizer.
    pub grad_norm: T,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Number of function evaluations.
    pub f_evals: usize,
    /// Number of gradient evaluations.
    pub grad_evals: usize,
}

/// Result of a dynamically-sized nonlinear least-squares algorithm.
#[derive(Debug, Clone)]
pub struct LeastSquaresResultDyn<T: FloatScalar> {
    /// Approximate minimizer of `0.5 * ||r(x)||^2`.
    pub x: DynVector<T>,
    /// Final cost: `0.5 * ||r(x)||^2`.
    pub cost: T,
    /// Gradient norm: `||J^T r||`.
    pub grad_norm: T,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Number of residual evaluations.
    pub r_evals: usize,
    /// Number of Jacobian evaluations.
    pub j_evals: usize,
}

// ── Vector helpers (private) ────────────────────────────────────────

#[inline]
fn vec_add<T: FloatScalar>(a: &DynVector<T>, b: &DynVector<T>) -> DynVector<T> {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut out = DynVector::zeros(n);
    for i in 0..n {
        out[i] = a[i] + b[i];
    }
    out
}

#[inline]
fn vec_sub<T: FloatScalar>(a: &DynVector<T>, b: &DynVector<T>) -> DynVector<T> {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut out = DynVector::zeros(n);
    for i in 0..n {
        out[i] = a[i] - b[i];
    }
    out
}

#[inline]
fn vec_neg<T: FloatScalar>(a: &DynVector<T>) -> DynVector<T> {
    let n = a.len();
    let mut out = DynVector::zeros(n);
    for i in 0..n {
        out[i] = T::zero() - a[i];
    }
    out
}

#[inline]
fn vec_scale<T: FloatScalar>(a: &DynVector<T>, s: T) -> DynVector<T> {
    let n = a.len();
    let mut out = DynVector::zeros(n);
    for i in 0..n {
        out[i] = a[i] * s;
    }
    out
}

/// `m * v` where `m` is M×N and `v` is N → M-vector.
#[inline]
fn mat_vec<T: FloatScalar>(m: &DynMatrix<T>, v: &DynVector<T>) -> DynVector<T> {
    let nr = m.nrows();
    let nc = m.ncols();
    debug_assert_eq!(nc, v.len());
    let mut out = DynVector::zeros(nr);
    for j in 0..nc {
        let vj = v[j];
        for i in 0..nr {
            out[i] = out[i] + m[(i, j)] * vj;
        }
    }
    out
}

/// `m^T * v` where `m` is M×N and `v` is M → N-vector.
#[inline]
fn matt_vec<T: FloatScalar>(m: &DynMatrix<T>, v: &DynVector<T>) -> DynVector<T> {
    let nr = m.nrows();
    let nc = m.ncols();
    debug_assert_eq!(nr, v.len());
    let mut out = DynVector::zeros(nc);
    for j in 0..nc {
        let mut s = T::zero();
        for i in 0..nr {
            s = s + m[(i, j)] * v[i];
        }
        out[j] = s;
    }
    out
}

// ── Finite differences ──────────────────────────────────────────────

/// Approximate the Jacobian of `f: R^n → R^m` using forward finite differences.
///
/// Uses step size `h_j = sqrt(ε) * max(|x_j|, 1)` for each component,
/// requiring `n + 1` function evaluations.
///
/// # Example
///
/// ```
/// use numeris::optim::finite_difference_jacobian_dyn;
/// use numeris::DynVector;
///
/// let x = DynVector::from_slice(&[3.0_f64, 4.0]);
/// let j = finite_difference_jacobian_dyn(
///     |x: &DynVector<f64>| DynVector::from_slice(&[x[0] * x[0], x[0] * x[1]]),
///     &x,
/// );
/// assert!((j[(0, 0)] - 6.0).abs() < 1e-5);
/// assert!((j[(1, 1)] - 3.0).abs() < 1e-5);
/// ```
pub fn finite_difference_jacobian_dyn<T: FloatScalar>(
    mut f: impl FnMut(&DynVector<T>) -> DynVector<T>,
    x: &DynVector<T>,
) -> DynMatrix<T> {
    let sqrt_eps = T::epsilon().sqrt();
    let f0 = f(x);
    let n = x.len();
    let m = f0.len();
    let mut jac = DynMatrix::zeros(m, n);

    for j in 0..n {
        let h = sqrt_eps * x[j].abs().max(T::one());
        let mut x_pert = x.clone();
        x_pert[j] = x_pert[j] + h;
        let f_pert = f(&x_pert);

        for i in 0..m {
            jac[(i, j)] = (f_pert[i] - f0[i]) / h;
        }
    }

    jac
}

/// Approximate the gradient of `f: R^n → R` using forward finite differences.
///
/// Uses step size `h_j = sqrt(ε) * max(|x_j|, 1)` for each component,
/// requiring `n + 1` function evaluations.
///
/// # Example
///
/// ```
/// use numeris::optim::finite_difference_gradient_dyn;
/// use numeris::DynVector;
///
/// let x = DynVector::from_slice(&[3.0_f64, 4.0]);
/// let g = finite_difference_gradient_dyn(
///     |x: &DynVector<f64>| x[0] * x[0] + x[1] * x[1] * 2.0,
///     &x,
/// );
/// assert!((g[0] - 6.0).abs() < 1e-5);
/// assert!((g[1] - 16.0).abs() < 1e-5);
/// ```
pub fn finite_difference_gradient_dyn<T: FloatScalar>(
    mut f: impl FnMut(&DynVector<T>) -> T,
    x: &DynVector<T>,
) -> DynVector<T> {
    let sqrt_eps = T::epsilon().sqrt();
    let f0 = f(x);
    let n = x.len();
    let mut grad = DynVector::zeros(n);

    for j in 0..n {
        let h = sqrt_eps * x[j].abs().max(T::one());
        let mut x_pert = x.clone();
        x_pert[j] = x_pert[j] + h;
        grad[j] = (f(&x_pert) - f0) / h;
    }

    grad
}

// ── BFGS ────────────────────────────────────────────────────────────

/// Minimize `f: R^n → R` using the BFGS quasi-Newton method (dynamic dimension).
///
/// Maintains an `n × n` inverse-Hessian approximation on the heap. Uses Armijo
/// backtracking line search.
///
/// # Arguments
///
/// * `f` — objective function
/// * `grad` — gradient `∇f`
/// * `x0` — initial guess
/// * `settings` — convergence tolerances and algorithm parameters
///
/// # Example
///
/// ```
/// use numeris::optim::{minimize_bfgs_dyn, BfgsSettings};
/// use numeris::DynVector;
///
/// // Minimize f(x) = (x0-1)^2 + (x1-2)^2
/// let r = minimize_bfgs_dyn(
///     |x: &DynVector<f64>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2),
///     |x: &DynVector<f64>| DynVector::from_slice(&[2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)]),
///     &DynVector::from_slice(&[0.0, 0.0]),
///     &BfgsSettings::default(),
/// ).unwrap();
/// assert!((r.x[0] - 1.0).abs() < 1e-6);
/// assert!((r.x[1] - 2.0).abs() < 1e-6);
/// ```
pub fn minimize_bfgs_dyn<T: FloatScalar>(
    mut f: impl FnMut(&DynVector<T>) -> T,
    mut grad: impl FnMut(&DynVector<T>) -> DynVector<T>,
    x0: &DynVector<T>,
    settings: &BfgsSettings<T>,
) -> Result<MinimizeResultDyn<T>, OptimError> {
    let n = x0.len();
    let mut x = x0.clone();
    let mut fx = f(&x);
    let mut g = grad(&x);
    let mut f_evals = 1usize;
    let mut grad_evals = 1usize;

    assert_eq!(g.len(), n, "gradient length must match x length");

    // Inverse Hessian approximation, initialized to identity
    let mut h = DynMatrix::<T>::zeros(n, n);
    for i in 0..n {
        h[(i, i)] = T::one();
    }

    for iter in 0..settings.max_iter {
        let g_norm = g.norm();

        if g_norm < settings.grad_tol {
            return Ok(MinimizeResultDyn {
                x,
                fx,
                grad_norm: g_norm,
                iterations: iter,
                f_evals,
                grad_evals,
            });
        }

        // Search direction: p = -H * g
        let hg = mat_vec(&h, &g);
        let p_init = vec_neg(&hg);

        let grad_dot_p = g.dot(&p_init);
        let (p, h_reset) = if grad_dot_p >= T::zero() {
            // Not a descent direction, reset H to identity
            (vec_neg(&g), true)
        } else {
            (p_init, false)
        };

        let grad_dot_p = if h_reset { -g.dot(&g) } else { grad_dot_p };

        let (alpha, f_new, ls_evals) = backtracking_armijo_dyn(
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

        let s = vec_scale(&p, alpha);
        let x_new = vec_add(&x, &s);
        let g_new = grad(&x_new);
        grad_evals += 1;

        let y = vec_sub(&g_new, &g);
        let ys = y.dot(&s);

        if s.norm() < settings.x_tol * (T::one() + x.norm()) {
            return Ok(MinimizeResultDyn {
                x: x_new,
                fx: f_new,
                grad_norm: g_new.norm(),
                iterations: iter + 1,
                f_evals,
                grad_evals,
            });
        }

        if (fx - f_new).abs() < settings.f_tol * (T::one() + fx.abs()) {
            return Ok(MinimizeResultDyn {
                x: x_new,
                fx: f_new,
                grad_norm: g_new.norm(),
                iterations: iter + 1,
                f_evals,
                grad_evals,
            });
        }

        if h_reset {
            for i in 0..n {
                for j in 0..n {
                    h[(i, j)] = if i == j { T::one() } else { T::zero() };
                }
            }
        }

        if ys > T::epsilon() {
            let rho = T::one() / ys;

            // hy = H * y
            let hy = mat_vec(&h, &y);
            let yhy = y.dot(&hy);
            let factor = (T::one() + rho * yhy) * rho;

            // H = H + factor * s s^T - rho * (hy s^T + s hy^T)
            for i in 0..n {
                for j in 0..n {
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

// ── Gauss-Newton ────────────────────────────────────────────────────

/// Solve a nonlinear least-squares problem using Gauss-Newton (dynamic dimension).
///
/// Minimizes `0.5 * ||r(x)||^2` where `r: R^n → R^m` and `J: R^n → R^{m×n}`.
/// Each iteration solves the linear least-squares subproblem via QR.
///
/// Requires `m >= n`.
///
/// # Example
///
/// ```
/// use numeris::optim::{least_squares_gn_dyn, GaussNewtonSettings};
/// use numeris::{DynMatrix, DynVector};
///
/// let a = DynMatrix::from_rows(3, 2, &[1.0_f64, 1.0, 1.0, 2.0, 1.0, 3.0]);
/// let b = DynVector::from_slice(&[1.0, 2.0, 3.0]);
/// let r = least_squares_gn_dyn(
///     |x: &DynVector<f64>| {
///         let mut out = DynVector::zeros(3);
///         for i in 0..3 {
///             out[i] = a[(i, 0)] * x[0] + a[(i, 1)] * x[1] - b[i];
///         }
///         out
///     },
///     |_: &DynVector<f64>| a.clone(),
///     &DynVector::from_slice(&[0.0, 0.0]),
///     &GaussNewtonSettings::default(),
/// ).unwrap();
/// assert!(r.iterations <= 2);
/// ```
pub fn least_squares_gn_dyn<T: FloatScalar>(
    mut residual: impl FnMut(&DynVector<T>) -> DynVector<T>,
    mut jacobian: impl FnMut(&DynVector<T>) -> DynMatrix<T>,
    x0: &DynVector<T>,
    settings: &GaussNewtonSettings<T>,
) -> Result<LeastSquaresResultDyn<T>, OptimError> {
    let n = x0.len();
    let mut x = x0.clone();
    let mut r = residual(&x);
    let mut r_evals = 1usize;
    let mut j_evals = 0usize;

    let half = T::one() / (T::one() + T::one());
    let mut cost = r.dot(&r) * half;

    for iter in 0..settings.max_iter {
        let j = jacobian(&x);
        j_evals += 1;
        assert_eq!(j.nrows(), r.len(), "jacobian rows must match residual length");
        assert_eq!(j.ncols(), n, "jacobian cols must match x length");

        let g = matt_vec(&j, &r);
        let g_norm = g.norm();

        if g_norm < settings.grad_tol {
            return Ok(LeastSquaresResultDyn {
                x,
                cost,
                grad_norm: g_norm,
                iterations: iter,
                r_evals,
                j_evals,
            });
        }

        // Solve J * delta = -r via QR
        let neg_r = vec_neg(&r);
        let qr = j.qr().map_err(|_| OptimError::Singular)?;
        let delta = qr.solve(&neg_r);

        let x_new = vec_add(&x, &delta);
        r = residual(&x_new);
        r_evals += 1;
        let cost_new = r.dot(&r) * half;

        if delta.norm() < settings.x_tol * (T::one() + x.norm()) {
            return Ok(LeastSquaresResultDyn {
                x: x_new,
                cost: cost_new,
                grad_norm: g_norm,
                iterations: iter + 1,
                r_evals,
                j_evals,
            });
        }

        if (cost - cost_new).abs() < settings.f_tol * (T::one() + cost.abs()) {
            return Ok(LeastSquaresResultDyn {
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

// ── Levenberg-Marquardt ─────────────────────────────────────────────

/// Solve a nonlinear least-squares problem using Levenberg-Marquardt (dynamic dimension).
///
/// Minimizes `0.5 * ||r(x)||^2` using damped normal equations:
/// `(J^T J + μ I) δ = -J^T r`.
///
/// # Example
///
/// ```
/// use numeris::optim::{least_squares_lm_dyn, LmSettings};
/// use numeris::{DynMatrix, DynVector};
///
/// let t = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
/// let y = [2.0, 2.7, 3.65, 4.95, 6.7];
///
/// let r = least_squares_lm_dyn(
///     |x: &DynVector<f64>| {
///         let mut res = DynVector::zeros(5);
///         for i in 0..5 {
///             res[i] = x[0] * (x[1] * t[i]).exp() - y[i];
///         }
///         res
///     },
///     |x: &DynVector<f64>| {
///         let mut j = DynMatrix::zeros(5, 2);
///         for i in 0..5 {
///             let e = (x[1] * t[i]).exp();
///             j[(i, 0)] = e;
///             j[(i, 1)] = x[0] * t[i] * e;
///         }
///         j
///     },
///     &DynVector::from_slice(&[1.0, 0.1]),
///     &LmSettings::default(),
/// ).unwrap();
/// assert!(r.cost < 0.1);
/// ```
pub fn least_squares_lm_dyn<T: FloatScalar>(
    mut residual: impl FnMut(&DynVector<T>) -> DynVector<T>,
    mut jacobian: impl FnMut(&DynVector<T>) -> DynMatrix<T>,
    x0: &DynVector<T>,
    settings: &LmSettings<T>,
) -> Result<LeastSquaresResultDyn<T>, OptimError> {
    let n = x0.len();
    let mut x = x0.clone();
    let mut r = residual(&x);
    let mut r_evals = 1usize;
    let mut j_evals = 0usize;

    let half = T::one() / (T::one() + T::one());
    let mut cost = r.dot(&r) * half;
    let mut mu = settings.mu_init;

    for iter in 0..settings.max_iter {
        let j = jacobian(&x);
        j_evals += 1;
        assert_eq!(j.nrows(), r.len(), "jacobian rows must match residual length");
        assert_eq!(j.ncols(), n, "jacobian cols must match x length");

        // J^T J (n×n)
        let jt = j.transpose();
        let jtj = &jt * &j;
        // g = J^T r
        let g = matt_vec(&j, &r);
        let g_norm = g.norm();

        if g_norm < settings.grad_tol {
            return Ok(LeastSquaresResultDyn {
                x,
                cost,
                grad_norm: g_norm,
                iterations: iter,
                r_evals,
                j_evals,
            });
        }

        let neg_g = vec_neg(&g);
        let delta = solve_damped_dyn(&jtj, mu, &neg_g)?;

        let x_new = vec_add(&x, &delta);
        let r_new = residual(&x_new);
        r_evals += 1;
        let cost_new = r_new.dot(&r_new) * half;

        // Predicted reduction: delta^T (mu * delta - g)
        let mu_delta_minus_g = vec_sub(&vec_scale(&delta, mu), &g);
        let predicted = delta.dot(&mu_delta_minus_g);
        let eps = T::epsilon();

        if predicted > T::zero() && predicted.abs() >= eps * cost.abs() {
            let actual = cost - cost_new;
            let gain_ratio = actual / predicted;

            if gain_ratio > T::zero() {
                x = x_new;
                r = r_new;
                cost = cost_new;
                mu = (mu * settings.mu_decrease).max(settings.mu_min);

                if delta.norm() < settings.x_tol * (T::one() + x.norm()) {
                    return Ok(LeastSquaresResultDyn {
                        x,
                        cost,
                        grad_norm: g_norm,
                        iterations: iter + 1,
                        r_evals,
                        j_evals,
                    });
                }

                if actual.abs() < settings.f_tol * (T::one() + cost.abs()) {
                    return Ok(LeastSquaresResultDyn {
                        x,
                        cost,
                        grad_norm: g_norm,
                        iterations: iter + 1,
                        r_evals,
                        j_evals,
                    });
                }
            } else {
                mu = (mu * settings.mu_increase).min(settings.mu_max);
            }
        } else {
            mu = (mu * settings.mu_increase).min(settings.mu_max);
        }
    }

    Err(OptimError::MaxIterations)
}

/// Solve `(A + mu * I) * x = b` via Cholesky, falling back to LU.
fn solve_damped_dyn<T: FloatScalar>(
    a: &DynMatrix<T>,
    mu: T,
    b: &DynVector<T>,
) -> Result<DynVector<T>, OptimError> {
    let n = a.nrows();
    let mut damped = a.clone();
    for i in 0..n {
        damped[(i, i)] = damped[(i, i)] + mu;
    }

    if let Ok(chol) = damped.cholesky() {
        return Ok(chol.solve(b));
    }

    let lu = damped.lu().map_err(|_| OptimError::Singular)?;
    Ok(lu.solve(b))
}

