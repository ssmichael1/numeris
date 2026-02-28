// Reference implementations:
//   E. Hairer & G. Wanner, "Solving Ordinary Differential Equations II" (1996), §IV.7
//   E. Hairer, rodas.f — http://www.unige.ch/~hairer/prog/stiff/rodas.f
//   OrdinaryDiffEq.jl (SciML/Julia) — Rosenbrock integrators

use crate::linalg::lu::{lu_in_place, lu_solve};
use crate::traits::FloatScalar;
use crate::matrix::vector::Vector;
use crate::Matrix;
use super::{AdaptiveSettings, OdeError, Solution};

/// Forward-difference Jacobian approximation.
///
/// Computes `J[i][j] = ∂f_i/∂y_j` using forward differences.
/// Reuses `fy = f(t, y)` already evaluated by the caller.
fn fd_jacobian<T: FloatScalar, const S: usize>(
    f: &mut impl FnMut(T, &Vector<T, S>) -> Vector<T, S>,
    t: T,
    y: &Vector<T, S>,
    fy: &Vector<T, S>,
) -> Matrix<T, S, S> {
    let eps_sqrt = T::epsilon().sqrt();
    let one = T::one();
    let mut jac = Matrix::<T, S, S>::zeros();

    for j in 0..S {
        let yj_abs = y[j].abs();
        let hj = eps_sqrt * if yj_abs > one { yj_abs } else { one };
        let mut y_pert = *y;
        y_pert[j] = y_pert[j] + hj;
        let fy_pert = f(t, &y_pert);
        let inv_hj = one / hj;
        for i in 0..S {
            jac[(i, j)] = (fy_pert[i] - fy[i]) * inv_hj;
        }
    }

    jac
}

/// Trait for Rosenbrock (linearly-implicit Runge-Kutta) solvers.
///
/// Each solver is a zero-size struct that implements this trait with
/// const coefficient arrays. `STAGES` is the number of stages.
///
/// Unlike explicit RK methods ([`super::RKAdaptive`]), Rosenbrock methods
/// solve linear systems involving the Jacobian at each stage, making them
/// suitable for stiff ODEs without requiring nonlinear Newton iterations.
///
/// Each step solves:
///
/// ```text
/// (I/(hγ) − J) k_i = f(t + α_i h, y + Σ a_ij k_j) + Σ (c_ij/h) k_j
/// ```
///
/// where `J = ∂f/∂y` is factored once per step via LU decomposition.
/// The solution is `y_{n+1} = y_n + Σ m_i k_i` and the error estimate
/// is `err = Σ (m_i − m̂_i) k_i`.
pub trait Rosenbrock<const STAGES: usize> {
    /// Stage coupling matrix (lower-triangular, zero diagonal).
    const A: [[f64; STAGES]; STAGES];
    /// Off-diagonal Γ coupling (lower-triangular, zero diagonal).
    const C: [[f64; STAGES]; STAGES];
    /// Shared diagonal element of the Γ matrix.
    const GAMMA_DIAG: f64;
    /// Time offsets for each stage.
    const ALPHA: [f64; STAGES];
    /// Row sums of the full Γ matrix (diagonal + off-diagonal).
    const GAMMA_SUM: [f64; STAGES];
    /// Solution weights (higher-order).
    const M: [f64; STAGES];
    /// Embedded solution weights (for error estimation).
    const MHAT: [f64; STAGES];
    /// Order of the higher-order method.
    const ORDER: usize;

    /// Integrate from `t0` to `tf` with user-supplied Jacobian.
    ///
    /// `jac(t, y)` must return the Jacobian matrix `∂f/∂y` at `(t, y)`.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::ode::{Rosenbrock, RODAS4, AdaptiveSettings};
    /// use numeris::{Vector, Matrix};
    ///
    /// // Stiff decay: y' = -1000*y, y(0) = 1, exact: e^{-1000t}
    /// let y0 = Vector::from_array([1.0_f64]);
    /// let settings = AdaptiveSettings::default();
    /// let sol = RODAS4::integrate(
    ///     0.0, 0.01, &y0,
    ///     |_t, y| Vector::from_array([-1000.0 * y[0]]),
    ///     |_t, _y| Matrix::new([[-1000.0]]),
    ///     &settings,
    /// ).unwrap();
    /// let exact = (-1000.0_f64 * 0.01).exp();
    /// assert!((sol.y[0] - exact).abs() < 1e-6);
    /// ```
    fn integrate<T: FloatScalar, const S: usize>(
        t0: T,
        tf: T,
        y0: &Vector<T, S>,
        f: impl FnMut(T, &Vector<T, S>) -> Vector<T, S>,
        jac: impl FnMut(T, &Vector<T, S>) -> Matrix<T, S, S>,
        settings: &AdaptiveSettings<T>,
    ) -> Result<Solution<T, S>, OdeError> {
        rosenbrock_step_loop::<Self, T, S, STAGES>(t0, tf, y0, f, JacSource::User(jac), settings)
    }

    /// Integrate from `t0` to `tf` with automatic finite-difference Jacobian.
    ///
    /// The Jacobian is computed internally via forward differences.
    /// Costs an extra `S` function evaluations per step.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::ode::{Rosenbrock, RODAS4, AdaptiveSettings};
    /// use numeris::Vector;
    ///
    /// // Stiff decay: y' = -1000*y
    /// let y0 = Vector::from_array([1.0_f64]);
    /// let settings = AdaptiveSettings::default();
    /// let sol = RODAS4::integrate_auto(
    ///     0.0, 0.01, &y0,
    ///     |_t, y| Vector::from_array([-1000.0 * y[0]]),
    ///     &settings,
    /// ).unwrap();
    /// let exact = (-1000.0_f64 * 0.01).exp();
    /// assert!((sol.y[0] - exact).abs() < 1e-6);
    /// ```
    fn integrate_auto<T: FloatScalar, const S: usize>(
        t0: T,
        tf: T,
        y0: &Vector<T, S>,
        f: impl FnMut(T, &Vector<T, S>) -> Vector<T, S>,
        settings: &AdaptiveSettings<T>,
    ) -> Result<Solution<T, S>, OdeError> {
        rosenbrock_step_loop::<Self, T, S, STAGES>(
            t0, tf, y0, f,
            JacSource::<T, S, fn(T, &Vector<T, S>) -> Matrix<T, S, S>>::Auto,
            settings,
        )
    }
}

/// Internal enum to distinguish user-supplied vs auto Jacobian.
enum JacSource<T, const S: usize, J> {
    User(J),
    Auto,
    #[allow(dead_code)]
    _Phantom(core::marker::PhantomData<T>),
}

/// Core integration loop shared by `integrate` and `integrate_auto`.
fn rosenbrock_step_loop<R, T, const S: usize, const STAGES: usize>(
    t0: T,
    tf: T,
    y0: &Vector<T, S>,
    mut f: impl FnMut(T, &Vector<T, S>) -> Vector<T, S>,
    mut jac_source: JacSource<T, S, impl FnMut(T, &Vector<T, S>) -> Matrix<T, S, S>>,
    settings: &AdaptiveSettings<T>,
) -> Result<Solution<T, S>, OdeError>
where
    R: Rosenbrock<STAGES> + ?Sized,
    T: FloatScalar,
{
    let one = T::one();
    let zero = T::zero();
    let tdir = if tf > t0 { one } else { -one };

    let mut t = t0;
    let mut y = *y0;
    let mut nevals: usize = 0;
    let mut naccept: usize = 0;
    let mut nreject: usize = 0;

    // PID controller state
    let mut enorm_prev = T::from(1.0e-4).unwrap();
    let mut enorm_prev2 = T::from(1.0e-4).unwrap();

    // Initial step-size guess (same heuristic as adaptive.rs)
    let mut h = {
        let sci = y0.abs() * settings.rel_tol + settings.abs_tol;
        let d0 = y0.element_div(&sci).scaled_norm();
        let ydot0 = f(t0, y0);
        let d1 = ydot0.element_div(&sci).scaled_norm();
        let h0 = T::from(0.01).unwrap() * d0 / d1 * tdir;
        let y1 = *y0 + ydot0 * h0;
        let ydot1 = f(t0 + h0, &y1);
        let d2 = (ydot1 - ydot0).element_div(&sci).scaled_norm() / h0;
        nevals += 2;

        let dmax = if d1 > d2 { d1 } else { d2 };
        let order_t = T::from(R::ORDER).unwrap();
        let h1 = if dmax < T::from(1e-15).unwrap() {
            let h0_abs = h0.abs();
            let floor = T::from(1e-6).unwrap();
            if h0_abs * T::from(1e-3).unwrap() > floor {
                h0_abs * T::from(1e-3).unwrap()
            } else {
                floor
            }
        } else {
            T::from(10.0)
                .unwrap()
                .powf(-(T::from(2.0).unwrap() + dmax.log10()) / order_t)
        };

        let h0_100 = T::from(100.0).unwrap() * h0.abs();
        let h1_abs = h1.abs();
        (if h0_100 < h1_abs { h0_100 } else { h1_abs }) * tdir
    };

    // Reusable LU storage
    let mut w_mat = Matrix::<T, S, S>::zeros();
    let mut perm = [0usize; S];

    #[cfg(feature = "std")]
    let mut dense_store = if settings.dense_output {
        Some(super::DenseOutput {
            t: Vec::new(),
            h: Vec::new(),
            stages: Vec::new(),
            y: Vec::new(),
        })
    } else {
        None
    };

    loop {
        // Clamp step to not overshoot end
        if (tdir > zero && (t + h) >= tf) || (tdir < zero && (t + h) <= tf) {
            h = tf - t;
        }

        let gamma = T::from(R::GAMMA_DIAG).unwrap();
        let inv_hgamma = one / (h * gamma);

        // Evaluate f at current point
        let fy = f(t, &y);
        nevals += 1;

        // Get Jacobian
        let jac_mat = match &mut jac_source {
            JacSource::User(jac_fn) => jac_fn(t, &y),
            JacSource::Auto => {
                let j = fd_jacobian(&mut f, t, &y, &fy);
                nevals += S;
                j
            }
            JacSource::_Phantom(_) => unreachable!(),
        };

        // Form W = I/(hγ) − J and LU-factorize
        for i in 0..S {
            for j in 0..S {
                w_mat[(i, j)] = -jac_mat[(i, j)];
            }
            w_mat[(i, i)] = w_mat[(i, i)] + inv_hgamma;
        }

        if lu_in_place(&mut w_mat, &mut perm).is_err() {
            return Err(OdeError::SingularJacobian);
        }

        // Compute stages k_1, ..., k_STAGES
        //
        // Stage formula (Hairer & Wanner, rodas.f):
        //   (I/(hγ) − J) k_i = f(t + α_i h, y + Σ a_ij k_j) + Σ (c_ij/h) k_j
        //
        // The C coefficients couple stages directly (no J multiplication).
        // Solution: y_{n+1} = y_n + Σ m_i k_i
        let mut karr = [Vector::<T, S>::zeros(); STAGES];

        for i in 0..STAGES {
            // Stage argument: y + Σ a_ij k_j
            let mut y_stage = y;
            for jj in 0..i {
                let a_ij = T::from(R::A[i][jj]).unwrap();
                if a_ij != zero {
                    y_stage = y_stage + karr[jj] * a_ij;
                }
            }

            // f at the stage point (reuse fy for stage 0)
            let ti = t + T::from(R::ALPHA[i]).unwrap() * h;
            let fi = if i == 0 {
                fy
            } else {
                nevals += 1;
                f(ti, &y_stage)
            };

            // RHS = fi + Σ (c_ij/h) k_j
            let mut rhs = fi;
            if i > 0 {
                let inv_h = one / h;
                for jj in 0..i {
                    let c_ij = T::from(R::C[i][jj]).unwrap();
                    if c_ij != zero {
                        rhs = rhs + karr[jj] * (c_ij * inv_h);
                    }
                }
            }

            // Solve W k_i = rhs
            let mut rhs_flat = [zero; S];
            for idx in 0..S {
                rhs_flat[idx] = rhs[idx];
            }
            let mut ki_flat = [zero; S];
            lu_solve(&w_mat, &perm, &rhs_flat, &mut ki_flat);
            karr[i] = Vector::from_array(ki_flat);
        }

        // Solution: y_{n+1} = y_n + Σ m_i k_i
        let mut ynp1 = y;
        for (idx, ki) in karr.iter().enumerate() {
            let m_idx = T::from(R::M[idx]).unwrap();
            if m_idx != zero {
                ynp1 = ynp1 + *ki * m_idx;
            }
        }

        // Error estimate: err = Σ (m_i − m̂_i) k_i
        let mut yerr = Vector::<T, S>::zeros();
        for (idx, ki) in karr.iter().enumerate() {
            let diff = R::M[idx] - R::MHAT[idx];
            if diff.abs() > 1.0e-20 {
                yerr = yerr + *ki * T::from(diff).unwrap();
            }
        }

        // Normalized error
        let enorm = {
            let ymax = y.abs().element_max(&ynp1.abs()) * settings.rel_tol + settings.abs_tol;
            yerr.element_div(&ymax).scaled_norm()
        };

        if !enorm.is_finite() {
            return Err(OdeError::StepNotFinite);
        }

        // PID step-size controller (Söderlind & Wang 2006)
        let order_f = T::from(R::ORDER).unwrap();
        let beta1 = T::from(0.7).unwrap() / order_f;
        let beta2 = T::from(0.4).unwrap() / order_f;
        let beta3 = T::from(0.1).unwrap() / order_f;
        let q = {
            let raw = enorm.powf(beta1)
                / enorm_prev.powf(beta2)
                * enorm_prev2.powf(beta3)
                / settings.safety;
            let lo = one / settings.max_factor;
            let hi = one / settings.min_factor;
            if raw < lo { lo } else if raw > hi { hi } else { raw }
        };

        if enorm < one || h.abs() <= settings.min_step {
            // Accept step
            #[cfg(feature = "std")]
            if let Some(ref mut ds) = dense_store {
                ds.t.push(t);
                ds.h.push(h);
                ds.stages.push(karr.to_vec());
                ds.y.push(y);
            }

            let floor = T::from(1.0e-4).unwrap();
            enorm_prev2 = enorm_prev;
            enorm_prev = if enorm > floor { enorm } else { floor };
            t = t + h;
            y = ynp1;
            h = h / q;

            naccept += 1;
            if (tdir > zero && t >= tf) || (tdir < zero && t <= tf) {
                break;
            }
        } else {
            // Reject step
            nreject += 1;
            let hi = one / settings.min_factor;
            let reject_q = enorm.powf(beta1) / settings.safety;
            let denom = if reject_q < hi { reject_q } else { hi };
            h = h / denom;
        }

        if naccept + nreject >= settings.max_steps {
            return Err(OdeError::MaxStepsExceeded);
        }
    }

    Ok(Solution {
        t,
        y,
        evals: nevals,
        accepted: naccept,
        rejected: nreject,
        #[cfg(feature = "std")]
        dense: dense_store,
    })
}
