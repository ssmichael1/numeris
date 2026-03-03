use crate::matrix::vector::ColumnVector;
use crate::traits::FloatScalar;
use crate::Matrix;

use super::{apply_var_floor, cholesky_with_jitter, fd_jacobian, EstimateError};

/// Extended Kalman Filter with const-generic state and measurement dimensions.
///
/// `N` is the state dimension, `M` is the measurement dimension.
/// The EKF linearizes nonlinear dynamics and measurement models via
/// user-supplied or finite-difference Jacobians.
///
/// All operations are stack-allocated — no heap, fully no-std compatible.
///
/// # Robustness features
///
/// - **Joseph form** covariance update: `P = (I-KH)P(I-KH)ᵀ + KRKᵀ`
/// - **Cholesky with jitter**: retries with small diagonal ε·I when S is near-singular
/// - **Covariance floor**: `set_min_variance` clamps P diagonal entries to a minimum
/// - **Fading memory**: `set_fading_memory` scales predicted covariance by γ≥1
/// - **Innovation gating**: `update_gated` / `update_fd_gated` reject outlier measurements
/// - **Iterated EKF**: `update_iterated` / `update_fd_iterated` re-linearize at each step
///
/// # Example
///
/// ```
/// use numeris::estimate::Ekf;
/// use numeris::{ColumnVector, Matrix};
///
/// // Constant-velocity model: state = [position, velocity]
/// let x0 = ColumnVector::from_column([0.0_f64, 1.0]);
/// let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
/// let mut ekf = Ekf::<f64, 2, 1>::new(x0, p0);
///
/// let dt = 0.1;
/// let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
/// let r = Matrix::new([[0.5]]);
///
/// // Predict (with process noise)
/// ekf.predict(
///     |x| ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]]),
///     |_x| Matrix::new([[1.0, dt], [0.0, 1.0]]),
///     Some(&q),
/// );
///
/// // Update with position measurement
/// ekf.update(
///     &ColumnVector::from_column([0.12]),
///     |x| ColumnVector::from_column([x[(0, 0)]]),
///     |_x| Matrix::new([[1.0, 0.0]]),
///     &r,
/// ).unwrap();
/// ```
pub struct Ekf<T: FloatScalar, const N: usize, const M: usize> {
    /// State estimate.
    pub x: ColumnVector<T, N>,
    /// State covariance.
    pub p: Matrix<T, N, N>,
    /// Minimum allowed diagonal variance (0 = disabled).
    min_variance: T,
    /// Fading-memory factor γ≥1 applied to predicted covariance (1 = standard).
    gamma: T,
}

impl<T: FloatScalar, const N: usize, const M: usize> Ekf<T, N, M> {
    /// Create a new EKF with initial state `x0` and covariance `p0`.
    pub fn new(x0: ColumnVector<T, N>, p0: Matrix<T, N, N>) -> Self {
        Self {
            x: x0,
            p: p0,
            min_variance: T::zero(),
            gamma: T::one(),
        }
    }

    /// Set a minimum diagonal variance floor applied after every predict/update.
    ///
    /// Prevents covariance from degenerating to zero or going negative
    /// from accumulated numerical subtractions. Pass `0` to disable (default).
    pub fn with_min_variance(mut self, min_variance: T) -> Self {
        self.min_variance = min_variance;
        self
    }

    /// Set a fading-memory factor `γ ≥ 1` applied to the propagated covariance.
    ///
    /// The predicted covariance becomes `γ · F P Fᵀ + Q`. Values `γ > 1`
    /// inflate uncertainty after prediction to compensate for unmodeled dynamics.
    /// Default is `1.0` (standard filter).
    pub fn with_fading_memory(mut self, gamma: T) -> Self {
        self.gamma = gamma;
        self
    }

    /// Reference to the current state estimate.
    #[inline]
    pub fn state(&self) -> &ColumnVector<T, N> {
        &self.x
    }

    /// Reference to the current state covariance.
    #[inline]
    pub fn covariance(&self) -> &Matrix<T, N, N> {
        &self.p
    }

    /// Predict step with explicit dynamics Jacobian.
    ///
    /// - `f` — state transition: `x_{k+1} = f(x_k)`
    /// - `fj` — Jacobian of `f` evaluated at `x_k`: `F = ∂f/∂x`
    /// - `q` — process noise covariance (pass `None` for zero process noise)
    ///
    /// Updates: `x = f(x)`, `P = γ · F P Fᵀ + Q`.
    pub fn predict(
        &mut self,
        f: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, N>,
        fj: impl Fn(&ColumnVector<T, N>) -> Matrix<T, N, N>,
        q: Option<&Matrix<T, N, N>>,
    ) {
        let big_f = fj(&self.x);
        self.x = f(&self.x);
        self.p = big_f * self.p * big_f.transpose() * self.gamma;
        if let Some(q) = q {
            self.p = self.p + *q;
        }
        let half = T::from(0.5).unwrap();
        self.p = (self.p + self.p.transpose()) * half;
        apply_var_floor(&mut self.p, self.min_variance);
    }

    /// Predict step with finite-difference Jacobian.
    ///
    /// Computes the Jacobian of `f` automatically using forward differences.
    /// Pass `None` for `q` if there is no process noise.
    pub fn predict_fd(
        &mut self,
        f: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, N>,
        q: Option<&Matrix<T, N, N>>,
    ) {
        let big_f = fd_jacobian(&f, &self.x);
        self.x = f(&self.x);
        self.p = big_f * self.p * big_f.transpose() * self.gamma;
        if let Some(q) = q {
            self.p = self.p + *q;
        }
        let half = T::from(0.5).unwrap();
        self.p = (self.p + self.p.transpose()) * half;
        apply_var_floor(&mut self.p, self.min_variance);
    }

    /// Update step with explicit measurement Jacobian.
    ///
    /// - `z` — measurement vector
    /// - `h` — measurement model: `z = h(x)`
    /// - `hj` — Jacobian of `h` evaluated at `x`: `H = ∂h/∂x`
    /// - `r` — measurement noise covariance
    ///
    /// Uses Joseph form for numerical stability:
    /// `P = (I - KH) P (I - KH)ᵀ + K R Kᵀ`
    ///
    /// Returns the Normalized Innovation Squared (NIS): `yᵀ S⁻¹ y`.
    pub fn update(
        &mut self,
        z: &ColumnVector<T, M>,
        h: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, M>,
        hj: impl Fn(&ColumnVector<T, N>) -> Matrix<T, M, N>,
        r: &Matrix<T, M, M>,
    ) -> Result<T, EstimateError> {
        let big_h = hj(&self.x);
        let y = *z - h(&self.x); // innovation
        let s = big_h * self.p * big_h.transpose() + *r; // innovation covariance

        // K = P Hᵀ S⁻¹  — S is SPD, so use Cholesky inverse (with jitter fallback).
        let s_inv = cholesky_with_jitter(&s)
            .map_err(|_| EstimateError::SingularInnovation)?
            .inverse();
        let k = self.p * big_h.transpose() * s_inv; // N×M

        // NIS = yᵀ S⁻¹ y
        let nis = (y.transpose() * s_inv * y)[(0, 0)];

        self.x = self.x + k * y;

        // Joseph form: P = (I - KH) P (I - KH)ᵀ + K R Kᵀ
        let eye: Matrix<T, N, N> = Matrix::eye();
        let i_kh = eye - k * big_h;
        self.p = i_kh * self.p * i_kh.transpose() + k * *r * k.transpose();
        let half = T::from(0.5).unwrap();
        self.p = (self.p + self.p.transpose()) * half;
        apply_var_floor(&mut self.p, self.min_variance);

        Ok(nis)
    }

    /// Update step with finite-difference measurement Jacobian.
    ///
    /// Computes the Jacobian of `h` automatically using forward differences.
    ///
    /// Returns the Normalized Innovation Squared (NIS): `yᵀ S⁻¹ y`.
    pub fn update_fd(
        &mut self,
        z: &ColumnVector<T, M>,
        h: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, M>,
        r: &Matrix<T, M, M>,
    ) -> Result<T, EstimateError> {
        let big_h = fd_jacobian(&h, &self.x);
        let y = *z - h(&self.x);
        let s = big_h * self.p * big_h.transpose() + *r;

        let s_inv = cholesky_with_jitter(&s)
            .map_err(|_| EstimateError::SingularInnovation)?
            .inverse();
        let k = self.p * big_h.transpose() * s_inv;

        // NIS = yᵀ S⁻¹ y
        let nis = (y.transpose() * s_inv * y)[(0, 0)];

        self.x = self.x + k * y;

        let eye: Matrix<T, N, N> = Matrix::eye();
        let i_kh = eye - k * big_h;
        self.p = i_kh * self.p * i_kh.transpose() + k * *r * k.transpose();
        let half = T::from(0.5).unwrap();
        self.p = (self.p + self.p.transpose()) * half;
        apply_var_floor(&mut self.p, self.min_variance);

        Ok(nis)
    }

    /// Update with innovation gating — skips state update if NIS exceeds `gate`.
    ///
    /// Returns `Ok(None)` when the measurement is rejected (outlier), or
    /// `Ok(Some(nis))` when accepted and applied.
    ///
    /// A chi-squared table gives typical gate thresholds:
    /// M=1 → 99%: 6.63 | M=2 → 9.21 | M=3 → 11.34 | M=6 → 16.81
    pub fn update_gated(
        &mut self,
        z: &ColumnVector<T, M>,
        h: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, M>,
        hj: impl Fn(&ColumnVector<T, N>) -> Matrix<T, M, N>,
        r: &Matrix<T, M, M>,
        gate: T,
    ) -> Result<Option<T>, EstimateError> {
        // Compute NIS without modifying state.
        let big_h = hj(&self.x);
        let y = *z - h(&self.x);
        let s = big_h * self.p * big_h.transpose() + *r;
        let s_inv = cholesky_with_jitter(&s)
            .map_err(|_| EstimateError::SingularInnovation)?
            .inverse();
        let nis = (y.transpose() * s_inv * y)[(0, 0)];
        if nis > gate {
            return Ok(None);
        }
        let nis = self.update(z, h, hj, r)?;
        Ok(Some(nis))
    }

    /// Update with gating and finite-difference Jacobian.
    ///
    /// Returns `Ok(None)` when rejected, `Ok(Some(nis))` when accepted.
    pub fn update_fd_gated(
        &mut self,
        z: &ColumnVector<T, M>,
        h: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, M>,
        r: &Matrix<T, M, M>,
        gate: T,
    ) -> Result<Option<T>, EstimateError> {
        let big_h = fd_jacobian(&h, &self.x);
        let y = *z - h(&self.x);
        let s = big_h * self.p * big_h.transpose() + *r;
        let s_inv = cholesky_with_jitter(&s)
            .map_err(|_| EstimateError::SingularInnovation)?
            .inverse();
        let nis = (y.transpose() * s_inv * y)[(0, 0)];
        if nis > gate {
            return Ok(None);
        }
        let nis = self.update_fd(z, h, r)?;
        Ok(Some(nis))
    }

    /// Iterated EKF update — re-linearizes at the current iterate until convergence.
    ///
    /// Substantially more accurate than the standard EKF for highly nonlinear
    /// measurement models (e.g., angle/range measurements, attitude-to-pixel projection).
    ///
    /// At each iteration the linearization point advances:
    /// `x_{i+1} = x̄ + K_i (z - h(x_i) - H_i (x̄ - x_i))`
    ///
    /// Iteration stops when `‖x_{i+1} - x_i‖² < tol²`. The final covariance is
    /// updated with the Joseph form at the converged linearization point.
    ///
    /// Returns the Normalized Innovation Squared at the converged solution.
    pub fn update_iterated(
        &mut self,
        z: &ColumnVector<T, M>,
        h: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, M>,
        hj: impl Fn(&ColumnVector<T, N>) -> Matrix<T, M, N>,
        r: &Matrix<T, M, M>,
        max_iter: usize,
        tol: T,
    ) -> Result<T, EstimateError> {
        let x_pred = self.x;
        let p_pred = self.p;
        let tol_sq = tol * tol;

        let mut x_iter = x_pred;
        for _ in 0..max_iter {
            let big_h = hj(&x_iter);
            // Linearized innovation anchored at x_pred, evaluated at x_iter:
            // y = z - h(x_iter) - H*(x_pred - x_iter) = z - h(x_iter) + H*(x_iter - x_pred)
            let y = *z - h(&x_iter) + big_h * (x_iter - x_pred);
            let s = big_h * p_pred * big_h.transpose() + *r;
            let s_inv = cholesky_with_jitter(&s)
                .map_err(|_| EstimateError::SingularInnovation)?
                .inverse();
            let k = p_pred * big_h.transpose() * s_inv;
            let x_new = x_pred + k * y;

            let delta = x_new - x_iter;
            let sq_norm = delta.frobenius_norm_squared();
            x_iter = x_new;
            if sq_norm < tol_sq {
                break;
            }
        }

        // Final covariance update (Joseph form) at the converged linearization point.
        let big_h = hj(&x_iter);
        let s = big_h * p_pred * big_h.transpose() + *r;
        let s_inv = cholesky_with_jitter(&s)
            .map_err(|_| EstimateError::SingularInnovation)?
            .inverse();
        let k = p_pred * big_h.transpose() * s_inv;

        let y_final = *z - h(&x_iter);
        let nis = (y_final.transpose() * s_inv * y_final)[(0, 0)];

        self.x = x_iter;
        let eye: Matrix<T, N, N> = Matrix::eye();
        let i_kh = eye - k * big_h;
        self.p = i_kh * p_pred * i_kh.transpose() + k * *r * k.transpose();
        let half = T::from(0.5).unwrap();
        self.p = (self.p + self.p.transpose()) * half;
        apply_var_floor(&mut self.p, self.min_variance);

        Ok(nis)
    }

    /// Iterated EKF update with finite-difference Jacobian.
    ///
    /// See [`update_iterated`](Self::update_iterated) for details.
    pub fn update_fd_iterated(
        &mut self,
        z: &ColumnVector<T, M>,
        h: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, M>,
        r: &Matrix<T, M, M>,
        max_iter: usize,
        tol: T,
    ) -> Result<T, EstimateError> {
        let x_pred = self.x;
        let p_pred = self.p;
        let tol_sq = tol * tol;

        let mut x_iter = x_pred;
        for _ in 0..max_iter {
            let big_h = fd_jacobian(&h, &x_iter);
            let y = *z - h(&x_iter) + big_h * (x_iter - x_pred);
            let s = big_h * p_pred * big_h.transpose() + *r;
            let s_inv = cholesky_with_jitter(&s)
                .map_err(|_| EstimateError::SingularInnovation)?
                .inverse();
            let k = p_pred * big_h.transpose() * s_inv;
            let x_new = x_pred + k * y;

            let delta = x_new - x_iter;
            let sq_norm = delta.frobenius_norm_squared();
            x_iter = x_new;
            if sq_norm < tol_sq {
                break;
            }
        }

        // Final covariance update at converged point.
        let big_h = fd_jacobian(&h, &x_iter);
        let s = big_h * p_pred * big_h.transpose() + *r;
        let s_inv = cholesky_with_jitter(&s)
            .map_err(|_| EstimateError::SingularInnovation)?
            .inverse();
        let k = p_pred * big_h.transpose() * s_inv;

        let y_final = *z - h(&x_iter);
        let nis = (y_final.transpose() * s_inv * y_final)[(0, 0)];

        self.x = x_iter;
        let eye: Matrix<T, N, N> = Matrix::eye();
        let i_kh = eye - k * big_h;
        self.p = i_kh * p_pred * i_kh.transpose() + k * *r * k.transpose();
        let half = T::from(0.5).unwrap();
        self.p = (self.p + self.p.transpose()) * half;
        apply_var_floor(&mut self.p, self.min_variance);

        Ok(nis)
    }
}
