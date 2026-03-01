use crate::matrix::vector::ColumnVector;
use crate::traits::FloatScalar;
use crate::Matrix;

use super::{fd_jacobian, EstimateError};

/// Extended Kalman Filter with const-generic state and measurement dimensions.
///
/// `N` is the state dimension, `M` is the measurement dimension.
/// The EKF linearizes nonlinear dynamics and measurement models via
/// user-supplied or finite-difference Jacobians.
///
/// All operations are stack-allocated — no heap, fully no-std compatible.
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
}

impl<T: FloatScalar, const N: usize, const M: usize> Ekf<T, N, M> {
    /// Create a new EKF with initial state `x0` and covariance `p0`.
    pub fn new(x0: ColumnVector<T, N>, p0: Matrix<T, N, N>) -> Self {
        Self { x: x0, p: p0 }
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
    /// Updates: `x = f(x)`, `P = F P Fᵀ + Q`.
    pub fn predict(
        &mut self,
        f: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, N>,
        fj: impl Fn(&ColumnVector<T, N>) -> Matrix<T, N, N>,
        q: Option<&Matrix<T, N, N>>,
    ) {
        let big_f = fj(&self.x);
        self.x = f(&self.x);
        self.p = big_f * self.p * big_f.transpose();
        if let Some(q) = q {
            self.p = self.p + *q;
        }
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
        self.p = big_f * self.p * big_f.transpose();
        if let Some(q) = q {
            self.p = self.p + *q;
        }
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
    pub fn update(
        &mut self,
        z: &ColumnVector<T, M>,
        h: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, M>,
        hj: impl Fn(&ColumnVector<T, N>) -> Matrix<T, M, N>,
        r: &Matrix<T, M, M>,
    ) -> Result<(), EstimateError> {
        let big_h = hj(&self.x);
        let y = *z - h(&self.x); // innovation
        let s = big_h * self.p * big_h.transpose() + *r; // innovation covariance

        // K = P Hᵀ S⁻¹  →  solve Sᵀ (P Hᵀ)ᵀ = Sᵀ (Hᵀᵀ Pᵀ) ... simpler: just invert S
        // For small M, LU inverse is fine.
        let s_inv = s.inverse().map_err(|_| EstimateError::SingularInnovation)?;
        let k = self.p * big_h.transpose() * s_inv; // N×M

        self.x = self.x + k * y;

        // Joseph form: P = (I - KH) P (I - KH)ᵀ + K R Kᵀ
        let eye: Matrix<T, N, N> = Matrix::eye();
        let i_kh = eye - k * big_h;
        self.p = i_kh * self.p * i_kh.transpose() + k * *r * k.transpose();

        Ok(())
    }

    /// Update step with finite-difference measurement Jacobian.
    ///
    /// Computes the Jacobian of `h` automatically using forward differences.
    pub fn update_fd(
        &mut self,
        z: &ColumnVector<T, M>,
        h: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, M>,
        r: &Matrix<T, M, M>,
    ) -> Result<(), EstimateError> {
        let big_h = fd_jacobian(&h, &self.x);
        let y = *z - h(&self.x);
        let s = big_h * self.p * big_h.transpose() + *r;

        let s_inv = s.inverse().map_err(|_| EstimateError::SingularInnovation)?;
        let k = self.p * big_h.transpose() * s_inv;

        self.x = self.x + k * y;

        let eye: Matrix<T, N, N> = Matrix::eye();
        let i_kh = eye - k * big_h;
        self.p = i_kh * self.p * i_kh.transpose() + k * *r * k.transpose();

        Ok(())
    }
}
