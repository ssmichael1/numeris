//! State estimation: EKF, UKF, SR-UKF, CKF, RTS smoother, batch least-squares.
//!
//! Filters use closure-based dynamics and measurement models with const-generic
//! state (`N`) and measurement (`M`) dimensions. The EKF and `BatchLsq` are
//! fully no-std compatible; the sigma-point filters (UKF, SR-UKF, CKF) and
//! the RTS smoother require `alloc` for temporary storage.
//!
//! # Extended Kalman Filter
//!
//! ```
//! use numeris::estimate::Ekf;
//! use numeris::{ColumnVector, Matrix};
//!
//! // 2-state constant-velocity, 1 measurement (position only)
//! let x0 = ColumnVector::from_column([0.0_f64, 1.0]); // [pos, vel]
//! let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
//! let mut ekf = Ekf::<f64, 2, 1>::new(x0, p0);
//!
//! let dt = 0.1;
//! let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
//! let r = Matrix::new([[0.5]]);
//!
//! // Predict with explicit Jacobian
//! ekf.predict(
//!     |x| ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]]),
//!     |_x| Matrix::new([[1.0, dt], [0.0, 1.0]]),
//!     Some(&q),
//! );
//!
//! // Update with measurement
//! ekf.update(
//!     &ColumnVector::from_column([0.12]),
//!     |x| ColumnVector::from_column([x[(0, 0)]]),
//!     |_x| Matrix::new([[1.0, 0.0]]),
//!     &r,
//! ).unwrap();
//! ```
//!
//! # Unscented Kalman Filter
//!
//! ```
//! use numeris::estimate::Ukf;
//! use numeris::{ColumnVector, Matrix};
//!
//! let x0 = ColumnVector::from_column([0.0_f64, 1.0]);
//! let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
//! let mut ukf = Ukf::<f64, 2, 1>::new(x0, p0);
//!
//! let dt = 0.1;
//! let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
//! let r = Matrix::new([[0.5]]);
//!
//! ukf.predict(
//!     |x| ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]]),
//!     Some(&q),
//! ).unwrap();
//!
//! ukf.update(
//!     &ColumnVector::from_column([0.12]),
//!     |x| ColumnVector::from_column([x[(0, 0)]]),
//!     &r,
//! ).unwrap();
//! ```
//!
//! # Batch Least-Squares (no-std)
//!
//! ```
//! use numeris::estimate::BatchLsq;
//! use numeris::{ColumnVector, Matrix};
//!
//! let mut lsq = BatchLsq::<f64, 1>::new();
//!
//! // Accumulate scalar observations: z = x + noise
//! let h = Matrix::new([[1.0_f64]]);
//! let r = Matrix::new([[0.1]]);
//! lsq.add_observation(&ColumnVector::from_column([1.05]), &h, &r).unwrap();
//! lsq.add_observation(&ColumnVector::from_column([0.95]), &h, &r).unwrap();
//!
//! let (x, p) = lsq.solve().unwrap();
//! assert!((x[(0, 0)] - 1.0).abs() < 0.1);
//! ```

mod ekf;
#[cfg(feature = "alloc")]
mod ukf;
mod cholupdate;
#[cfg(feature = "alloc")]
mod srukf;
#[cfg(feature = "alloc")]
mod ckf;
#[cfg(feature = "alloc")]
mod rts;
mod batch;

#[cfg(test)]
mod tests;

pub use ekf::Ekf;
#[cfg(feature = "alloc")]
pub use ukf::Ukf;
#[cfg(feature = "alloc")]
pub use srukf::SrUkf;
#[cfg(feature = "alloc")]
pub use ckf::Ckf;
#[cfg(feature = "alloc")]
pub use rts::{EkfStep, rts_smooth};
pub use batch::BatchLsq;

use crate::matrix::vector::ColumnVector;
use crate::traits::FloatScalar;
use crate::Matrix;

/// Errors from state estimation algorithms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EstimateError {
    /// Covariance matrix is not positive definite (Cholesky failed).
    CovarianceNotPD,
    /// Innovation covariance is singular (cannot compute Kalman gain).
    SingularInnovation,
    /// Cholesky downdate produced a non-positive-definite result.
    CholdowndateFailed,
}

impl core::fmt::Display for EstimateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EstimateError::CovarianceNotPD => {
                write!(f, "covariance matrix is not positive definite")
            }
            EstimateError::SingularInnovation => {
                write!(f, "innovation covariance is singular")
            }
            EstimateError::CholdowndateFailed => {
                write!(f, "Cholesky downdate failed: result not positive definite")
            }
        }
    }
}

/// Forward-difference Jacobian of `f: ColumnVector<T,N> → ColumnVector<T,M>`.
///
/// Uses step size `h_j = sqrt(ε) * max(|x_j|, 1)` for each component.
pub(crate) fn fd_jacobian<T: FloatScalar, const N: usize, const M: usize>(
    f: &impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, M>,
    x: &ColumnVector<T, N>,
) -> Matrix<T, M, N> {
    let sqrt_eps = T::epsilon().sqrt();
    let f0 = f(x);
    let mut jac = Matrix::<T, M, N>::zeros();

    for j in 0..N {
        let xj = x[(j, 0)];
        let h = sqrt_eps * xj.abs().max(T::one());
        let mut x_pert = *x;
        x_pert[(j, 0)] = xj + h;
        let f_pert = f(&x_pert);

        for i in 0..M {
            jac[(i, j)] = (f_pert[(i, 0)] - f0[(i, 0)]) / h;
        }
    }

    jac
}
