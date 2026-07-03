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
//! use numeris::{Vector, Matrix};
//!
//! // 2-state constant-velocity, 1 measurement (position only)
//! let x0 = Vector::from_array([0.0_f64, 1.0]); // [pos, vel]
//! let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
//! let mut ekf = Ekf::<f64, 2, 1>::new(x0, p0);
//!
//! let dt = 0.1;
//! let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
//! let r = Matrix::new([[0.5]]);
//!
//! // Predict with explicit Jacobian
//! ekf.predict(
//!     |x| Vector::from_array([x[0] + dt * x[1], x[1]]),
//!     |_x| Matrix::new([[1.0, dt], [0.0, 1.0]]),
//!     Some(&q),
//! );
//!
//! // Update with measurement
//! ekf.update(
//!     &Vector::from_array([0.12]),
//!     |x| Vector::from_array([x[0]]),
//!     |_x| Matrix::new([[1.0, 0.0]]),
//!     &r,
//! ).unwrap();
//! ```
//!
//! # Unscented Kalman Filter
//!
//! ```
//! use numeris::estimate::Ukf;
//! use numeris::{Vector, Matrix};
//!
//! let x0 = Vector::from_array([0.0_f64, 1.0]);
//! let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
//! let mut ukf = Ukf::<f64, 2, 1>::new(x0, p0);
//!
//! let dt = 0.1;
//! let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
//! let r = Matrix::new([[0.5]]);
//!
//! ukf.predict(
//!     |x| Vector::from_array([x[0] + dt * x[1], x[1]]),
//!     Some(&q),
//! ).unwrap();
//!
//! ukf.update(
//!     &Vector::from_array([0.12]),
//!     |x| Vector::from_array([x[0]]),
//!     &r,
//! ).unwrap();
//! ```
//!
//! # Batch Least-Squares (no-std)
//!
//! ```
//! use numeris::estimate::BatchLsq;
//! use numeris::{Vector, Matrix};
//!
//! let mut lsq = BatchLsq::<f64, 1>::new();
//!
//! // Accumulate scalar observations: z = x + noise
//! let h = Matrix::new([[1.0_f64]]);
//! let r = Matrix::new([[0.1]]);
//! lsq.add_observation(&Vector::from_array([1.05]), &h, &r).unwrap();
//! lsq.add_observation(&Vector::from_array([0.95]), &h, &r).unwrap();
//!
//! let (x, p) = lsq.solve().unwrap();
//! assert!((x[0] - 1.0).abs() < 0.1);
//! ```

mod batch;
#[cfg(feature = "alloc")]
mod ckf;
mod ekf;
#[cfg(feature = "alloc")]
mod rts;
#[cfg(feature = "alloc")]
mod srukf;
#[cfg(feature = "alloc")]
mod ukf;

#[cfg(test)]
mod tests;

pub use batch::BatchLsq;
#[cfg(feature = "alloc")]
pub use ckf::Ckf;
pub use ekf::Ekf;
#[cfg(feature = "alloc")]
pub use rts::{rts_smooth, EkfStep};
#[cfg(feature = "alloc")]
pub use srukf::SrUkf;
#[cfg(feature = "alloc")]
pub use ukf::Ukf;

use crate::linalg::CholeskyDecomposition;
use crate::matrix::vector::Vector;
use crate::traits::FloatScalar;
use crate::Matrix;

/// Errors from state estimation algorithms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EstimateError {
    /// Covariance matrix is not positive definite (Cholesky failed).
    CovarianceNotPD,
    /// Innovation covariance is singular (cannot compute Kalman gain).
    SingularInnovation,
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
        }
    }
}

/// Attempt Cholesky decomposition, retrying with increasing diagonal jitter if needed.
///
/// Accumulated floating-point errors can make a mathematically positive-definite matrix
/// appear non-PD by ~1e-14. Rather than failing immediately, this tries adding
/// `ε·I` for ε ∈ {1e-9, 1e-7, 1e-5} before giving up.
pub(crate) fn cholesky_with_jitter<T: FloatScalar, const N: usize>(
    p: &Matrix<T, N, N>,
) -> Result<CholeskyDecomposition<T, N>, EstimateError> {
    if let Ok(c) = CholeskyDecomposition::new(p) {
        return Ok(c);
    }
    // Fixed jitter ladder ε ∈ {1e-9, 1e-7, 1e-5}. Written as literals (not
    // `10f64.powi(..)`) so this compiles in no_std, where `f64::powi` is absent.
    for eps in [1e-9_f64, 1e-7, 1e-5] {
        let eps = T::from(eps).unwrap();
        let mut pj = *p;
        for i in 0..N {
            pj[(i, i)] = pj[(i, i)] + eps;
        }
        if let Ok(c) = CholeskyDecomposition::new(&pj) {
            return Ok(c);
        }
    }
    Err(EstimateError::CovarianceNotPD)
}

/// Clamp the diagonal of `p` so that `p[i,i] >= min_var` for all `i`.
///
/// A no-op when `min_var <= 0`. Prevents the covariance from degenerating to
/// zero or going negative after many updates.
pub(crate) fn apply_var_floor<T: FloatScalar, const N: usize>(p: &mut Matrix<T, N, N>, min_var: T) {
    if min_var > T::zero() {
        for i in 0..N {
            if p[(i, i)] < min_var {
                p[(i, i)] = min_var;
            }
        }
    }
}

/// Symmetrize a covariance and apply the diagonal variance floor — the common
/// tail of every sigma-point predict/update. `P ← (P + Pᵀ)/2`, then floor.
#[cfg(feature = "alloc")]
pub(crate) fn symmetrize_and_floor<T: FloatScalar, const N: usize>(
    p: &mut Matrix<T, N, N>,
    min_variance: T,
) {
    let half = T::from(0.5).unwrap();
    *p = (*p + p.transpose()) * half;
    apply_var_floor(p, min_variance);
}

/// Store a sigma-point filter's predicted mean/covariance:
/// `P ← γ·P_sigma + Q`, symmetrized and floored. Shared by UKF and CKF `predict`.
#[cfg(feature = "alloc")]
pub(crate) fn store_predicted<T: FloatScalar, const N: usize>(
    x: &mut Vector<T, N>,
    p: &mut Matrix<T, N, N>,
    x_mean: Vector<T, N>,
    p_sigma: Matrix<T, N, N>,
    gamma: T,
    q: Option<&Matrix<T, N, N>>,
    min_variance: T,
) {
    let scaled = p_sigma * gamma;
    *x = x_mean;
    *p = match q {
        Some(q) => scaled + *q,
        None => scaled,
    };
    symmetrize_and_floor(p, min_variance);
}

/// Kalman gain and symmetric covariance update shared by the UKF and CKF:
/// `K = Pxz·S⁻¹`, `x += K·innovation`, `P -= K·S·Kᵀ`, then symmetrize + floor.
/// The `P - K S Kᵀ` form is manifestly PSD-subtracted.
#[cfg(feature = "alloc")]
pub(crate) fn sigma_point_update<T: FloatScalar, const N: usize, const M: usize>(
    x: &mut Vector<T, N>,
    p: &mut Matrix<T, N, N>,
    s_mat: &Matrix<T, M, M>,
    s_inv: &Matrix<T, M, M>,
    pxz: &Matrix<T, N, M>,
    innovation: &Vector<T, M>,
    min_variance: T,
) {
    let k = *pxz * *s_inv;
    *x += k * *innovation;
    *p -= k * *s_mat * k.transpose();
    symmetrize_and_floor(p, min_variance);
}

/// Forward-difference Jacobian of `f: Vector<T,N> → Vector<T,M>`.
///
/// Uses step size `h_j = sqrt(ε) * max(|x_j|, 1)` for each component.
pub(crate) fn fd_jacobian<T: FloatScalar, const N: usize, const M: usize>(
    f: &impl Fn(&Vector<T, N>) -> Vector<T, M>,
    x: &Vector<T, N>,
) -> Matrix<T, M, N> {
    let sqrt_eps = T::epsilon().sqrt();
    let f0 = f(x);
    let mut jac = Matrix::<T, M, N>::zeros();

    for j in 0..N {
        let xj = x[j];
        let h = sqrt_eps * xj.abs().max(T::one());
        let mut x_pert = *x;
        x_pert[j] = xj + h;
        let f_pert = f(&x_pert);

        for i in 0..M {
            jac[(i, j)] = (f_pert[i] - f0[i]) / h;
        }
    }

    jac
}
