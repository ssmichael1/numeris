extern crate alloc;
use alloc::vec::Vec;

use crate::matrix::vector::ColumnVector;
use crate::traits::FloatScalar;
use crate::Matrix;

use super::EstimateError;

/// Record of a single EKF forward-pass step, stored by the user for RTS smoothing.
///
/// Each step captures the state/covariance before and after the measurement
/// update, plus the dynamics Jacobian used in the predict step.
///
/// # Fields
///
/// - `x_predicted` / `p_predicted` — state & covariance after `predict`, before `update`
/// - `x_updated` / `p_updated` — state & covariance after `update`
/// - `f_jacobian` — the `F` matrix used in the predict step
pub struct EkfStep<T: FloatScalar, const N: usize> {
    /// State after predict, before update.
    pub x_predicted: ColumnVector<T, N>,
    /// Covariance after predict, before update.
    pub p_predicted: Matrix<T, N, N>,
    /// State after update.
    pub x_updated: ColumnVector<T, N>,
    /// Covariance after update.
    pub p_updated: Matrix<T, N, N>,
    /// Dynamics Jacobian `F` used in the predict step.
    pub f_jacobian: Matrix<T, N, N>,
}

/// Rauch–Tung–Striebel fixed-interval smoother (backward pass).
///
/// Given a sequence of forward EKF steps (predict → update at each timestep),
/// returns smoothed `(x, P)` pairs that incorporate future measurements.
///
/// The smoothed estimates always have equal or smaller covariance than the
/// filtered estimates.
///
/// # Arguments
///
/// - `steps` — forward EKF steps in chronological order
///
/// # Returns
///
/// `Vec<(ColumnVector<T,N>, Matrix<T,N,N>)>` of smoothed (state, covariance)
/// for each timestep, in the same order as `steps`.
///
/// Returns `SingularInnovation` if any predicted covariance is singular
/// (required for smoother gain computation).
///
/// # Example
///
/// ```
/// use numeris::estimate::{Ekf, EkfStep, rts_smooth};
/// use numeris::{ColumnVector, Matrix};
///
/// let dt = 0.1_f64;
/// let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
/// let r = Matrix::new([[0.5]]);
/// let f_jac = Matrix::new([[1.0, dt], [0.0, 1.0]]);
///
/// let mut ekf = Ekf::<f64, 2, 1>::new(
///     ColumnVector::from_column([0.0, 0.0]),
///     Matrix::new([[10.0, 0.0], [0.0, 10.0]]),
/// );
///
/// let measurements = [0.1, 0.22, 0.35, 0.45, 0.58];
/// let mut steps = Vec::new();
///
/// for &z_val in &measurements {
///     // Predict
///     let x_pre = ekf.x;
///     ekf.predict(
///         |x| ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]]),
///         |_| f_jac,
///         Some(&q),
///     );
///     let x_predicted = ekf.x;
///     let p_predicted = ekf.p;
///
///     // Update
///     ekf.update(
///         &ColumnVector::from_column([z_val]),
///         |x| ColumnVector::from_column([x[(0, 0)]]),
///         |_| Matrix::new([[1.0, 0.0]]),
///         &r,
///     ).unwrap();
///
///     steps.push(EkfStep {
///         x_predicted,
///         p_predicted,
///         x_updated: ekf.x,
///         p_updated: ekf.p,
///         f_jacobian: f_jac,
///     });
/// }
///
/// let smoothed = rts_smooth(&steps).unwrap();
/// assert_eq!(smoothed.len(), steps.len());
/// ```
pub fn rts_smooth<T: FloatScalar, const N: usize>(
    steps: &[EkfStep<T, N>],
) -> Result<Vec<(ColumnVector<T, N>, Matrix<T, N, N>)>, EstimateError> {
    let n = steps.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut smoothed: Vec<(ColumnVector<T, N>, Matrix<T, N, N>)> = Vec::with_capacity(n);
    // Pre-fill with zeros so we can index backwards
    for _ in 0..n {
        smoothed.push((ColumnVector::zeros(), Matrix::zeros()));
    }

    // Last step: smoothed = filtered
    smoothed[n - 1] = (steps[n - 1].x_updated, steps[n - 1].p_updated);

    // Backward pass
    for k in (0..n - 1).rev() {
        // Smoother gain: G_k = P_updated_k · F_{k+1}ᵀ · P_predicted_{k+1}⁻¹
        //
        // Note: F_{k+1} is the Jacobian used in the predict step that produced
        // steps[k+1].x_predicted from steps[k].x_updated. So we use
        // steps[k+1].f_jacobian.
        let f_next = steps[k + 1].f_jacobian;
        let p_pred_next_inv = steps[k + 1]
            .p_predicted
            .inverse()
            .map_err(|_| EstimateError::SingularInnovation)?;

        let g = steps[k].p_updated * f_next.transpose() * p_pred_next_inv;

        // Smoothed state
        let x_smooth = steps[k].x_updated
            + g * (smoothed[k + 1].0 - steps[k + 1].x_predicted);

        // Smoothed covariance
        let p_smooth = steps[k].p_updated
            + g * (smoothed[k + 1].1 - steps[k + 1].p_predicted) * g.transpose();

        smoothed[k] = (x_smooth, p_smooth);
    }

    Ok(smoothed)
}
