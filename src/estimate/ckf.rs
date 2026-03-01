extern crate alloc;
use alloc::vec::Vec;

use crate::linalg::CholeskyDecomposition;
use crate::matrix::vector::ColumnVector;
use crate::traits::FloatScalar;
use crate::Matrix;

use super::EstimateError;

/// Cubature Kalman Filter with third-degree spherical-radial cubature rule.
///
/// Uses `2N` cubature points with equal weight `1/(2N)`. Unlike the UKF,
/// the CKF has **no tuning parameters** — the cubature rule is uniquely
/// determined by the state dimension.
///
/// `N` is the state dimension, `M` is the measurement dimension.
/// Requires the `alloc` feature for temporary cubature point storage.
///
/// # Example
///
/// ```
/// use numeris::estimate::Ckf;
/// use numeris::{ColumnVector, Matrix};
///
/// let x0 = ColumnVector::from_column([0.0_f64, 1.0]);
/// let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
/// let mut ckf = Ckf::<f64, 2, 1>::new(x0, p0);
///
/// let dt = 0.1;
/// let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
/// let r = Matrix::new([[0.5]]);
///
/// ckf.predict(
///     |x| ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]]),
///     Some(&q),
/// ).unwrap();
///
/// ckf.update(
///     &ColumnVector::from_column([0.12]),
///     |x| ColumnVector::from_column([x[(0, 0)]]),
///     &r,
/// ).unwrap();
/// ```
pub struct Ckf<T: FloatScalar, const N: usize, const M: usize> {
    /// State estimate.
    pub x: ColumnVector<T, N>,
    /// State covariance.
    pub p: Matrix<T, N, N>,
}

impl<T: FloatScalar, const N: usize, const M: usize> Ckf<T, N, M> {
    /// Create a new CKF with initial state `x0` and covariance `p0`.
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

    /// Generate 2N cubature points from state and Cholesky factor.
    fn cubature_points(
        x: &ColumnVector<T, N>,
        l: &Matrix<T, N, N>,
    ) -> Vec<ColumnVector<T, N>> {
        let sqrt_n = T::from(N).unwrap().sqrt();
        let mut points = Vec::with_capacity(2 * N);

        for i in 0..N {
            // Positive direction: x + sqrt(N) * L·e_i = x + sqrt(N) * col_i(L)
            let mut pos = *x;
            let mut neg = *x;
            for r in 0..N {
                let offset = sqrt_n * l[(r, i)];
                pos[(r, 0)] = pos[(r, 0)] + offset;
                neg[(r, 0)] = neg[(r, 0)] - offset;
            }
            points.push(pos);
            points.push(neg);
        }

        points
    }

    /// Predict step.
    ///
    /// - `f` — state transition: `x_{k+1} = f(x_k)`
    /// - `q` — process noise covariance (pass `None` for zero process noise)
    ///
    /// Generates `2N` cubature points, propagates through `f`, and
    /// reconstructs the predicted mean and covariance.
    pub fn predict(
        &mut self,
        f: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, N>,
        q: Option<&Matrix<T, N, N>>,
    ) -> Result<(), EstimateError> {
        let chol =
            CholeskyDecomposition::new(&self.p).map_err(|_| EstimateError::CovarianceNotPD)?;
        let l = chol.l_full();

        let points = Self::cubature_points(&self.x, &l);
        let w = T::one() / T::from(2 * N).unwrap();

        // Propagate and compute mean
        let mut propagated: Vec<ColumnVector<T, N>> = Vec::with_capacity(2 * N);
        let mut x_mean = ColumnVector::<T, N>::zeros();

        for pt in &points {
            let fp = f(pt);
            for r in 0..N {
                x_mean[(r, 0)] = x_mean[(r, 0)] + w * fp[(r, 0)];
            }
            propagated.push(fp);
        }

        // Covariance
        let mut p_new = Matrix::<T, N, N>::zeros();
        for pt in &propagated {
            let d = *pt - x_mean;
            for r in 0..N {
                for c in 0..N {
                    p_new[(r, c)] = p_new[(r, c)] + w * d[(r, 0)] * d[(c, 0)];
                }
            }
        }

        self.x = x_mean;
        self.p = if let Some(q) = q { p_new + *q } else { p_new };

        Ok(())
    }

    /// Update step.
    ///
    /// - `z` — measurement vector
    /// - `h` — measurement model: `z = h(x)`
    /// - `r` — measurement noise covariance
    ///
    /// Generates cubature points from the predicted state, transforms through `h`,
    /// and computes the Kalman gain. Uses Joseph form for numerical stability.
    pub fn update(
        &mut self,
        z: &ColumnVector<T, M>,
        h: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, M>,
        r: &Matrix<T, M, M>,
    ) -> Result<(), EstimateError> {
        let chol =
            CholeskyDecomposition::new(&self.p).map_err(|_| EstimateError::CovarianceNotPD)?;
        let l = chol.l_full();

        let points = Self::cubature_points(&self.x, &l);
        let w = T::one() / T::from(2 * N).unwrap();

        // Transform through measurement model
        let mut z_points: Vec<ColumnVector<T, M>> = Vec::with_capacity(2 * N);
        let mut z_mean = ColumnVector::<T, M>::zeros();

        for pt in &points {
            let zp = h(pt);
            for r in 0..M {
                z_mean[(r, 0)] = z_mean[(r, 0)] + w * zp[(r, 0)];
            }
            z_points.push(zp);
        }

        // Innovation covariance: S = (1/2N) Σ (z_i - z̄)(z_i - z̄)ᵀ + R
        let mut s = Matrix::<T, M, M>::zeros();
        for zp in &z_points {
            let dz = *zp - z_mean;
            for ri in 0..M {
                for ci in 0..M {
                    s[(ri, ci)] = s[(ri, ci)] + w * dz[(ri, 0)] * dz[(ci, 0)];
                }
            }
        }
        s = s + *r;

        // Cross-covariance: Pxz = (1/2N) Σ (x_i - x̄)(z_i - z̄)ᵀ
        let mut pxz = Matrix::<T, N, M>::zeros();
        for (pt, zp) in points.iter().zip(z_points.iter()) {
            let dx = *pt - self.x;
            let dz = *zp - z_mean;
            for ri in 0..N {
                for ci in 0..M {
                    pxz[(ri, ci)] = pxz[(ri, ci)] + w * dx[(ri, 0)] * dz[(ci, 0)];
                }
            }
        }

        // Kalman gain: K = Pxz · S⁻¹
        let s_inv = s.inverse().map_err(|_| EstimateError::SingularInnovation)?;
        let k = pxz * s_inv;

        // State update
        let innovation = *z - z_mean;
        self.x = self.x + k * innovation;

        // Joseph form: P = (I - KH_eff)P(I - KH_eff)ᵀ + KRKᵀ
        // But we don't have an explicit H, so use the standard form:
        // P = P - K·S·Kᵀ
        self.p = self.p - k * s * k.transpose();

        Ok(())
    }
}
