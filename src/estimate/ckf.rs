extern crate alloc;
use alloc::vec::Vec;

use crate::matrix::vector::ColumnVector;
use crate::traits::FloatScalar;
use crate::Matrix;

use super::{apply_var_floor, cholesky_with_jitter, EstimateError};

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
    /// Minimum allowed diagonal variance (0 = disabled).
    min_variance: T,
    /// Fading-memory factor γ≥1 applied to the sigma-point covariance (1 = standard).
    gamma: T,
}

impl<T: FloatScalar, const N: usize, const M: usize> Ckf<T, N, M> {
    /// Create a new CKF with initial state `x0` and covariance `p0`.
    pub fn new(x0: ColumnVector<T, N>, p0: Matrix<T, N, N>) -> Self {
        Self {
            x: x0,
            p: p0,
            min_variance: T::zero(),
            gamma: T::one(),
        }
    }

    /// Set a minimum diagonal variance floor applied after every predict/update.
    pub fn with_min_variance(mut self, min_variance: T) -> Self {
        self.min_variance = min_variance;
        self
    }

    /// Set a fading-memory factor `γ ≥ 1` applied to the propagated covariance.
    ///
    /// The predicted covariance becomes `γ · P_cubature + Q`. Default is `1.0`.
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

    /// Generate 2N cubature points from state and Cholesky factor.
    fn cubature_points(
        x: &ColumnVector<T, N>,
        l: &Matrix<T, N, N>,
    ) -> Vec<ColumnVector<T, N>> {
        let sqrt_n = T::from(N).unwrap().sqrt();
        let mut points = Vec::with_capacity(2 * N);

        for i in 0..N {
            // x ± sqrt(N) · L·e_i = x ± sqrt(N) · col_i(L)
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
    /// reconstructs the predicted mean and covariance as `γ · P_cubature + Q`.
    pub fn predict(
        &mut self,
        f: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, N>,
        q: Option<&Matrix<T, N, N>>,
    ) -> Result<(), EstimateError> {
        let chol = cholesky_with_jitter(&self.p)?;
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

        // Cubature covariance (without Q)
        let mut p_new = Matrix::<T, N, N>::zeros();
        for pt in &propagated {
            let d = *pt - x_mean;
            for r in 0..N {
                for c in 0..N {
                    p_new[(r, c)] = p_new[(r, c)] + w * d[(r, 0)] * d[(c, 0)];
                }
            }
        }

        // Fading memory: scale cubature covariance by γ before adding Q.
        p_new = p_new * self.gamma;
        self.x = x_mean;
        self.p = if let Some(q) = q { p_new + *q } else { p_new };
        let half = T::from(0.5).unwrap();
        self.p = (self.p + self.p.transpose()) * half;
        apply_var_floor(&mut self.p, self.min_variance);

        Ok(())
    }

    /// Update step.
    ///
    /// - `z` — measurement vector
    /// - `h` — measurement model: `z = h(x)`
    /// - `r` — measurement noise covariance
    ///
    /// Covariance update uses the symmetric `P - K S Kᵀ` form.
    ///
    /// Returns the Normalized Innovation Squared (NIS): `yᵀ S⁻¹ y`.
    pub fn update(
        &mut self,
        z: &ColumnVector<T, M>,
        h: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, M>,
        r: &Matrix<T, M, M>,
    ) -> Result<T, EstimateError> {
        let chol = cholesky_with_jitter(&self.p)?;
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

        // Innovation covariance S = (1/2N) Σ (z_i - z̄)(z_i - z̄)ᵀ + R
        let mut s_mat = Matrix::<T, M, M>::zeros();
        for zp in &z_points {
            let dz = *zp - z_mean;
            for ri in 0..M {
                for ci in 0..M {
                    s_mat[(ri, ci)] = s_mat[(ri, ci)] + w * dz[(ri, 0)] * dz[(ci, 0)];
                }
            }
        }
        s_mat = s_mat + *r;

        // Cross-covariance Pxz = (1/2N) Σ (x_i - x̄)(z_i - z̄)ᵀ
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
        let s_inv = cholesky_with_jitter(&s_mat)
            .map_err(|_| EstimateError::SingularInnovation)?
            .inverse();
        let k = pxz * s_inv;

        // Innovation and NIS = yᵀ S⁻¹ y
        let innovation = *z - z_mean;
        let nis = (innovation.transpose() * s_inv * innovation)[(0, 0)];

        // Update state and covariance: P = P - K·S·Kᵀ (symmetric, manifestly PSD-subtracted)
        self.x = self.x + k * innovation;
        self.p = self.p - k * s_mat * k.transpose();
        let half = T::from(0.5).unwrap();
        self.p = (self.p + self.p.transpose()) * half;
        apply_var_floor(&mut self.p, self.min_variance);

        Ok(nis)
    }

    /// Update with innovation gating — skips state update if NIS exceeds `gate`.
    ///
    /// Returns `Ok(None)` when rejected, `Ok(Some(nis))` when accepted.
    ///
    /// Chi-squared thresholds: M=1 → 99%: 6.63 | M=2 → 9.21 | M=3 → 11.34
    pub fn update_gated(
        &mut self,
        z: &ColumnVector<T, M>,
        h: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, M>,
        r: &Matrix<T, M, M>,
        gate: T,
    ) -> Result<Option<T>, EstimateError> {
        // Compute NIS without modifying state.
        let chol = cholesky_with_jitter(&self.p)?;
        let l = chol.l_full();

        let points = Self::cubature_points(&self.x, &l);
        let w = T::one() / T::from(2 * N).unwrap();

        let mut z_points: Vec<ColumnVector<T, M>> = Vec::with_capacity(2 * N);
        let mut z_mean = ColumnVector::<T, M>::zeros();

        for pt in &points {
            let zp = h(pt);
            for r_i in 0..M {
                z_mean[(r_i, 0)] = z_mean[(r_i, 0)] + w * zp[(r_i, 0)];
            }
            z_points.push(zp);
        }

        let mut s_mat = Matrix::<T, M, M>::zeros();
        for zp in &z_points {
            let dz = *zp - z_mean;
            for ri in 0..M {
                for ci in 0..M {
                    s_mat[(ri, ci)] = s_mat[(ri, ci)] + w * dz[(ri, 0)] * dz[(ci, 0)];
                }
            }
        }
        s_mat = s_mat + *r;

        let s_inv = cholesky_with_jitter(&s_mat)
            .map_err(|_| EstimateError::SingularInnovation)?
            .inverse();
        let innovation = *z - z_mean;
        let nis = (innovation.transpose() * s_inv * innovation)[(0, 0)];

        if nis > gate {
            return Ok(None);
        }
        let nis = self.update(z, h, r)?;
        Ok(Some(nis))
    }
}
