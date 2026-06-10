extern crate alloc;
use alloc::vec::Vec;

use crate::matrix::vector::Vector;
use crate::traits::FloatScalar;
use crate::Matrix;

use super::{apply_var_floor, cholesky_with_jitter, EstimateError};

/// Square-Root Unscented Kalman Filter.
///
/// Propagates the Cholesky factor `S` where `P = S·Sᵀ` instead of the
/// covariance `P` directly. This guarantees positive-definiteness and
/// improves numerical conditioning, especially for ill-conditioned systems.
///
/// Uses Merwe-scaled sigma points (same as [`super::Ukf`]).
///
/// `N` is the state dimension, `M` is the measurement dimension.
/// Requires the `alloc` feature for temporary sigma point storage.
///
/// # Default parameters
///
/// `alpha=1.0`, `beta=2.0`, `kappa=0.0` — all sigma-point weights non-negative.
///
/// # Example
///
/// ```
/// use numeris::estimate::SrUkf;
/// use numeris::{Vector, Matrix};
///
/// let x0 = Vector::from_array([0.0_f64, 1.0]);
/// let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
/// let mut srukf = SrUkf::<f64, 2, 1>::from_covariance(x0, p0).unwrap();
///
/// let dt = 0.1;
/// let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
/// let r = Matrix::new([[0.5]]);
///
/// srukf.predict(
///     |x| Vector::from_array([x[0] + dt * x[1], x[1]]),
///     Some(&q),
/// ).unwrap();
///
/// srukf.update(
///     &Vector::from_array([0.12]),
///     |x| Vector::from_array([x[0]]),
///     &r,
/// ).unwrap();
/// ```
pub struct SrUkf<T: FloatScalar, const N: usize, const M: usize> {
    /// State estimate.
    pub x: Vector<T, N>,
    /// Lower-triangular Cholesky factor: `P = S·Sᵀ`.
    pub s: Matrix<T, N, N>,
    alpha: T,
    beta: T,
    kappa: T,
    /// Minimum allowed diagonal variance (0 = disabled). Applied to P before re-Cholesky.
    min_variance: T,
    /// Fading-memory factor γ≥1 applied to the sigma-point covariance (1 = standard).
    gamma: T,
}

impl<T: FloatScalar, const N: usize, const M: usize> SrUkf<T, N, M> {
    /// Create from an existing Cholesky factor `s0` (must be lower-triangular).
    ///
    /// Uses default Merwe parameters: `alpha=1.0`, `beta=2.0`, `kappa=0.0`.
    pub fn new(x0: Vector<T, N>, s0: Matrix<T, N, N>) -> Self {
        Self::with_params(x0, s0, T::one(), T::from(2.0).unwrap(), T::zero())
    }

    /// Create from a covariance matrix `p0` (Cholesky is computed internally).
    ///
    /// Returns `CovarianceNotPD` if `p0` is not positive-definite.
    pub fn from_covariance(x0: Vector<T, N>, p0: Matrix<T, N, N>) -> Result<Self, EstimateError> {
        let chol = cholesky_with_jitter(&p0)?;
        Ok(Self::new(x0, chol.l_full()))
    }

    /// Create from a covariance matrix with custom Merwe scaling parameters.
    pub fn from_covariance_with_params(
        x0: Vector<T, N>,
        p0: Matrix<T, N, N>,
        alpha: T,
        beta: T,
        kappa: T,
    ) -> Result<Self, EstimateError> {
        let chol = cholesky_with_jitter(&p0)?;
        Ok(Self::with_params(x0, chol.l_full(), alpha, beta, kappa))
    }

    /// Create from a Cholesky factor with custom Merwe scaling parameters.
    pub fn with_params(x0: Vector<T, N>, s0: Matrix<T, N, N>, alpha: T, beta: T, kappa: T) -> Self {
        Self {
            x: x0,
            s: s0,
            alpha,
            beta,
            kappa,
            min_variance: T::zero(),
            gamma: T::one(),
        }
    }

    /// Set a minimum diagonal variance floor applied before re-Cholesky in each step.
    pub fn with_min_variance(mut self, min_variance: T) -> Self {
        self.min_variance = min_variance;
        self
    }

    /// Set a fading-memory factor `γ ≥ 1` applied to the sigma-point covariance.
    ///
    /// The predicted covariance becomes `γ · P_sigma + Q`. Default is `1.0`.
    pub fn with_fading_memory(mut self, gamma: T) -> Self {
        self.gamma = gamma;
        self
    }

    /// Reference to the current state estimate.
    #[inline]
    pub fn state(&self) -> &Vector<T, N> {
        &self.x
    }

    /// Reference to the current Cholesky factor `S`.
    #[inline]
    pub fn cholesky_factor(&self) -> &Matrix<T, N, N> {
        &self.s
    }

    /// Reconstruct the full covariance `P = S·Sᵀ`.
    pub fn covariance(&self) -> Matrix<T, N, N> {
        self.s * self.s.transpose()
    }

    /// Merwe weights: `(wm_0, wc_0, w_i)`.
    fn weights(&self) -> (T, T, T) {
        let n = T::from(N).unwrap();
        let lambda = self.alpha * self.alpha * (n + self.kappa) - n;
        let denom = n + lambda;
        let wm_0 = lambda / denom;
        let wc_0 = wm_0 + (T::one() - self.alpha * self.alpha + self.beta);
        let w_i = T::one() / (T::from(2.0).unwrap() * denom);
        (wm_0, wc_0, w_i)
    }

    /// Cholesky factor scaled by `sqrt(N + lambda)` for sigma point generation.
    fn scaled_cholesky(&self) -> Matrix<T, N, N> {
        let n = T::from(N).unwrap();
        let lambda = self.alpha * self.alpha * (n + self.kappa) - n;
        self.s * (n + lambda).sqrt()
    }

    /// Generate the `2N+1` sigma points `[x, x ± scaled_S[:,i]]` from the current state.
    fn sigma_points(&self, scaled_s: &Matrix<T, N, N>) -> Vec<Vector<T, N>> {
        let mut sigmas: Vec<Vector<T, N>> = Vec::with_capacity(2 * N + 1);
        sigmas.push(self.x);
        for i in 0..N {
            let mut col = Vector::<T, N>::zeros();
            for r in 0..N {
                col[r] = scaled_s[(r, i)];
            }
            sigmas.push(self.x + col);
            sigmas.push(self.x - col);
        }
        sigmas
    }

    /// Predict step.
    ///
    /// - `f` — state transition: `x_{k+1} = f(x_k)`
    /// - `q` — process noise covariance (pass `None` for zero process noise)
    pub fn predict(
        &mut self,
        f: impl Fn(&Vector<T, N>) -> Vector<T, N>,
        q: Option<&Matrix<T, N, N>>,
    ) -> Result<(), EstimateError> {
        let (wm_0, wc_0, w_i) = self.weights();
        let scaled_s = self.scaled_cholesky();

        // Generate sigma points and propagate in place
        let mut sigmas = self.sigma_points(&scaled_s);
        for s in sigmas.iter_mut() {
            *s = f(s);
        }

        // Weighted mean
        let mut x_mean = Vector::<T, N>::zeros();
        for r in 0..N {
            x_mean[r] = wm_0 * sigmas[0][r];
        }
        for i in 0..N {
            for r in 0..N {
                x_mean[r] = x_mean[r] + w_i * (sigmas[2 * i + 1][r] + sigmas[2 * i + 2][r]);
            }
        }

        // Sigma-point covariance (without Q)
        let mut p_new = Matrix::<T, N, N>::zeros();

        // Non-central sigma points
        for i in 0..N {
            let dp = sigmas[2 * i + 1] - x_mean;
            let dm = sigmas[2 * i + 2] - x_mean;
            for r in 0..N {
                for c in 0..N {
                    p_new[(r, c)] = p_new[(r, c)] + w_i * (dp[r] * dp[c] + dm[r] * dm[c]);
                }
            }
        }

        // Central sigma point
        let d0 = sigmas[0] - x_mean;
        for r in 0..N {
            for c in 0..N {
                p_new[(r, c)] = p_new[(r, c)] + wc_0 * d0[r] * d0[c];
            }
        }

        // Fading memory: scale sigma-point covariance by γ before adding Q.
        p_new *= self.gamma;
        if let Some(q) = q {
            p_new += *q;
        }
        let half = T::from(0.5).unwrap();
        p_new = (p_new + p_new.transpose()) * half;
        apply_var_floor(&mut p_new, self.min_variance);

        // Re-Cholesky (with jitter fallback).
        let chol = cholesky_with_jitter(&p_new)?;

        self.x = x_mean;
        self.s = chol.l_full();

        Ok(())
    }

    /// Update step.
    ///
    /// - `z` — measurement vector
    /// - `h` — measurement model: `z = h(x)`
    /// - `r` — measurement noise covariance
    ///
    /// Covariance update uses `P - K·S·Kᵀ` (symmetric, manifestly PSD-subtracted),
    /// then re-Choleskyizes to maintain the square-root factor.
    ///
    /// Returns the Normalized Innovation Squared (NIS): `yᵀ S⁻¹ y`.
    pub fn update(
        &mut self,
        z: &Vector<T, M>,
        h: impl Fn(&Vector<T, N>) -> Vector<T, M>,
        r: &Matrix<T, M, M>,
    ) -> Result<T, EstimateError> {
        let (s_mat, s_inv, pxz, innovation, nis) = self.measurement_transform(z, &h, r)?;
        self.apply_update(&s_mat, &s_inv, &pxz, &innovation)?;
        Ok(nis)
    }

    /// Sigma-point measurement transform shared by `update` and `update_gated`.
    ///
    /// Returns `(S, S⁻¹, Pxz, innovation, NIS)` without modifying the state.
    #[allow(clippy::type_complexity)]
    fn measurement_transform(
        &self,
        z: &Vector<T, M>,
        h: &impl Fn(&Vector<T, N>) -> Vector<T, M>,
        r: &Matrix<T, M, M>,
    ) -> Result<
        (
            Matrix<T, M, M>,
            Matrix<T, M, M>,
            Matrix<T, N, M>,
            Vector<T, M>,
            T,
        ),
        EstimateError,
    > {
        let (wm_0, wc_0, w_i) = self.weights();
        let scaled_s = self.scaled_cholesky();

        // Generate state sigma points and transform through the measurement model
        let sigmas_x = self.sigma_points(&scaled_s);
        let mut sigmas_z: Vec<Vector<T, M>> = Vec::with_capacity(2 * N + 1);
        for sx in &sigmas_x {
            sigmas_z.push(h(sx));
        }

        // Measurement mean
        let mut z_mean = Vector::<T, M>::zeros();
        for r in 0..M {
            z_mean[r] = wm_0 * sigmas_z[0][r];
        }
        for i in 0..N {
            for r in 0..M {
                z_mean[r] = z_mean[r] + w_i * (sigmas_z[2 * i + 1][r] + sigmas_z[2 * i + 2][r]);
            }
        }

        // Innovation covariance Pzz + R
        let mut pzz = Matrix::<T, M, M>::zeros();
        let dz0 = sigmas_z[0] - z_mean;
        for ri in 0..M {
            for ci in 0..M {
                pzz[(ri, ci)] = pzz[(ri, ci)] + wc_0 * dz0[ri] * dz0[ci];
            }
        }
        for i in 0..N {
            let dzp = sigmas_z[2 * i + 1] - z_mean;
            let dzm = sigmas_z[2 * i + 2] - z_mean;
            for ri in 0..M {
                for ci in 0..M {
                    pzz[(ri, ci)] = pzz[(ri, ci)] + w_i * (dzp[ri] * dzp[ci] + dzm[ri] * dzm[ci]);
                }
            }
        }
        let s_mat = pzz + *r; // full innovation covariance

        // Cross-covariance Pxz
        let mut pxz = Matrix::<T, N, M>::zeros();
        let dx0 = sigmas_x[0] - self.x;
        for ri in 0..N {
            for ci in 0..M {
                pxz[(ri, ci)] = pxz[(ri, ci)] + wc_0 * dx0[ri] * dz0[ci];
            }
        }
        for i in 0..N {
            let dxp = sigmas_x[2 * i + 1] - self.x;
            let dxm = sigmas_x[2 * i + 2] - self.x;
            let dzp = sigmas_z[2 * i + 1] - z_mean;
            let dzm = sigmas_z[2 * i + 2] - z_mean;
            for ri in 0..N {
                for ci in 0..M {
                    pxz[(ri, ci)] = pxz[(ri, ci)] + w_i * (dxp[ri] * dzp[ci] + dxm[ri] * dzm[ci]);
                }
            }
        }

        let s_inv = cholesky_with_jitter(&s_mat)
            .map_err(|_| EstimateError::SingularInnovation)?
            .inverse();
        let innovation = *z - z_mean;
        let nis = (innovation.transpose() * s_inv * innovation)[(0, 0)];

        Ok((s_mat, s_inv, pxz, innovation, nis))
    }

    /// Apply the Kalman gain, covariance update, and re-Cholesky from a measurement transform.
    fn apply_update(
        &mut self,
        s_mat: &Matrix<T, M, M>,
        s_inv: &Matrix<T, M, M>,
        pxz: &Matrix<T, N, M>,
        innovation: &Vector<T, M>,
    ) -> Result<(), EstimateError> {
        // Kalman gain: K = Pxz · S⁻¹
        let k = *pxz * *s_inv;

        // State update
        self.x += k * *innovation;

        // Covariance update: P_new = P - K·S·Kᵀ, then re-Cholesky.
        let mut p_new = self.covariance() - k * *s_mat * k.transpose();
        let half = T::from(0.5).unwrap();
        p_new = (p_new + p_new.transpose()) * half;
        apply_var_floor(&mut p_new, self.min_variance);
        let chol = cholesky_with_jitter(&p_new)?;
        self.s = chol.l_full();

        Ok(())
    }

    /// Update with innovation gating — skips state update if NIS exceeds `gate`.
    ///
    /// Returns `Ok(None)` when rejected, `Ok(Some(nis))` when accepted.
    ///
    /// Chi-squared thresholds: M=1 → 99%: 6.63 | M=2 → 9.21 | M=3 → 11.34
    pub fn update_gated(
        &mut self,
        z: &Vector<T, M>,
        h: impl Fn(&Vector<T, N>) -> Vector<T, M>,
        r: &Matrix<T, M, M>,
        gate: T,
    ) -> Result<Option<T>, EstimateError> {
        let (s_mat, s_inv, pxz, innovation, nis) = self.measurement_transform(z, &h, r)?;
        if nis > gate {
            return Ok(None);
        }
        self.apply_update(&s_mat, &s_inv, &pxz, &innovation)?;
        Ok(Some(nis))
    }
}
