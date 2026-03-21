extern crate alloc;
use alloc::vec::Vec;

use crate::matrix::vector::Vector;
use crate::traits::FloatScalar;
use crate::Matrix;

use super::{apply_var_floor, cholesky_with_jitter, EstimateError};

/// Unscented Kalman Filter with Merwe-scaled sigma points.
///
/// `N` is the state dimension, `M` is the measurement dimension.
/// Uses `2N+1` sigma points to capture mean and covariance through
/// nonlinear transformations without requiring Jacobians.
///
/// Requires the `alloc` feature for temporary sigma point storage
/// in `predict` and `update`.
///
/// # Default parameters
///
/// `alpha=1.0`, `beta=2.0`, `kappa=0.0` — all sigma-point weights are non-negative,
/// sigma points are placed at distance `√N · L[:,i]` from the mean.
/// For tight sigma point clustering, reduce `alpha` toward 0.1.
///
/// # Example
///
/// ```
/// use numeris::estimate::Ukf;
/// use numeris::{Vector, Matrix};
///
/// let x0 = Vector::from_array([0.0_f64, 1.0]);
/// let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
/// let mut ukf = Ukf::<f64, 2, 1>::new(x0, p0);
///
/// let dt = 0.1;
/// let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
/// let r = Matrix::new([[0.5]]);
///
/// // Predict
/// ukf.predict(
///     |x| Vector::from_array([x[0] + dt * x[1], x[1]]),
///     Some(&q),
/// ).unwrap();
///
/// // Update
/// ukf.update(
///     &Vector::from_array([0.12]),
///     |x| Vector::from_array([x[0]]),
///     &r,
/// ).unwrap();
/// ```
pub struct Ukf<T: FloatScalar, const N: usize, const M: usize> {
    /// State estimate.
    pub x: Vector<T, N>,
    /// State covariance.
    pub p: Matrix<T, N, N>,
    alpha: T,
    beta: T,
    kappa: T,
    /// Minimum allowed diagonal variance (0 = disabled).
    min_variance: T,
    /// Fading-memory factor γ≥1 applied to predicted covariance (1 = standard).
    gamma: T,
}

impl<T: FloatScalar, const N: usize, const M: usize> Ukf<T, N, M> {
    /// Create a new UKF with default Merwe parameters: `alpha=1.0`, `beta=2.0`, `kappa=0.0`.
    ///
    /// These defaults yield non-negative sigma-point weights for all state dimensions.
    /// For tighter sigma point spread, use `with_params` to reduce `alpha`.
    pub fn new(x0: Vector<T, N>, p0: Matrix<T, N, N>) -> Self {
        Self::with_params(
            x0,
            p0,
            T::one(),
            T::from(2.0).unwrap(),
            T::zero(),
        )
    }

    /// Create a new UKF with custom scaling parameters.
    ///
    /// - `alpha` — spread of sigma points around mean (1.0 gives non-negative weights;
    ///   values < ~0.52 give negative central weight `wc_0`)
    /// - `beta` — prior distribution knowledge (2.0 optimal for Gaussian)
    /// - `kappa` — secondary scaling (0 or 3-N)
    pub fn with_params(
        x0: Vector<T, N>,
        p0: Matrix<T, N, N>,
        alpha: T,
        beta: T,
        kappa: T,
    ) -> Self {
        Self {
            x: x0,
            p: p0,
            alpha,
            beta,
            kappa,
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
    /// The predicted covariance becomes `γ · P_sigma + Q` where `P_sigma` is
    /// the sigma-point weighted covariance (without Q). Default is `1.0`.
    pub fn with_fading_memory(mut self, gamma: T) -> Self {
        self.gamma = gamma;
        self
    }

    /// Reference to the current state estimate.
    #[inline]
    pub fn state(&self) -> &Vector<T, N> {
        &self.x
    }

    /// Reference to the current state covariance.
    #[inline]
    pub fn covariance(&self) -> &Matrix<T, N, N> {
        &self.p
    }

    /// Compute Merwe weights: `(wm_0, wc_0, w_i)` where `w_i = wm_i = wc_i` for i > 0.
    fn weights(&self) -> (T, T, T) {
        let n = T::from(N).unwrap();
        let lambda = self.alpha * self.alpha * (n + self.kappa) - n;
        let denom = n + lambda;
        let wm_0 = lambda / denom;
        let wc_0 = wm_0 + (T::one() - self.alpha * self.alpha + self.beta);
        let w_i = T::one() / (T::from(2.0).unwrap() * denom);
        (wm_0, wc_0, w_i)
    }

    /// Compute lambda = alpha^2 * (N + kappa) - N.
    fn lambda(&self) -> T {
        let n = T::from(N).unwrap();
        self.alpha * self.alpha * (n + self.kappa) - n
    }

    /// Generate scaled Cholesky factor for sigma point generation.
    ///
    /// Returns `scaled_L = sqrt(N + lambda) · L` where `L = chol(P)`.
    fn sigma_cholesky(&self) -> Result<Matrix<T, N, N>, EstimateError> {
        let n = T::from(N).unwrap();
        let lambda = self.lambda();
        let scale = (n + lambda).sqrt();
        let chol = cholesky_with_jitter(&self.p)?;
        Ok(chol.l_full() * scale)
    }

    /// Predict step.
    ///
    /// - `f` — state transition: `x_{k+1} = f(x_k)`
    /// - `q` — process noise covariance (pass `None` for zero process noise)
    ///
    /// Generates `2N+1` sigma points, propagates through `f`, and
    /// reconstructs the predicted mean and covariance as `γ · P_sigma + Q`.
    pub fn predict(
        &mut self,
        f: impl Fn(&Vector<T, N>) -> Vector<T, N>,
        q: Option<&Matrix<T, N, N>>,
    ) -> Result<(), EstimateError> {
        let scaled_l = self.sigma_cholesky()?;
        let (wm_0, wc_0, w_i) = self.weights();

        // Propagate sigma points and store
        let mut sigmas: Vec<Vector<T, N>> = Vec::with_capacity(2 * N + 1);

        // Central point
        sigmas.push(f(&self.x));

        // Positive and negative perturbations
        for i in 0..N {
            let mut col = Vector::<T, N>::zeros();
            for r in 0..N {
                col[r] = scaled_l[(r, i)];
            }
            sigmas.push(f(&(self.x + col)));
            sigmas.push(f(&(self.x - col)));
        }

        // Weighted mean
        let mut x_mean = Vector::<T, N>::zeros();
        for r in 0..N {
            x_mean[r] = wm_0 * sigmas[0][r];
        }
        for i in 0..N {
            for r in 0..N {
                x_mean[r] =
                    x_mean[r] + w_i * (sigmas[2 * i + 1][r] + sigmas[2 * i + 2][r]);
            }
        }

        // Weighted covariance (sigma-point part, without Q)
        let mut p_new = Matrix::<T, N, N>::zeros();
        let d0 = sigmas[0] - x_mean;
        for r in 0..N {
            for c in 0..N {
                p_new[(r, c)] = p_new[(r, c)] + wc_0 * d0[r] * d0[c];
            }
        }
        for i in 0..N {
            let dp = sigmas[2 * i + 1] - x_mean;
            let dm = sigmas[2 * i + 2] - x_mean;
            for r in 0..N {
                for c in 0..N {
                    p_new[(r, c)] = p_new[(r, c)]
                        + w_i * (dp[r] * dp[c] + dm[r] * dm[c]);
                }
            }
        }

        // Fading memory: scale sigma-point covariance by γ before adding Q.
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
    /// Generates sigma points from the predicted state, transforms through `h`,
    /// and computes the Kalman gain via the cross-covariance and innovation covariance.
    ///
    /// Covariance update uses the symmetric `P - K S Kᵀ` form (equivalent to
    /// `P - K Pxzᵀ` but manifestly PSD-subtracted).
    ///
    /// Returns the Normalized Innovation Squared (NIS): `yᵀ S⁻¹ y`.
    pub fn update(
        &mut self,
        z: &Vector<T, M>,
        h: impl Fn(&Vector<T, N>) -> Vector<T, M>,
        r: &Matrix<T, M, M>,
    ) -> Result<T, EstimateError> {
        let scaled_l = self.sigma_cholesky()?;
        let (wm_0, wc_0, w_i) = self.weights();

        // Generate state sigma points
        let mut sigmas_x: Vec<Vector<T, N>> = Vec::with_capacity(2 * N + 1);
        sigmas_x.push(self.x);
        for i in 0..N {
            let mut col = Vector::<T, N>::zeros();
            for r in 0..N {
                col[r] = scaled_l[(r, i)];
            }
            sigmas_x.push(self.x + col);
            sigmas_x.push(self.x - col);
        }

        // Transform through measurement model
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
                z_mean[r] = z_mean[r]
                    + w_i * (sigmas_z[2 * i + 1][r] + sigmas_z[2 * i + 2][r]);
            }
        }

        // Innovation covariance S = Σ wc_i (z_i - z̄)(z_i - z̄)ᵀ + R
        let mut s = Matrix::<T, M, M>::zeros();
        let dz0 = sigmas_z[0] - z_mean;
        for r in 0..M {
            for c in 0..M {
                s[(r, c)] = s[(r, c)] + wc_0 * dz0[r] * dz0[c];
            }
        }
        for i in 0..N {
            let dzp = sigmas_z[2 * i + 1] - z_mean;
            let dzm = sigmas_z[2 * i + 2] - z_mean;
            for r in 0..M {
                for c in 0..M {
                    s[(r, c)] = s[(r, c)]
                        + w_i * (dzp[r] * dzp[c] + dzm[r] * dzm[c]);
                }
            }
        }
        let s_mat = s + *r; // full innovation covariance

        // Cross-covariance Pxz = Σ wc_i (x_i - x̄)(z_i - z̄)ᵀ
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
                    pxz[(ri, ci)] = pxz[(ri, ci)]
                        + w_i * (dxp[ri] * dzp[ci] + dxm[ri] * dzm[ci]);
                }
            }
        }

        // Kalman gain K = Pxz S⁻¹
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
        z: &Vector<T, M>,
        h: impl Fn(&Vector<T, N>) -> Vector<T, M>,
        r: &Matrix<T, M, M>,
        gate: T,
    ) -> Result<Option<T>, EstimateError> {
        // Compute NIS without modifying state by running the full sigma-point transform.
        let scaled_l = self.sigma_cholesky()?;
        let (wm_0, wc_0, w_i) = self.weights();

        let mut sigmas_x: Vec<Vector<T, N>> = Vec::with_capacity(2 * N + 1);
        sigmas_x.push(self.x);
        for i in 0..N {
            let mut col = Vector::<T, N>::zeros();
            for r_i in 0..N {
                col[r_i] = scaled_l[(r_i, i)];
            }
            sigmas_x.push(self.x + col);
            sigmas_x.push(self.x - col);
        }

        let mut sigmas_z: Vec<Vector<T, M>> = Vec::with_capacity(2 * N + 1);
        for sx in &sigmas_x {
            sigmas_z.push(h(sx));
        }

        let mut z_mean = Vector::<T, M>::zeros();
        for r_i in 0..M {
            z_mean[r_i] = wm_0 * sigmas_z[0][r_i];
        }
        for i in 0..N {
            for r_i in 0..M {
                z_mean[r_i] = z_mean[r_i]
                    + w_i * (sigmas_z[2 * i + 1][r_i] + sigmas_z[2 * i + 2][r_i]);
            }
        }

        let mut s_mat = Matrix::<T, M, M>::zeros();
        let dz0 = sigmas_z[0] - z_mean;
        for r_i in 0..M {
            for ci in 0..M {
                s_mat[(r_i, ci)] = s_mat[(r_i, ci)] + wc_0 * dz0[r_i] * dz0[ci];
            }
        }
        for i in 0..N {
            let dzp = sigmas_z[2 * i + 1] - z_mean;
            let dzm = sigmas_z[2 * i + 2] - z_mean;
            for r_i in 0..M {
                for ci in 0..M {
                    s_mat[(r_i, ci)] = s_mat[(r_i, ci)]
                        + w_i * (dzp[r_i] * dzp[ci] + dzm[r_i] * dzm[ci]);
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
