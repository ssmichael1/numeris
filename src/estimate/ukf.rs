extern crate alloc;
use alloc::vec::Vec;

use crate::linalg::CholeskyDecomposition;
use crate::matrix::vector::ColumnVector;
use crate::traits::FloatScalar;
use crate::Matrix;

use super::EstimateError;

/// Unscented Kalman Filter with Merwe-scaled sigma points.
///
/// `N` is the state dimension, `M` is the measurement dimension.
/// Uses `2N+1` sigma points to capture mean and covariance through
/// nonlinear transformations without requiring Jacobians.
///
/// Requires the `alloc` feature for temporary sigma point storage
/// in `predict` and `update`.
///
/// # Example
///
/// ```
/// use numeris::estimate::Ukf;
/// use numeris::{ColumnVector, Matrix};
///
/// let x0 = ColumnVector::from_column([0.0_f64, 1.0]);
/// let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
/// let mut ukf = Ukf::<f64, 2, 1>::new(x0, p0);
///
/// let dt = 0.1;
/// let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
/// let r = Matrix::new([[0.5]]);
///
/// // Predict
/// ukf.predict(
///     |x| ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]]),
///     Some(&q),
/// ).unwrap();
///
/// // Update
/// ukf.update(
///     &ColumnVector::from_column([0.12]),
///     |x| ColumnVector::from_column([x[(0, 0)]]),
///     &r,
/// ).unwrap();
/// ```
pub struct Ukf<T: FloatScalar, const N: usize, const M: usize> {
    /// State estimate.
    pub x: ColumnVector<T, N>,
    /// State covariance.
    pub p: Matrix<T, N, N>,
    alpha: T,
    beta: T,
    kappa: T,
}

impl<T: FloatScalar, const N: usize, const M: usize> Ukf<T, N, M> {
    /// Create a new UKF with default Merwe parameters: `alpha=0.001`, `beta=2.0`, `kappa=0.0`.
    pub fn new(x0: ColumnVector<T, N>, p0: Matrix<T, N, N>) -> Self {
        Self::with_params(
            x0,
            p0,
            T::from(0.001).unwrap(),
            T::from(2.0).unwrap(),
            T::zero(),
        )
    }

    /// Create a new UKF with custom scaling parameters.
    ///
    /// - `alpha` — spread of sigma points around mean (typically 1e-4 to 1)
    /// - `beta` — prior distribution knowledge (2.0 optimal for Gaussian)
    /// - `kappa` — secondary scaling (typically 0 or 3-N)
    pub fn with_params(
        x0: ColumnVector<T, N>,
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
        }
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

    /// Compute Merwe weights: `(wm, wc)` where `wm[0]` and `wc[0]` are special.
    ///
    /// Returns `(wm_0, wc_0, w_i)` where `w_i = wm_i = wc_i` for i > 0.
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

    /// Generate sigma points and Cholesky factor.
    ///
    /// Returns (L, scale) where L is the Cholesky factor and scale = sqrt(N + lambda).
    fn sigma_cholesky(&self) -> Result<(Matrix<T, N, N>, T), EstimateError> {
        let n = T::from(N).unwrap();
        let lambda = self.lambda();
        let scale = (n + lambda).sqrt();

        // Compute Cholesky of P
        let chol =
            CholeskyDecomposition::new(&self.p).map_err(|_| EstimateError::CovarianceNotPD)?;
        let l = chol.l_full();

        // Scale L by sqrt(N + lambda)
        let scaled_l = l * scale;

        Ok((scaled_l, scale))
    }

    /// Predict step.
    ///
    /// - `f` — state transition: `x_{k+1} = f(x_k)`
    /// - `q` — process noise covariance (pass `None` for zero process noise)
    ///
    /// Generates `2N+1` sigma points, propagates through `f`, and
    /// reconstructs the predicted mean and covariance.
    pub fn predict(
        &mut self,
        f: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, N>,
        q: Option<&Matrix<T, N, N>>,
    ) -> Result<(), EstimateError> {
        let (scaled_l, _) = self.sigma_cholesky()?;
        let (wm_0, wc_0, w_i) = self.weights();

        // Propagate sigma points and store
        let mut sigmas: Vec<ColumnVector<T, N>> = Vec::with_capacity(2 * N + 1);

        // Central point
        sigmas.push(f(&self.x));

        // Positive and negative perturbations
        for i in 0..N {
            let mut col = ColumnVector::<T, N>::zeros();
            for r in 0..N {
                col[(r, 0)] = scaled_l[(r, i)];
            }
            sigmas.push(f(&(self.x + col)));
            sigmas.push(f(&(self.x - col)));
        }

        // Weighted mean
        let mut x_mean = ColumnVector::<T, N>::zeros();
        for r in 0..N {
            x_mean[(r, 0)] = wm_0 * sigmas[0][(r, 0)];
        }
        for i in 0..N {
            for r in 0..N {
                x_mean[(r, 0)] =
                    x_mean[(r, 0)] + w_i * (sigmas[2 * i + 1][(r, 0)] + sigmas[2 * i + 2][(r, 0)]);
            }
        }

        // Weighted covariance
        let mut p_new = Matrix::<T, N, N>::zeros();
        // Central point contribution
        let d0 = sigmas[0] - x_mean;
        for r in 0..N {
            for c in 0..N {
                p_new[(r, c)] = p_new[(r, c)] + wc_0 * d0[(r, 0)] * d0[(c, 0)];
            }
        }
        // Remaining sigma points
        for i in 0..N {
            let dp = sigmas[2 * i + 1] - x_mean;
            let dm = sigmas[2 * i + 2] - x_mean;
            for r in 0..N {
                for c in 0..N {
                    p_new[(r, c)] = p_new[(r, c)]
                        + w_i * (dp[(r, 0)] * dp[(c, 0)] + dm[(r, 0)] * dm[(c, 0)]);
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
    /// Generates sigma points from the predicted state, transforms through `h`,
    /// and computes the Kalman gain via the cross-covariance and innovation covariance.
    pub fn update(
        &mut self,
        z: &ColumnVector<T, M>,
        h: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, M>,
        r: &Matrix<T, M, M>,
    ) -> Result<(), EstimateError> {
        let (scaled_l, _) = self.sigma_cholesky()?;
        let (wm_0, wc_0, w_i) = self.weights();

        // Generate state sigma points
        let mut sigmas_x: Vec<ColumnVector<T, N>> = Vec::with_capacity(2 * N + 1);
        sigmas_x.push(self.x);
        for i in 0..N {
            let mut col = ColumnVector::<T, N>::zeros();
            for r in 0..N {
                col[(r, 0)] = scaled_l[(r, i)];
            }
            sigmas_x.push(self.x + col);
            sigmas_x.push(self.x - col);
        }

        // Transform through measurement model
        let mut sigmas_z: Vec<ColumnVector<T, M>> = Vec::with_capacity(2 * N + 1);
        for sx in &sigmas_x {
            sigmas_z.push(h(sx));
        }

        // Measurement mean
        let mut z_mean = ColumnVector::<T, M>::zeros();
        for r in 0..M {
            z_mean[(r, 0)] = wm_0 * sigmas_z[0][(r, 0)];
        }
        for i in 0..N {
            for r in 0..M {
                z_mean[(r, 0)] = z_mean[(r, 0)]
                    + w_i * (sigmas_z[2 * i + 1][(r, 0)] + sigmas_z[2 * i + 2][(r, 0)]);
            }
        }

        // Innovation covariance S = Σ wc_i (z_i - z̄)(z_i - z̄)ᵀ + R
        let mut s = Matrix::<T, M, M>::zeros();
        let dz0 = sigmas_z[0] - z_mean;
        for r in 0..M {
            for c in 0..M {
                s[(r, c)] = s[(r, c)] + wc_0 * dz0[(r, 0)] * dz0[(c, 0)];
            }
        }
        for i in 0..N {
            let dzp = sigmas_z[2 * i + 1] - z_mean;
            let dzm = sigmas_z[2 * i + 2] - z_mean;
            for r in 0..M {
                for c in 0..M {
                    s[(r, c)] = s[(r, c)]
                        + w_i * (dzp[(r, 0)] * dzp[(c, 0)] + dzm[(r, 0)] * dzm[(c, 0)]);
                }
            }
        }
        s = s + *r;

        // Cross-covariance Pxz = Σ wc_i (x_i - x̄)(z_i - z̄)ᵀ
        let mut pxz = Matrix::<T, N, M>::zeros();
        let dx0 = sigmas_x[0] - self.x;
        for ri in 0..N {
            for ci in 0..M {
                pxz[(ri, ci)] = pxz[(ri, ci)] + wc_0 * dx0[(ri, 0)] * dz0[(ci, 0)];
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
                        + w_i * (dxp[(ri, 0)] * dzp[(ci, 0)] + dxm[(ri, 0)] * dzm[(ci, 0)]);
                }
            }
        }

        // Kalman gain K = Pxz S⁻¹
        let s_inv = s.inverse().map_err(|_| EstimateError::SingularInnovation)?;
        let k = pxz * s_inv;

        // Update state and covariance
        let innovation = *z - z_mean;
        self.x = self.x + k * innovation;
        self.p = self.p - k * s * k.transpose();

        Ok(())
    }
}
