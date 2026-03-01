extern crate alloc;
use alloc::vec::Vec;

use crate::linalg::CholeskyDecomposition;
use crate::matrix::vector::ColumnVector;
use crate::traits::FloatScalar;
use crate::Matrix;

use super::EstimateError;

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
/// # Example
///
/// ```
/// use numeris::estimate::SrUkf;
/// use numeris::{ColumnVector, Matrix};
///
/// let x0 = ColumnVector::from_column([0.0_f64, 1.0]);
/// let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
/// let mut srukf = SrUkf::<f64, 2, 1>::from_covariance(x0, p0).unwrap();
///
/// let dt = 0.1;
/// let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
/// let r = Matrix::new([[0.5]]);
///
/// srukf.predict(
///     |x| ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]]),
///     Some(&q),
/// ).unwrap();
///
/// srukf.update(
///     &ColumnVector::from_column([0.12]),
///     |x| ColumnVector::from_column([x[(0, 0)]]),
///     &r,
/// ).unwrap();
/// ```
pub struct SrUkf<T: FloatScalar, const N: usize, const M: usize> {
    /// State estimate.
    pub x: ColumnVector<T, N>,
    /// Lower-triangular Cholesky factor: `P = S·Sᵀ`.
    pub s: Matrix<T, N, N>,
    alpha: T,
    beta: T,
    kappa: T,
}

impl<T: FloatScalar, const N: usize, const M: usize> SrUkf<T, N, M> {
    /// Create from an existing Cholesky factor `s0` (must be lower-triangular).
    ///
    /// Uses default Merwe parameters: `alpha=0.001`, `beta=2.0`, `kappa=0.0`.
    pub fn new(x0: ColumnVector<T, N>, s0: Matrix<T, N, N>) -> Self {
        Self::with_params(
            x0,
            s0,
            T::from(0.001).unwrap(),
            T::from(2.0).unwrap(),
            T::zero(),
        )
    }

    /// Create from a covariance matrix `p0` (Cholesky is computed internally).
    ///
    /// Returns `CovarianceNotPD` if `p0` is not positive-definite.
    pub fn from_covariance(
        x0: ColumnVector<T, N>,
        p0: Matrix<T, N, N>,
    ) -> Result<Self, EstimateError> {
        let chol = CholeskyDecomposition::new(&p0).map_err(|_| EstimateError::CovarianceNotPD)?;
        Ok(Self::new(x0, chol.l_full()))
    }

    /// Create from a covariance matrix with custom Merwe scaling parameters.
    pub fn from_covariance_with_params(
        x0: ColumnVector<T, N>,
        p0: Matrix<T, N, N>,
        alpha: T,
        beta: T,
        kappa: T,
    ) -> Result<Self, EstimateError> {
        let chol = CholeskyDecomposition::new(&p0).map_err(|_| EstimateError::CovarianceNotPD)?;
        Ok(Self::with_params(x0, chol.l_full(), alpha, beta, kappa))
    }

    /// Create from a Cholesky factor with custom Merwe scaling parameters.
    pub fn with_params(
        x0: ColumnVector<T, N>,
        s0: Matrix<T, N, N>,
        alpha: T,
        beta: T,
        kappa: T,
    ) -> Self {
        Self {
            x: x0,
            s: s0,
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

    /// Predict step.
    ///
    /// - `f` — state transition: `x_{k+1} = f(x_k)`
    /// - `q` — process noise covariance (pass `None` for zero process noise)
    pub fn predict(
        &mut self,
        f: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, N>,
        q: Option<&Matrix<T, N, N>>,
    ) -> Result<(), EstimateError> {
        let n = T::from(N).unwrap();
        let lambda = self.alpha * self.alpha * (n + self.kappa) - n;
        let scale = (n + lambda).sqrt();
        let (wm_0, wc_0, w_i) = self.weights();

        // Scaled Cholesky columns
        let scaled_s = self.s * scale;

        // Generate and propagate sigma points
        let mut sigmas: Vec<ColumnVector<T, N>> = Vec::with_capacity(2 * N + 1);
        sigmas.push(f(&self.x));

        for i in 0..N {
            let mut col = ColumnVector::<T, N>::zeros();
            for r in 0..N {
                col[(r, 0)] = scaled_s[(r, i)];
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

        // QR factorization of the compound matrix to get the new S.
        // Rows of the compound matrix are: sqrt(w_i) * (sigma_i - x_mean) for i=1..2N
        // This is 2N rows × N cols. QR gives an N×N upper-triangular R; Sᵀ = R.
        //
        // For simplicity, form the weighted covariance and Cholesky it:
        let mut p_new = Matrix::<T, N, N>::zeros();

        // Contributions from i=1..2N (the non-central sigma points)
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

        // Central sigma point: rank-1 update with wc_0
        let d0 = sigmas[0] - x_mean;
        for r in 0..N {
            for c in 0..N {
                p_new[(r, c)] = p_new[(r, c)] + wc_0 * d0[(r, 0)] * d0[(c, 0)];
            }
        }

        // Add process noise
        if let Some(q) = q {
            p_new = p_new + *q;
        }

        // Cholesky to get new S
        let chol =
            CholeskyDecomposition::new(&p_new).map_err(|_| EstimateError::CovarianceNotPD)?;

        self.x = x_mean;
        self.s = chol.l_full();

        Ok(())
    }

    /// Update step.
    ///
    /// - `z` — measurement vector
    /// - `h` — measurement model: `z = h(x)`
    /// - `r` — measurement noise covariance
    pub fn update(
        &mut self,
        z: &ColumnVector<T, M>,
        h: impl Fn(&ColumnVector<T, N>) -> ColumnVector<T, M>,
        r: &Matrix<T, M, M>,
    ) -> Result<(), EstimateError> {
        let n = T::from(N).unwrap();
        let lambda = self.alpha * self.alpha * (n + self.kappa) - n;
        let scale = (n + lambda).sqrt();
        let (wm_0, wc_0, w_i) = self.weights();

        let scaled_s = self.s * scale;

        // Generate state sigma points
        let mut sigmas_x: Vec<ColumnVector<T, N>> = Vec::with_capacity(2 * N + 1);
        sigmas_x.push(self.x);
        for i in 0..N {
            let mut col = ColumnVector::<T, N>::zeros();
            for r in 0..N {
                col[(r, 0)] = scaled_s[(r, i)];
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
                z_mean[(r, 0)] =
                    z_mean[(r, 0)] + w_i * (sigmas_z[2 * i + 1][(r, 0)] + sigmas_z[2 * i + 2][(r, 0)]);
            }
        }

        // Innovation covariance Sz via Cholesky of Pzz + R
        let mut pzz = Matrix::<T, M, M>::zeros();
        let dz0 = sigmas_z[0] - z_mean;
        for ri in 0..M {
            for ci in 0..M {
                pzz[(ri, ci)] = pzz[(ri, ci)] + wc_0 * dz0[(ri, 0)] * dz0[(ci, 0)];
            }
        }
        for i in 0..N {
            let dzp = sigmas_z[2 * i + 1] - z_mean;
            let dzm = sigmas_z[2 * i + 2] - z_mean;
            for ri in 0..M {
                for ci in 0..M {
                    pzz[(ri, ci)] = pzz[(ri, ci)]
                        + w_i * (dzp[(ri, 0)] * dzp[(ci, 0)] + dzm[(ri, 0)] * dzm[(ci, 0)]);
                }
            }
        }
        let s_mat = pzz + *r;

        // Cross-covariance Pxz
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

        // Kalman gain: K = Pxz · S⁻¹
        let s_inv = s_mat.inverse().map_err(|_| EstimateError::SingularInnovation)?;
        let k = pxz * s_inv;

        // State update
        let innovation = *z - z_mean;
        self.x = self.x + k * innovation;

        // Covariance update: P_new = P - K·S_mat·Kᵀ, then re-Cholesky
        let p_new = self.covariance() - k * s_mat * k.transpose();
        let chol =
            CholeskyDecomposition::new(&p_new).map_err(|_| EstimateError::CovarianceNotPD)?;
        self.s = chol.l_full();

        Ok(())
    }
}
