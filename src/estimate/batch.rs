use crate::matrix::vector::Vector;
use crate::traits::FloatScalar;
use crate::Matrix;

use super::EstimateError;

/// Linear batch least-squares estimator via information accumulation.
///
/// Accumulates the information matrix `Λ = Σ HᵀR⁻¹H` and information
/// vector `η = Σ HᵀR⁻¹z` from multiple observations, then solves for
/// the state estimate `x = Λ⁻¹η` and covariance `P = Λ⁻¹`.
///
/// Fully no-std compatible — no heap allocation required.
/// The measurement dimension `M` is a method-level const generic,
/// allowing observations of different sizes to be accumulated.
///
/// # Example
///
/// ```
/// use numeris::estimate::BatchLsq;
/// use numeris::{Vector, Matrix};
///
/// let mut lsq = BatchLsq::<f64, 1>::new();
///
/// // Accumulate scalar observations
/// let h = Matrix::new([[1.0_f64]]);
/// let r = Matrix::new([[0.1]]);
/// lsq.add_observation(&Vector::from_array([1.05]), &h, &r).unwrap();
/// lsq.add_observation(&Vector::from_array([0.95]), &h, &r).unwrap();
///
/// let (x, _p) = lsq.solve().unwrap();
/// assert!((x[0] - 1.0).abs() < 0.1);
/// ```
pub struct BatchLsq<T: FloatScalar, const N: usize> {
    /// Information matrix: Λ = Σ HᵀR⁻¹H.
    info: Matrix<T, N, N>,
    /// Information vector: η = Σ HᵀR⁻¹z.
    eta: Vector<T, N>,
}

impl<T: FloatScalar, const N: usize> BatchLsq<T, N> {
    /// Create a new batch estimator with zero prior information.
    pub fn new() -> Self {
        Self {
            info: Matrix::zeros(),
            eta: Vector::zeros(),
        }
    }

    /// Create a batch estimator with a Gaussian prior `N(x0, P0)`.
    ///
    /// Sets `Λ = P₀⁻¹` and `η = P₀⁻¹·x₀`.
    /// Returns `SingularInnovation` if `P0` is singular.
    pub fn with_prior(
        x0: &Vector<T, N>,
        p0: &Matrix<T, N, N>,
    ) -> Result<Self, EstimateError> {
        let p0_inv = p0
            .cholesky()
            .map_err(|_| EstimateError::SingularInnovation)?
            .inverse();
        let eta = p0_inv * *x0;
        Ok(Self {
            info: p0_inv,
            eta,
        })
    }

    /// Accumulate a linear observation `z = H·x + noise`, `noise ~ N(0, R)`.
    ///
    /// `M` is the measurement dimension (method-level const generic).
    ///
    /// Updates: `Λ += HᵀR⁻¹H`, `η += HᵀR⁻¹z`.
    pub fn add_observation<const M: usize>(
        &mut self,
        z: &Vector<T, M>,
        h: &Matrix<T, M, N>,
        r: &Matrix<T, M, M>,
    ) -> Result<(), EstimateError> {
        let r_inv = r
            .cholesky()
            .map_err(|_| EstimateError::SingularInnovation)?
            .inverse();
        let ht = h.transpose(); // N×M
        let ht_rinv = ht * r_inv; // N×M
        self.info = self.info + ht_rinv * *h; // N×N
        self.eta = self.eta + ht_rinv * *z; // N×1
        Ok(())
    }

    /// Solve for the state estimate and covariance.
    ///
    /// Returns `(x, P)` where `P = Λ⁻¹` and `x = P·η`.
    /// Returns `SingularInnovation` if the information matrix is singular
    /// (insufficient observations to determine all states).
    pub fn solve(&self) -> Result<(Vector<T, N>, Matrix<T, N, N>), EstimateError> {
        let p = self
            .info
            .cholesky()
            .map_err(|_| EstimateError::SingularInnovation)?
            .inverse();
        let x = p * self.eta;
        Ok((x, p))
    }

    /// Reference to the accumulated information matrix.
    #[inline]
    pub fn information_matrix(&self) -> &Matrix<T, N, N> {
        &self.info
    }

    /// Reference to the accumulated information vector.
    #[inline]
    pub fn information_vector(&self) -> &Vector<T, N> {
        &self.eta
    }

    /// Reset to zero information (discard all accumulated data).
    pub fn reset(&mut self) {
        self.info = Matrix::zeros();
        self.eta = Vector::zeros();
    }
}
