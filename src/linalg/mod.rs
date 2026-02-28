pub(crate) mod cholesky;
pub(crate) mod hessenberg;
pub(crate) mod lu;
pub(crate) mod qr;
pub(crate) mod schur;
pub(crate) mod svd;
pub(crate) mod symmetric_eigen;

pub use cholesky::CholeskyDecomposition;
pub use lu::LuDecomposition;
pub use qr::QrDecomposition;
pub use schur::SchurDecomposition;
pub use svd::SvdDecomposition;
pub use symmetric_eigen::SymmetricEigen;

/// Errors from linear algebra operations.
///
/// Returned by decomposition constructors and convenience methods
/// (`solve`, `inverse`, `cholesky`, `qr`, `lu`).
///
/// ```
/// use numeris::Matrix;
/// use numeris::linalg::LinalgError;
///
/// let singular = Matrix::new([[1.0_f64, 2.0], [2.0, 4.0]]);
/// assert_eq!(singular.lu().unwrap_err(), LinalgError::Singular);
///
/// let not_pd = Matrix::new([[1.0_f64, 5.0], [5.0, 1.0]]);
/// assert_eq!(not_pd.cholesky().unwrap_err(), LinalgError::NotPositiveDefinite);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinalgError {
    /// Matrix is singular or nearly singular.
    Singular,
    /// Matrix is not positive definite (required for Cholesky).
    NotPositiveDefinite,
    /// Iterative algorithm did not converge within the iteration budget.
    ConvergenceFailure,
}

impl core::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LinalgError::Singular => write!(f, "matrix is singular"),
            LinalgError::NotPositiveDefinite => write!(f, "matrix is not positive definite"),
            LinalgError::ConvergenceFailure => write!(f, "iterative algorithm did not converge"),
        }
    }
}
