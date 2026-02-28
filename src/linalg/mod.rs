mod cholesky;
mod lu;
mod qr;

pub use cholesky::CholeskyDecomposition;
pub use lu::LuDecomposition;
pub use qr::QrDecomposition;

/// Errors from linear algebra operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinalgError {
    /// Matrix is singular or nearly singular.
    Singular,
    /// Matrix is not positive definite (required for Cholesky).
    NotPositiveDefinite,
}

impl core::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LinalgError::Singular => write!(f, "matrix is singular"),
            LinalgError::NotPositiveDefinite => write!(f, "matrix is not positive definite"),
        }
    }
}
