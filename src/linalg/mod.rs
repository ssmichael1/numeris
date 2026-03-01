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

use crate::traits::MatrixMut;

/// Get mutable references to sub-column slices of two different columns
/// simultaneously. Requires `col_a != col_b`.
///
/// Returns `(a_slice, b_slice)` where:
/// - `a_slice = &mut m[row_start..nrows, col_a]`
/// - `b_slice = &mut m[row_start..nrows, col_b]`
#[inline]
pub(crate) fn split_two_col_slices<'a, T>(
    m: &'a mut impl MatrixMut<T>,
    col_a: usize,
    col_b: usize,
    row_start: usize,
) -> (&'a mut [T], &'a mut [T]) {
    debug_assert_ne!(col_a, col_b);
    // Safety: col_a and col_b are different columns, so the slices don't overlap.
    // MatrixMut guarantees column slices are contiguous and non-overlapping.
    let ptr = m as *mut dyn MatrixMut<T>;
    let a = unsafe { &mut *ptr }.col_as_mut_slice(col_a, row_start);
    let b = unsafe { &mut *ptr }.col_as_mut_slice(col_b, row_start);
    (a, b)
}

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
