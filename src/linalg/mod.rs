pub(crate) mod cholesky;
pub(crate) mod expm;
pub(crate) mod hessenberg;
pub(crate) mod lu;
pub(crate) mod qr;
pub(crate) mod schur;
pub(crate) mod svd;
pub(crate) mod symmetric_eigen;

pub use cholesky::CholeskyDecomposition;
pub use cholesky::{cholesky_rank1_downdate, cholesky_rank1_update};
pub use expm::expm;
pub use lu::LuDecomposition;
pub use qr::QrDecomposition;
pub use qr::QrPivotDecomposition;
pub use schur::SchurDecomposition;
pub use svd::SvdDecomposition;
pub use symmetric_eigen::SymmetricEigen;

use crate::traits::MatrixMut;

/// Get mutable references to sub-column slices of two different columns
/// simultaneously.
///
/// Returns `(a_slice, b_slice)` where:
/// - `a_slice = &mut m[row_start..nrows, col_a]`
/// - `b_slice = &mut m[row_start..nrows, col_b]`
///
/// # Panics
///
/// Panics if `col_a == col_b` — the disjointness of the two slices is what
/// makes the aliasing below sound, so it is checked in release builds too.
#[inline]
#[allow(unsafe_code)] // audited exception to the crate-wide deny; see lib.rs
pub(crate) fn split_two_col_slices<T>(
    m: &mut impl MatrixMut<T>,
    col_a: usize,
    col_b: usize,
    row_start: usize,
) -> (&mut [T], &mut [T]) {
    assert_ne!(col_a, col_b, "split_two_col_slices: columns must differ");
    let ptr = m as *mut dyn MatrixMut<T>;
    // SAFETY: two exclusive reborrows of the same matrix coexist here, which is
    // sound only because they reach *disjoint* memory: `col_a != col_b`
    // (asserted above), and `MatrixMut` hands out each column as a separate
    // non-overlapping contiguous region, so no aliasing `&mut` to the same
    // bytes ever exist. Both reborrows derive from the same raw pointer, so
    // creating the second does not invalidate the first's provenance.
    let a = unsafe { &mut *ptr }.col_as_mut_slice(col_a, row_start);
    // SAFETY: as above — the same argument, reaching the second disjoint column.
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
