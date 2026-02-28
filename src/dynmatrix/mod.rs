pub mod aliases;
mod block;
mod linalg;
mod mixed_ops;
mod norm;
mod ops;
mod slice;
mod square;
mod util;
mod vector;

pub use aliases::*;
pub use linalg::{DynCholesky, DynLu, DynQr};
pub use vector::DynVector;

use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Index, IndexMut};

use crate::traits::{MatrixMut, MatrixRef, Scalar};
use crate::Matrix;

/// Dimension mismatch error for fallible conversions.
///
/// Returned by `TryFrom<&DynMatrix<T>>` for `Matrix<T, M, N>` when
/// the runtime dimensions don't match the compile-time dimensions.
///
/// # Example
///
/// ```
/// use numeris::{DynMatrix, Matrix};
/// use numeris::dynmatrix::DimensionMismatch;
///
/// let d = DynMatrix::zeros(2, 3, 0.0_f64);
/// let result: Result<Matrix<f64, 2, 2>, _> = (&d).try_into();
/// assert!(result.is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DimensionMismatch {
    /// Expected `(rows, cols)`.
    pub expected: (usize, usize),
    /// Got `(rows, cols)`.
    pub got: (usize, usize),
}

impl core::fmt::Display for DimensionMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "dimension mismatch: expected {}x{}, got {}x{}",
            self.expected.0, self.expected.1, self.got.0, self.got.1
        )
    }
}

/// Dynamically-sized heap-allocated matrix.
///
/// Row-major `Vec<T>` storage, matching the layout of fixed-size [`Matrix`].
/// Dimensions are set at runtime. Implements [`MatrixRef`] and [`MatrixMut`],
/// so all generic linalg free functions work with `DynMatrix` out of the box.
///
/// # Examples
///
/// ```
/// use numeris::DynMatrix;
///
/// let a = DynMatrix::from_slice(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]);
/// assert_eq!(a[(0, 1)], 2.0);
/// assert_eq!(a.nrows(), 2);
/// assert_eq!(a.ncols(), 2);
///
/// let b = DynMatrix::eye(3, 0.0_f64);
/// assert_eq!(b[(0, 0)], 1.0);
/// assert_eq!(b[(0, 1)], 0.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DynMatrix<T> {
    data: Vec<T>,
    nrows: usize,
    ncols: usize,
}

// ── Constructors ────────────────────────────────────────────────────

impl<T: Scalar> DynMatrix<T> {
    /// Create an `nrows x ncols` matrix filled with `value`.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::zeros(2, 3, 0.0_f64);
    /// assert_eq!(m.nrows(), 2);
    /// assert_eq!(m.ncols(), 3);
    /// assert_eq!(m[(1, 2)], 0.0);
    /// ```
    pub fn zeros(nrows: usize, ncols: usize, _zero: T) -> Self {
        Self {
            data: vec![T::zero(); nrows * ncols],
            nrows,
            ncols,
        }
    }

    /// Create a matrix filled with a given value.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::fill(2, 3, 7.0_f64);
    /// assert_eq!(m[(0, 0)], 7.0);
    /// assert_eq!(m[(1, 2)], 7.0);
    /// ```
    pub fn fill(nrows: usize, ncols: usize, value: T) -> Self {
        Self {
            data: vec![value; nrows * ncols],
            nrows,
            ncols,
        }
    }

    /// Create an `n x n` identity matrix.
    ///
    /// The `_zero` parameter is only used for type inference.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let id = DynMatrix::eye(3, 0.0_f64);
    /// assert_eq!(id[(0, 0)], 1.0);
    /// assert_eq!(id[(0, 1)], 0.0);
    /// assert_eq!(id[(2, 2)], 1.0);
    /// ```
    pub fn eye(n: usize, _zero: T) -> Self {
        let mut m = Self::zeros(n, n, T::zero());
        for i in 0..n {
            m[(i, i)] = T::one();
        }
        m
    }

    /// Create a matrix from a flat slice in row-major order.
    ///
    /// Panics if `slice.len() != nrows * ncols`.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(m[(0, 2)], 3.0);
    /// assert_eq!(m[(1, 0)], 4.0);
    /// ```
    pub fn from_slice(nrows: usize, ncols: usize, slice: &[T]) -> Self {
        assert_eq!(
            slice.len(),
            nrows * ncols,
            "slice length {} does not match {}x{} matrix",
            slice.len(),
            nrows,
            ncols,
        );
        Self {
            data: slice.to_vec(),
            nrows,
            ncols,
        }
    }

    /// Create a matrix from an owned `Vec<T>` in row-major order.
    ///
    /// Panics if `data.len() != nrows * ncols`.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(m[(0, 0)], 1.0);
    /// assert_eq!(m[(1, 1)], 4.0);
    /// ```
    pub fn from_vec(nrows: usize, ncols: usize, data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "vec length {} does not match {}x{} matrix",
            data.len(),
            nrows,
            ncols,
        );
        Self { data, nrows, ncols }
    }
}

impl<T> DynMatrix<T> {
    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Whether the matrix is square.
    #[inline]
    pub fn is_square(&self) -> bool {
        self.nrows == self.ncols
    }

    /// Create a matrix by calling `f(row, col)` for each element.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::from_fn(3, 3, |i, j| if i == j { 1.0_f64 } else { 0.0 });
    /// assert_eq!(m[(0, 0)], 1.0);
    /// assert_eq!(m[(0, 1)], 0.0);
    /// ```
    pub fn from_fn(nrows: usize, ncols: usize, f: impl Fn(usize, usize) -> T) -> Self {
        let mut data = Vec::with_capacity(nrows * ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                data.push(f(i, j));
            }
        }
        Self { data, nrows, ncols }
    }
}

// ── MatrixRef / MatrixMut ───────────────────────────────────────────

impl<T> MatrixRef<T> for DynMatrix<T> {
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.ncols
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> &T {
        &self.data[row * self.ncols + col]
    }
}

impl<T> MatrixMut<T> for DynMatrix<T> {
    #[inline]
    fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.data[row * self.ncols + col]
    }
}

// ── Index ───────────────────────────────────────────────────────────

impl<T> Index<(usize, usize)> for DynMatrix<T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &T {
        &self.data[row * self.ncols + col]
    }
}

impl<T> IndexMut<(usize, usize)> for DynMatrix<T> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        &mut self.data[row * self.ncols + col]
    }
}

// ── Conversions: Matrix ↔ DynMatrix ─────────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> From<Matrix<T, M, N>> for DynMatrix<T> {
    /// Convert a fixed-size `Matrix` into a `DynMatrix`.
    ///
    /// ```
    /// use numeris::{Matrix, DynMatrix};
    /// let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let d: DynMatrix<f64> = m.into();
    /// assert_eq!(d.nrows(), 2);
    /// assert_eq!(d[(1, 1)], 4.0);
    /// ```
    fn from(m: Matrix<T, M, N>) -> Self {
        Self {
            data: m.as_slice().to_vec(),
            nrows: M,
            ncols: N,
        }
    }
}

impl<T: Scalar, const M: usize, const N: usize> From<&Matrix<T, M, N>> for DynMatrix<T> {
    fn from(m: &Matrix<T, M, N>) -> Self {
        Self {
            data: m.as_slice().to_vec(),
            nrows: M,
            ncols: N,
        }
    }
}

impl<T: Scalar, const M: usize, const N: usize> TryFrom<&DynMatrix<T>> for Matrix<T, M, N> {
    type Error = DimensionMismatch;

    /// Try to convert a `DynMatrix` into a fixed-size `Matrix`.
    ///
    /// Fails if the runtime dimensions don't match `M x N`.
    ///
    /// ```
    /// use numeris::{Matrix, DynMatrix};
    /// let d = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    /// let m: Matrix<f64, 2, 2> = (&d).try_into().unwrap();
    /// assert_eq!(m[(0, 0)], 1.0);
    /// assert_eq!(m[(1, 1)], 4.0);
    /// ```
    fn try_from(d: &DynMatrix<T>) -> Result<Self, Self::Error> {
        if d.nrows != M || d.ncols != N {
            return Err(DimensionMismatch {
                expected: (M, N),
                got: (d.nrows, d.ncols),
            });
        }
        Ok(Matrix::from_slice(d.data.as_slice()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros() {
        let m = DynMatrix::zeros(3, 4, 0.0_f64);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 4);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(m[(i, j)], 0.0);
            }
        }
    }

    #[test]
    fn fill() {
        let m = DynMatrix::fill(2, 3, 7.0_f64);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(m[(i, j)], 7.0);
            }
        }
    }

    #[test]
    fn eye() {
        let m = DynMatrix::eye(3, 0.0_f64);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(m[(i, j)], expected);
            }
        }
    }

    #[test]
    fn from_slice() {
        let m = DynMatrix::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 2)], 3.0);
        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(1, 2)], 6.0);
    }

    #[test]
    #[should_panic(expected = "slice length")]
    fn from_slice_wrong_length() {
        let _ = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn from_vec() {
        let m = DynMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 4.0);
    }

    #[test]
    fn from_fn() {
        let m = DynMatrix::from_fn(3, 3, |i, j| (i * 3 + j) as f64);
        assert_eq!(m[(0, 0)], 0.0);
        assert_eq!(m[(1, 1)], 4.0);
        assert_eq!(m[(2, 2)], 8.0);
    }

    #[test]
    fn index_mut() {
        let mut m = DynMatrix::zeros(2, 2, 0.0_f64);
        m[(0, 1)] = 5.0;
        assert_eq!(m[(0, 1)], 5.0);
    }

    #[test]
    fn matrix_ref_trait() {
        let m = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        fn trace<T: Scalar>(m: &impl MatrixRef<T>) -> T {
            let mut sum = T::zero();
            let n = m.nrows().min(m.ncols());
            for i in 0..n {
                sum = sum + *m.get(i, i);
            }
            sum
        }
        assert_eq!(trace(&m), 5.0);
    }

    #[test]
    fn matrix_mut_trait() {
        let mut m = DynMatrix::zeros(2, 2, 0.0_f64);
        fn set_diag<T: Scalar>(m: &mut impl MatrixMut<T>, val: T) {
            let n = m.nrows().min(m.ncols());
            for i in 0..n {
                *m.get_mut(i, i) = val;
            }
        }
        set_diag(&mut m, 7.0);
        assert_eq!(m[(0, 0)], 7.0);
        assert_eq!(m[(1, 1)], 7.0);
        assert_eq!(m[(0, 1)], 0.0);
    }

    #[test]
    fn from_matrix() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let d: DynMatrix<f64> = m.into();
        assert_eq!(d.nrows(), 2);
        assert_eq!(d.ncols(), 2);
        assert_eq!(d[(0, 0)], 1.0);
        assert_eq!(d[(1, 1)], 4.0);
    }

    #[test]
    fn from_matrix_ref() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let d: DynMatrix<f64> = (&m).into();
        assert_eq!(d[(0, 0)], 1.0);
    }

    #[test]
    fn try_into_matrix() {
        let d = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let m: Matrix<f64, 2, 2> = (&d).try_into().unwrap();
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 4.0);
    }

    #[test]
    fn try_into_matrix_wrong_dims() {
        let d = DynMatrix::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result: Result<Matrix<f64, 2, 2>, _> = (&d).try_into();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.expected, (2, 2));
        assert_eq!(err.got, (2, 3));
    }

    #[test]
    fn is_square() {
        let sq = DynMatrix::zeros(3, 3, 0.0_f64);
        assert!(sq.is_square());
        let rect = DynMatrix::zeros(2, 3, 0.0_f64);
        assert!(!rect.is_square());
    }

    #[test]
    fn clone_eq() {
        let a = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = a.clone();
        assert_eq!(a, b);
    }
}
