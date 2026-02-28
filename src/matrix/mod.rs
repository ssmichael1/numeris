mod block;
mod norm;
mod ops;
mod slice;
mod square;
mod util;
pub mod vector;

use core::ops::{Index, IndexMut};

use crate::traits::{MatrixMut, MatrixRef, Scalar};

/// Fixed-size matrix with `M` rows and `N` columns.
///
/// Storage is row-major: `data[row][col]`.
/// Stack-allocated, no-std compatible.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix<T, const M: usize, const N: usize> {
    data: [[T; N]; M],
}

impl<T, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Create a matrix from a raw 2D array (row-major).
    #[inline]
    pub fn new(data: [[T; N]; M]) -> Self {
        Self { data }
    }

    /// Number of rows.
    #[inline]
    pub const fn nrows(&self) -> usize {
        M
    }

    /// Number of columns.
    #[inline]
    pub const fn ncols(&self) -> usize {
        N
    }
}

impl<T: Scalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Create a matrix filled with zeros.
    pub fn zeros() -> Self {
        Self {
            data: [[T::zero(); N]; M],
        }
    }
}

impl<T: Scalar, const N: usize> Matrix<T, N, N> {
    /// Create an identity matrix (square matrices only).
    pub fn eye() -> Self {
        let mut m = Self::zeros();
        for i in 0..N {
            m.data[i][i] = T::one();
        }
        m
    }
}

impl<T, const M: usize, const N: usize> MatrixRef<T> for Matrix<T, M, N> {
    #[inline]
    fn nrows(&self) -> usize {
        M
    }

    #[inline]
    fn ncols(&self) -> usize {
        N
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> &T {
        &self.data[row][col]
    }
}

impl<T, const M: usize, const N: usize> MatrixMut<T> for Matrix<T, M, N> {
    #[inline]
    fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.data[row][col]
    }
}

// Index by (row, col) tuple
impl<T, const M: usize, const N: usize> Index<(usize, usize)> for Matrix<T, M, N> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &T {
        &self.data[row][col]
    }
}

impl<T, const M: usize, const N: usize> IndexMut<(usize, usize)> for Matrix<T, M, N> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        &mut self.data[row][col]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_and_eye() {
        let z: Matrix<f64, 3, 3> = Matrix::zeros();
        assert_eq!(z[(0, 0)], 0.0);
        assert_eq!(z[(2, 2)], 0.0);

        let id: Matrix<f64, 3, 3> = Matrix::eye();
        assert_eq!(id[(0, 0)], 1.0);
        assert_eq!(id[(1, 1)], 1.0);
        assert_eq!(id[(0, 1)], 0.0);
    }

    #[test]
    fn new_and_index() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 0)], 3.0);
        assert_eq!(m[(1, 1)], 4.0);
    }

    #[test]
    fn index_mut() {
        let mut m: Matrix<f64, 2, 2> = Matrix::zeros();
        m[(0, 1)] = 5.0;
        assert_eq!(m[(0, 1)], 5.0);
    }

    #[test]
    fn non_square() {
        let m: Matrix<f64, 2, 3> = Matrix::zeros();
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 3);
    }

    #[test]
    fn matrix_ref_trait() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);

        // Use via trait, as a generic algorithm would
        fn trace_generic<T: Scalar>(m: &impl MatrixRef<T>) -> T {
            let mut sum = T::zero();
            let n = m.nrows().min(m.ncols());
            for i in 0..n {
                sum = sum + *m.get(i, i);
            }
            sum
        }

        assert_eq!(trace_generic(&m), 5.0);
    }

    #[test]
    fn matrix_mut_trait() {
        let mut m: Matrix<f64, 2, 2> = Matrix::zeros();

        // Use via trait
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
    fn integer_matrix() {
        let m: Matrix<i32, 2, 2> = Matrix::eye();
        assert_eq!(m[(0, 0)], 1);
        assert_eq!(m[(0, 1)], 0);
    }
}
