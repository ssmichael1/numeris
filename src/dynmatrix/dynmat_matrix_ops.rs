//! Matrix operations between `DynMatrix` and `Matrix`
//!

use crate::dynmatrix::{DynMatrix, DynMatrixError, DynMatrixResult};
use crate::matrix::Matrix;
use crate::matrix::MatrixElem;

impl<const ROWS: usize, const COLS: usize, T> std::ops::Add<Matrix<ROWS, COLS, T>> for DynMatrix<T>
where
    T: MatrixElem + std::ops::Add<Output = T> + Copy,
{
    type Output = DynMatrixResult<DynMatrix<T>>;

    fn add(self, rhs: Matrix<ROWS, COLS, T>) -> Self::Output {
        if (self.rows_, self.cols_) != (ROWS, COLS) {
            return Err(DynMatrixError::DimensionMismatch);
        }
        let mut result = DynMatrix::zeros(ROWS, COLS);
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = self[(i, j)] + rhs[(i, j)];
            }
        }
        Ok(result)
    }
}

impl<const ROWS: usize, const COLS: usize, T> std::ops::Mul<Matrix<ROWS, COLS, T>> for DynMatrix<T>
where
    T: MatrixElem + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy + Default,
{
    type Output = DynMatrixResult<DynMatrix<T>>;

    fn mul(self, rhs: Matrix<ROWS, COLS, T>) -> Self::Output {
        if self.cols() != ROWS {
            return Err(DynMatrixError::DimensionMismatch);
        }
        let mut result = DynMatrix::zeros(self.rows(), COLS);
        for i in 0..self.rows() {
            for j in 0..COLS {
                let mut sum = T::default();
                for k in 0..self.cols() {
                    sum += self[(i, k)] * rhs[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        Ok(result)
    }
}

impl<const ROWS: usize, const COLS: usize, T> std::ops::Mul<&DynMatrix<T>>
    for &Matrix<ROWS, COLS, T>
where
    T: MatrixElem + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy + Default,
{
    type Output = DynMatrixResult<DynMatrix<T>>;

    fn mul(self, rhs: &DynMatrix<T>) -> Self::Output {
        if self.cols() != rhs.rows() {
            return Err(DynMatrixError::DimensionMismatch);
        }
        let mut result = DynMatrix::zeros(ROWS, rhs.cols());
        for i in 0..ROWS {
            for j in 0..rhs.cols() {
                let mut sum = T::default();
                for k in 0..COLS {
                    sum += self[(i, k)] * rhs[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        Ok(result)
    }
}

impl<const ROWS: usize, const COLS: usize, T> std::ops::Mul<DynMatrix<T>> for Matrix<ROWS, COLS, T>
where
    T: MatrixElem + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy + Default,
{
    type Output = DynMatrixResult<DynMatrix<T>>;

    fn mul(self, rhs: DynMatrix<T>) -> Self::Output {
        if self.cols() != rhs.rows() {
            return Err(DynMatrixError::DimensionMismatch);
        }
        let mut result = DynMatrix::zeros(ROWS, rhs.cols());
        for i in 0..ROWS {
            for j in 0..rhs.cols() {
                let mut sum = T::default();
                for k in 0..COLS {
                    sum += self[(i, k)] * rhs[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        Ok(result)
    }
}

impl<const ROWS: usize, const COLS: usize, T> std::ops::Mul<DynMatrix<T>> for &Matrix<ROWS, COLS, T>
where
    T: MatrixElem + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy + Default,
{
    type Output = DynMatrixResult<DynMatrix<T>>;

    fn mul(self, rhs: DynMatrix<T>) -> Self::Output {
        if self.cols() != rhs.rows() {
            return Err(DynMatrixError::DimensionMismatch);
        }
        let mut result = DynMatrix::zeros(ROWS, rhs.cols());
        for i in 0..ROWS {
            for j in 0..rhs.cols() {
                let mut sum = T::default();
                for k in 0..COLS {
                    sum += self[(i, k)] * rhs[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        Ok(result)
    }
}

impl<const ROWS: usize, const COLS: usize, T> std::ops::Mul<&Matrix<ROWS, COLS, T>>
    for &DynMatrix<T>
where
    T: MatrixElem + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy + Default,
{
    type Output = DynMatrixResult<DynMatrix<T>>;

    fn mul(self, rhs: &Matrix<ROWS, COLS, T>) -> Self::Output {
        if self.cols() != ROWS {
            return Err(DynMatrixError::DimensionMismatch);
        }
        let mut result = DynMatrix::zeros(self.rows(), COLS);
        for i in 0..self.rows() {
            for j in 0..COLS {
                let mut sum = T::default();
                for k in 0..self.cols() {
                    sum += self[(i, k)] * rhs[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        Ok(result)
    }
}
#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_dyn_fixed_mult() {
        // Dynamic matrix against itself
        let a = DynMatrix::from_vec_row_major(vec![1, 2, 3, 4], 2, 2).unwrap();
        let b = DynMatrix::from_vec_row_major(vec![5, 6, 7, 8], 2, 2).unwrap();
        let c = &a * &b;
        assert_eq!(
            c,
            Ok(DynMatrix::from_vec_row_major(vec![19, 22, 43, 50], 2, 2).unwrap())
        );
        // Dynamic matrix against fixed matrix
        let b = Matrix2i::from_row_major([[5, 6], [7, 8]]);
        let c = &a * &b;
        assert_eq!(
            c,
            Ok(DynMatrix::from_vec_row_major(vec![19, 22, 43, 50], 2, 2).unwrap())
        );
        // Fixed against dynamic
        let c = &b * &a;
        assert_eq!(
            c,
            Ok(DynMatrix::from_vec_row_major(vec![23, 34, 31, 46], 2, 2).unwrap())
        );
    }
}
