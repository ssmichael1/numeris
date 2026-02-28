use core::ops::{Add, Mul, Sub};

use crate::traits::Scalar;
use crate::Matrix;

use super::DynMatrix;

// ── Matrix * DynMatrix → DynMatrix ──────────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Mul<DynMatrix<T>> for Matrix<T, M, N> {
    type Output = DynMatrix<T>;

    fn mul(self, rhs: DynMatrix<T>) -> DynMatrix<T> {
        DynMatrix::from(self) * rhs
    }
}

impl<T: Scalar, const M: usize, const N: usize> Mul<&DynMatrix<T>> for Matrix<T, M, N> {
    type Output = DynMatrix<T>;

    fn mul(self, rhs: &DynMatrix<T>) -> DynMatrix<T> {
        &DynMatrix::from(self) * rhs
    }
}

impl<T: Scalar, const M: usize, const N: usize> Mul<DynMatrix<T>> for &Matrix<T, M, N> {
    type Output = DynMatrix<T>;

    fn mul(self, rhs: DynMatrix<T>) -> DynMatrix<T> {
        DynMatrix::from(self) * rhs
    }
}

impl<T: Scalar, const M: usize, const N: usize> Mul<&DynMatrix<T>> for &Matrix<T, M, N> {
    type Output = DynMatrix<T>;

    fn mul(self, rhs: &DynMatrix<T>) -> DynMatrix<T> {
        &DynMatrix::from(self) * rhs
    }
}

// ── DynMatrix * Matrix → DynMatrix ──────────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Mul<Matrix<T, M, N>> for DynMatrix<T> {
    type Output = DynMatrix<T>;

    fn mul(self, rhs: Matrix<T, M, N>) -> DynMatrix<T> {
        self * DynMatrix::from(rhs)
    }
}

impl<T: Scalar, const M: usize, const N: usize> Mul<&Matrix<T, M, N>> for DynMatrix<T> {
    type Output = DynMatrix<T>;

    fn mul(self, rhs: &Matrix<T, M, N>) -> DynMatrix<T> {
        self * DynMatrix::from(rhs)
    }
}

impl<T: Scalar, const M: usize, const N: usize> Mul<Matrix<T, M, N>> for &DynMatrix<T> {
    type Output = DynMatrix<T>;

    fn mul(self, rhs: Matrix<T, M, N>) -> DynMatrix<T> {
        self * &DynMatrix::from(rhs)
    }
}

impl<T: Scalar, const M: usize, const N: usize> Mul<&Matrix<T, M, N>> for &DynMatrix<T> {
    type Output = DynMatrix<T>;

    fn mul(self, rhs: &Matrix<T, M, N>) -> DynMatrix<T> {
        self * &DynMatrix::from(rhs)
    }
}

// ── Matrix + DynMatrix → DynMatrix ──────────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Add<DynMatrix<T>> for Matrix<T, M, N> {
    type Output = DynMatrix<T>;

    fn add(self, rhs: DynMatrix<T>) -> DynMatrix<T> {
        DynMatrix::from(self) + rhs
    }
}

impl<T: Scalar, const M: usize, const N: usize> Add<&DynMatrix<T>> for Matrix<T, M, N> {
    type Output = DynMatrix<T>;

    fn add(self, rhs: &DynMatrix<T>) -> DynMatrix<T> {
        &DynMatrix::from(self) + rhs
    }
}

// ── DynMatrix + Matrix → DynMatrix ──────────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Add<Matrix<T, M, N>> for DynMatrix<T> {
    type Output = DynMatrix<T>;

    fn add(self, rhs: Matrix<T, M, N>) -> DynMatrix<T> {
        self + DynMatrix::from(rhs)
    }
}

impl<T: Scalar, const M: usize, const N: usize> Add<&Matrix<T, M, N>> for DynMatrix<T> {
    type Output = DynMatrix<T>;

    fn add(self, rhs: &Matrix<T, M, N>) -> DynMatrix<T> {
        self + DynMatrix::from(rhs)
    }
}

// ── Matrix - DynMatrix → DynMatrix ──────────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Sub<DynMatrix<T>> for Matrix<T, M, N> {
    type Output = DynMatrix<T>;

    fn sub(self, rhs: DynMatrix<T>) -> DynMatrix<T> {
        DynMatrix::from(self) - rhs
    }
}

impl<T: Scalar, const M: usize, const N: usize> Sub<&DynMatrix<T>> for Matrix<T, M, N> {
    type Output = DynMatrix<T>;

    fn sub(self, rhs: &DynMatrix<T>) -> DynMatrix<T> {
        &DynMatrix::from(self) - rhs
    }
}

// ── DynMatrix - Matrix → DynMatrix ──────────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Sub<Matrix<T, M, N>> for DynMatrix<T> {
    type Output = DynMatrix<T>;

    fn sub(self, rhs: Matrix<T, M, N>) -> DynMatrix<T> {
        self - DynMatrix::from(rhs)
    }
}

impl<T: Scalar, const M: usize, const N: usize> Sub<&Matrix<T, M, N>> for DynMatrix<T> {
    type Output = DynMatrix<T>;

    fn sub(self, rhs: &Matrix<T, M, N>) -> DynMatrix<T> {
        self - DynMatrix::from(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_times_dynmatrix() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let d = DynMatrix::from_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        let result = m * &d;
        assert_eq!(result[(0, 0)], 19.0);
        assert_eq!(result[(1, 1)], 50.0);
    }

    #[test]
    fn dynmatrix_times_matrix() {
        let d = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let m = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);
        let result = d * m;
        assert_eq!(result[(0, 0)], 19.0);
        assert_eq!(result[(1, 1)], 50.0);
    }

    #[test]
    fn matrix_add_dynmatrix() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let d = DynMatrix::from_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        let result = m + d;
        assert_eq!(result[(0, 0)], 6.0);
        assert_eq!(result[(1, 1)], 12.0);
    }

    #[test]
    fn dynmatrix_sub_matrix() {
        let d = DynMatrix::from_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let result = d - m;
        assert_eq!(result[(0, 0)], 4.0);
        assert_eq!(result[(1, 1)], 4.0);
    }

    #[test]
    fn mixed_non_square_multiply() {
        let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let d = DynMatrix::from_slice(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let result = m * &d;
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 2);
        assert_eq!(result[(0, 0)], 58.0);
    }
}
