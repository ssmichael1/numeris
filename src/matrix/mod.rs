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
///
/// # Examples
///
/// ```
/// use numeris::Matrix;
///
/// let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
/// assert_eq!(a[(0, 1)], 2.0);
/// assert_eq!(a.nrows(), 2);
/// assert_eq!(a.ncols(), 2);
///
/// let b: Matrix<f64, 3, 3> = Matrix::eye();
/// assert_eq!(b[(0, 0)], 1.0);
/// assert_eq!(b[(0, 1)], 0.0);
/// ```
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

// ── Size aliases ────────────────────────────────────────────────────

/// Square matrix aliases: `Matrix1` through `Matrix6`.
macro_rules! square_alias {
    ($name:ident, $n:literal, $doc:expr) => {
        #[doc = $doc]
        pub type $name<T> = Matrix<T, $n, $n>;
    };
}

square_alias!(Matrix1, 1, "1×1 matrix.");
square_alias!(Matrix2, 2, "2×2 matrix.");
square_alias!(Matrix3, 3, "3×3 matrix.");
square_alias!(Matrix4, 4, "4×4 matrix.");
square_alias!(Matrix5, 5, "5×5 matrix.");
square_alias!(Matrix6, 6, "6×6 matrix.");

/// Rectangular matrix aliases: `Matrix{M}x{N}` for all M,N in 1..=6 where M ≠ N.
macro_rules! rect_alias {
    ($name:ident, $m:literal, $n:literal, $doc:expr) => {
        #[doc = $doc]
        pub type $name<T> = Matrix<T, $m, $n>;
    };
}

rect_alias!(Matrix1x2, 1, 2, "1×2 matrix.");
rect_alias!(Matrix1x3, 1, 3, "1×3 matrix.");
rect_alias!(Matrix1x4, 1, 4, "1×4 matrix.");
rect_alias!(Matrix1x5, 1, 5, "1×5 matrix.");
rect_alias!(Matrix1x6, 1, 6, "1×6 matrix.");

rect_alias!(Matrix2x1, 2, 1, "2×1 matrix.");
rect_alias!(Matrix2x3, 2, 3, "2×3 matrix.");
rect_alias!(Matrix2x4, 2, 4, "2×4 matrix.");
rect_alias!(Matrix2x5, 2, 5, "2×5 matrix.");
rect_alias!(Matrix2x6, 2, 6, "2×6 matrix.");

rect_alias!(Matrix3x1, 3, 1, "3×1 matrix.");
rect_alias!(Matrix3x2, 3, 2, "3×2 matrix.");
rect_alias!(Matrix3x4, 3, 4, "3×4 matrix.");
rect_alias!(Matrix3x5, 3, 5, "3×5 matrix.");
rect_alias!(Matrix3x6, 3, 6, "3×6 matrix.");

rect_alias!(Matrix4x1, 4, 1, "4×1 matrix.");
rect_alias!(Matrix4x2, 4, 2, "4×2 matrix.");
rect_alias!(Matrix4x3, 4, 3, "4×3 matrix.");
rect_alias!(Matrix4x5, 4, 5, "4×5 matrix.");
rect_alias!(Matrix4x6, 4, 6, "4×6 matrix.");

rect_alias!(Matrix5x1, 5, 1, "5×1 matrix.");
rect_alias!(Matrix5x2, 5, 2, "5×2 matrix.");
rect_alias!(Matrix5x3, 5, 3, "5×3 matrix.");
rect_alias!(Matrix5x4, 5, 4, "5×4 matrix.");
rect_alias!(Matrix5x6, 5, 6, "5×6 matrix.");

rect_alias!(Matrix6x1, 6, 1, "6×1 matrix.");
rect_alias!(Matrix6x2, 6, 2, "6×2 matrix.");
rect_alias!(Matrix6x3, 6, 3, "6×3 matrix.");
rect_alias!(Matrix6x4, 6, 4, "6×4 matrix.");
rect_alias!(Matrix6x5, 6, 5, "6×5 matrix.");

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
