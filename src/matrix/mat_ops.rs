use super::{Matrix, MatrixElem};

// Matrix addition by scalar
impl<const ROWS: usize, const COLS: usize, T> std::ops::Add<T> for Matrix<ROWS, COLS, T>
where
    T: MatrixElem,
{
    type Output = Matrix<ROWS, COLS, T>;

    fn add(self, other: T) -> Self::Output {
        let mut result = Matrix::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = self[(i, j)] + other;
            }
        }
        result
    }
}

// Matrix subtraction by scalar
impl<const ROWS: usize, const COLS: usize, T> std::ops::Sub<T> for Matrix<ROWS, COLS, T>
where
    T: MatrixElem,
{
    type Output = Matrix<ROWS, COLS, T>;

    fn sub(self, other: T) -> Self::Output {
        let mut result = Matrix::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = self[(i, j)] - other;
            }
        }
        result
    }
}

// Matrix multiplication by scalar
impl<const ROWS: usize, const COLS: usize, T> std::ops::Mul<T> for Matrix<ROWS, COLS, T>
where
    T: MatrixElem,
{
    type Output = Matrix<ROWS, COLS, T>;

    fn mul(self, other: T) -> Self::Output {
        let mut result = Matrix::<ROWS, COLS, T>::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = self[(i, j)] * other;
            }
        }
        result
    }
}

// Matrix division by scalar
impl<const ROWS: usize, const COLS: usize, T> std::ops::Div<T> for Matrix<ROWS, COLS, T>
where
    T: MatrixElem,
{
    type Output = Matrix<ROWS, COLS, T>;

    fn div(self, other: T) -> Self::Output {
        let mut result = Matrix::<ROWS, COLS, T>::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = self[(i, j)] / other;
            }
        }
        result
    }
}

// Matrix addition
impl<const ROWS: usize, const COLS: usize, T> std::ops::Add for Matrix<ROWS, COLS, T>
where
    T: MatrixElem,
{
    type Output = Matrix<ROWS, COLS, T>;

    fn add(self, other: Self) -> Self::Output {
        let mut result = Matrix::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = self[(i, j)] + other[(i, j)];
            }
        }
        result
    }
}

/// Matrix subtraction
impl<const ROWS: usize, const COLS: usize, T> std::ops::Sub for Matrix<ROWS, COLS, T>
where
    T: MatrixElem,
{
    type Output = Matrix<ROWS, COLS, T>;

    fn sub(self, other: Self) -> Self::Output {
        let mut result = Matrix::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = self[(i, j)] - other[(i, j)];
            }
        }
        result
    }
}

// Generic Matrix multiplication by another matrix
impl<const ROWS: usize, const INNER: usize, T, const COLS: usize>
    std::ops::Mul<Matrix<INNER, COLS, T>> for Matrix<ROWS, INNER, T>
where
    T: MatrixElem,
{
    type Output = Matrix<ROWS, COLS, T>;

    fn mul(self, other: Matrix<INNER, COLS, T>) -> Self::Output {
        let mut result = Self::Output::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = (0..INNER).map(|k| self[(i, k)] * other[(k, j)]).sum::<T>();
            }
        }
        result
    }
}

// Generic Matrix multiplication by another matrix, by reference
impl<const ROWS: usize, const INNER: usize, T, const COLS: usize>
    std::ops::Mul<Matrix<INNER, COLS, T>> for &Matrix<ROWS, INNER, T>
where
    T: MatrixElem,
{
    type Output = Matrix<ROWS, COLS, T>;

    fn mul(self, other: Matrix<INNER, COLS, T>) -> Self::Output {
        let mut result = Matrix::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = (0..INNER).map(|k| self[(i, k)] * other[(k, j)]).sum::<T>();
            }
        }
        result
    }
}

// Generic Matrix multiplication by another reference matrix, by reference
impl<const ROWS: usize, const INNER: usize, T, const COLS: usize>
    std::ops::Mul<&Matrix<INNER, COLS, T>> for &Matrix<ROWS, INNER, T>
where
    T: MatrixElem,
{
    type Output = Matrix<ROWS, COLS, T>;

    fn mul(self, other: &Matrix<INNER, COLS, T>) -> Self::Output {
        let mut result = Matrix::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = (0..INNER).map(|k| self[(i, k)] * other[(k, j)]).sum::<T>();
            }
        }
        result
    }
}

// Generic Matrix multiplication by another reference matrix, by reference
impl<const ROWS: usize, const INNER: usize, T, const COLS: usize>
    std::ops::Mul<&Matrix<INNER, COLS, T>> for Matrix<ROWS, INNER, T>
where
    T: MatrixElem,
{
    type Output = Matrix<ROWS, COLS, T>;

    fn mul(self, other: &Matrix<INNER, COLS, T>) -> Self::Output {
        let mut result = Matrix::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = (0..INNER).map(|k| self[(i, k)] * other[(k, j)]).sum::<T>();
            }
        }
        result
    }
}

// Matrix multiplication by a scalar, by reference
impl<const ROWS: usize, const COLS: usize, T> std::ops::Mul<T> for &Matrix<ROWS, COLS, T>
where
    T: MatrixElem,
{
    type Output = Matrix<ROWS, COLS, T>;

    fn mul(self, other: T) -> Self::Output {
        let mut result = Matrix::<ROWS, COLS, T>::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = self[(i, j)] * other;
            }
        }
        result
    }
}

// Scalar LHS multiply a matrix for f32
impl<const ROWS: usize, const COLS: usize> std::ops::Mul<Matrix<ROWS, COLS, f32>> for f32 {
    type Output = Matrix<ROWS, COLS, f32>;

    fn mul(self, other: Matrix<ROWS, COLS, f32>) -> Self::Output {
        let mut result = Matrix::<ROWS, COLS, f32>::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = self * other[(i, j)];
            }
        }
        result
    }
}

// Scalar LHS multiply a reference matrix for f32
impl<const ROWS: usize, const COLS: usize> std::ops::Mul<&Matrix<ROWS, COLS, f32>> for f32 {
    type Output = Matrix<ROWS, COLS, f32>;

    fn mul(self, other: &Matrix<ROWS, COLS, f32>) -> Self::Output {
        let mut result = Matrix::<ROWS, COLS, f32>::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = self * other[(i, j)];
            }
        }
        result
    }
}

// Scalar LHS multiply a matrix for f64
impl<const ROWS: usize, const COLS: usize> std::ops::Mul<Matrix<ROWS, COLS, f64>> for f64 {
    type Output = Matrix<ROWS, COLS, f64>;

    fn mul(self, other: Matrix<ROWS, COLS, f64>) -> Self::Output {
        let mut result = Matrix::<ROWS, COLS, f64>::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = self * other[(i, j)];
            }
        }
        result
    }
}

// Scalar LHS multiply a reference matrix for f64
impl<const ROWS: usize, const COLS: usize> std::ops::Mul<&Matrix<ROWS, COLS, f64>> for f64 {
    type Output = Matrix<ROWS, COLS, f64>;

    fn mul(self, other: &Matrix<ROWS, COLS, f64>) -> Self::Output {
        let mut result = Matrix::<ROWS, COLS, f64>::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = self * other[(i, j)];
            }
        }
        result
    }
}

// Index
impl<const ROWS: usize, const COLS: usize, T> std::ops::Index<(usize, usize)>
    for Matrix<ROWS, COLS, T>
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        &self.data[col][row]
    }
}

/// Explicitly create matrix from array of array
///
/// # Example
///
/// ```rust
/// use tiny_matrix::prelude::*;
/// let mat = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
/// let mat2: Matrix<2, 2, f64> = [[5.0, 6.0], [7.0, 8.0]].into();
/// ```
impl<const ROWS: usize, const COLS: usize, T> From<[[T; ROWS]; COLS]> for Matrix<ROWS, COLS, T>
where
    T: MatrixElem,
{
    fn from(arr: [[T; ROWS]; COLS]) -> Self {
        Self { data: arr }
    }
}

/// Index mut
impl<const ROWS: usize, const COLS: usize, T> std::ops::IndexMut<(usize, usize)>
    for Matrix<ROWS, COLS, T>
{
    #[inline(always)]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        &mut self.data[col][row]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_multiplication() {
        let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);
        let result = a * b;
        // Expected value for column-major inputs
        let expected: Matrix<2, 2, f64> = Matrix::new([[23.0, 34.0], [31.0, 46.0]]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_addition() {
        let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);
        let result = a + b;
        let expected: Matrix<2, 2, f64> = Matrix::new([[6.0, 8.0], [10.0, 12.0]]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_scalar_multiplication() {
        let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let scalar = 2.0;
        let result = scalar * a;
        let expected = Matrix::new([[2.0, 4.0], [6.0, 8.0]]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_indexing() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 4.0);
        assert_eq!(m[(0, 1)], 3.0);
    }

    #[test]
    fn test_indexing_mut() {
        let mut m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        m[(0, 0)] = 5.0;
        assert_eq!(m[(0, 0)], 5.0);
    }

    #[test]
    fn test_lhs_scalar_multiply() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let result = 2.0 * m;
        let expected = Matrix::new([[2.0, 4.0], [6.0, 8.0]]);
        assert_eq!(result, expected);
    }
}
