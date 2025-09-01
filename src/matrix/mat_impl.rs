use super::{Matrix, MatrixElem};

impl<const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T>
where
    T: MatrixElem,
{
    /// Return the matrix shape
    ///
    /// # Returns:
    ///
    /// The shape of the matrix as a tuple (rows, columns).
    ///
    /// # Example:
    /// ```rust
    /// use tiny_matrix::mat;
    /// let m = mat![[1.0, 2.0], [3.0, 4.0]];
    /// assert!(m.shape() == (2, 2));
    /// ```
    pub const fn shape(&self) -> (usize, usize) {
        (ROWS, COLS)
    }

    /// Returns the number of rows in the matrix.
    ///
    /// # Returns
    ///
    /// The number of rows in the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::mat;
    /// let m = mat![[1.0, 2.0], [3.0, 4.0]];
    /// assert!(m.rows() == 2);
    /// ```
    pub const fn rows(&self) -> usize {
        ROWS
    }

    /// Returns the number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// The number of columns in the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::mat;
    /// let m = mat![[1.0, 2.0], [3.0, 4.0]];
    /// assert!(m.cols() == 2);
    /// ```
    pub const fn cols(&self) -> usize {
        COLS
    }

    /// Create a new matrix from column-major array (the default)
    ///
    /// # Returns
    ///
    /// The new matrix.
    pub fn new(data: [[T; ROWS]; COLS]) -> Self {
        Self { data }
    }

    /// Create a new matrix from column-major order data
    ///
    /// # Returns
    ///
    /// The new matrix.
    pub fn from_column_major(data: [[T; ROWS]; COLS]) -> Self {
        Self { data }
    }

    /// Create a new matrix from row-major order data
    ///
    /// # Returns
    ///
    /// The new matrix.
    pub fn from_row_major(data: [[T; COLS]; ROWS]) -> Self {
        Matrix::<COLS, ROWS, T> { data }.transpose()
    }

    /// Transpose
    ///
    /// Computes the transpose of the matrix by swapping rows and columns.
    ///
    /// # Returns:
    ///
    /// The transposed matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::mat;
    /// let m = mat![[1.0, 2.0], [3.0, 4.0]];
    /// let t = m.transpose();
    /// assert!(t == mat![[1.0, 3.0], [2.0, 4.0]]);
    /// ```
    pub fn transpose(&self) -> Matrix<COLS, ROWS, T> {
        let mut transposed = Matrix::<COLS, ROWS, T>::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                transposed[(j, i)] = self[(i, j)];
            }
        }
        transposed
    }

    /// Matrix minor
    ///
    /// Computes the minor of the matrix by removing the specified row and column.
    ///
    /// # Returns:
    ///
    /// The minor of the matrix.
    ///
    /// # Generic Arguments:
    ///
    /// * `RR`: The number of rows in the minor matrix.
    /// * `CC`: The number of columns in the minor matrix.
    ///
    /// # Panics:
    ///
    /// This function will panic if the specified row or column is out of bounds,
    /// or if the minor matrix size is invalid.
    ///
    /// # Example:
    ///
    /// ```
    /// use tiny_matrix::mat;
    /// let m = mat![[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]];
    /// let minor = m.minor(0, 0);
    /// assert_eq!(minor, mat![[1.0, 4.0], [6.0, 0.0]]);
    /// ```
    pub fn minor<const RR: usize, const CC: usize>(&self, r: usize, c: usize) -> Matrix<RR, CC, T> {
        // This seems to be evalutaed at compile time ... good!
        if RR + 1 != ROWS || CC + 1 != COLS || r >= ROWS || c >= COLS {
            panic!("Invalid minor size or indices");
        }

        let mut minor = Matrix::<RR, CC, T>::zeros();
        let mut rr = 0;
        for ri in 0..ROWS {
            if ri == r {
                continue;
            };
            let mut cc = 0;
            for ci in 0..COLS {
                if ci == c {
                    continue;
                }
                minor[(rr, cc)] = self[(ri, ci)];
                cc += 1;
            }
            rr += 1;
        }
        minor
    }

    /// Create a new matrix filled with zeros
    ///
    /// # Returns:
    ///
    /// The new matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = Matrix2d::zeros();
    /// assert_eq!(m.shape(), (2, 2));
    /// ```
    pub fn zeros() -> Self {
        let data = [[T::zero(); ROWS]; COLS];
        Self { data }
    }

    /// Create a new matrix filled with ones
    ///
    /// # Returns:
    ///
    /// The new matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = Matrix2d::ones();
    /// assert_eq!(m.shape(), (2, 2));
    /// assert_eq!(m[(0, 0)], 1.0);
    /// ```
    pub fn ones() -> Self {
        let data = [[T::one(); ROWS]; COLS];
        Self { data }
    }

    /// Map elements of a matrix
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = Matrix2d::ones();
    /// let m2 = m.map(|x| x + 1.0);
    /// assert_eq!(m2[(0, 0)], 2.0);
    /// ```
    pub fn map<F, U>(&self, f: F) -> Matrix<ROWS, COLS, U>
    where
        F: Fn(T) -> U,
        U: MatrixElem,
    {
        let mut result = Matrix::<ROWS, COLS, U>::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = f(self[(i, j)]);
            }
        }
        result
    }

    /// Zip two matrices together and perform a map
    ///
    /// # Arguments
    ///
    /// * `other` - The other matrix to zip with.
    /// * `f` - The function to apply to each pair of elements.
    ///
    /// # Returns
    ///
    /// The resulting matrix after applying the function to each pair of elements.
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m1 = Matrix2d::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let m2 = Matrix2d::from([[5.0, 6.0], [7.0, 8.0]]);
    /// let result = m1.zip_map(&m2, |a, b| a + b);
    /// assert_eq!(result, Matrix2d::from([[6.0, 8.0], [10.0, 12.0]]));
    /// ```
    pub fn zip_map<F, U>(&self, other: &Matrix<ROWS, COLS, U>, f: F) -> Matrix<ROWS, COLS, U>
    where
        F: Fn(T, U) -> U,
        U: MatrixElem,
    {
        let mut result = Matrix::<ROWS, COLS, U>::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = f(self[(i, j)], other[(i, j)]);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_matrix_creation() {
        let mat = Matrix3d::zeros();
        assert_eq!(mat.data, [[0.0; 3]; 3]);
    }

    #[test]
    fn test_minor() {
        let mat = mat_row_major![[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]];
        let minor = mat.minor(0, 0);
        assert_eq!(minor, mat_row_major![[1.0, 4.0], [6.0, 0.0]]);
    }
}
