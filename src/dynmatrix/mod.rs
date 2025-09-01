//! Dynamically-sized matrices
//!
//! This module provides a `DynMatrix` type for working with matrices of
//! arbitrary size at runtime.
//!

mod dynmat_display;
mod dynmat_err;
mod dynmat_ops;

use dynmat_err::*;

/// Dynamically-sized matrices
///
/// # Overview
///
/// The `DynMatrix` type provides a flexible and efficient way to work with
/// matrices of arbitrary size at runtime. It is designed for ease of use
/// and performance, making it suitable for a wide range of applications.
///
/// # Features
///
/// - Dynamic sizing: Create matrices of any size without knowing the
///   dimensions at compile time.
/// - Efficient memory management: The matrix data is stored in a
///   contiguous vector, ensuring good cache performance.
/// - Comprehensive API: The `DynMatrix` type implements a wide range
///   of matrix operations, making it easy to perform common tasks.
///
/// # Notes
///
/// - common operators (+, -, / *) act element-wise on the array if the right-hand
///   side is another array.  Because the sizes must match, the operators return a `Result`
///   indicating success or failure.
///
/// - for operators with scalars, the operators always succeed and return a new array.
///
/// # Example
///
/// ```
/// use tiny_matrix::prelude::*;
/// let m = DynMatrix::<f64>::zeros(2, 3);
/// assert_eq!(m.shape(), (2, 3));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DynMatrix<T>
where
    T: crate::matrix::MatrixElem,
{
    pub(crate) data: Vec<T>,
    pub(crate) rows_: usize,
    pub(crate) cols_: usize,
}

// Implement MatrixTrait for DynMatrix<T>
impl<T> DynMatrix<T>
where
    T: crate::matrix::MatrixElem,
{
    /// Returns the number of rows in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = DynMatrix::<f64>::ones(2, 3);
    /// assert_eq!(m.rows(), 2);
    /// ```
    pub fn rows(&self) -> usize {
        self.rows_
    }

    /// Returns the number of columns in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = DynMatrix::<f64>::ones(2, 3);
    /// assert_eq!(m.cols(), 3);
    /// ```
    pub fn cols(&self) -> usize {
        self.cols_
    }

    /// Returns the shape of the matrix as (rows, cols).
    ///
    /// # Examples
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = DynMatrix::<f64>::ones(2, 3);
    /// assert_eq!(m.shape(), (2, 3));
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    /// Creates a new `DynMatrix` from the given data, rows, and columns.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector containing the matrix elements in column-major order.
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// A new `DynMatrix` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = DynMatrix::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// assert_eq!(m.shape(), (2, 2));
    /// ```
    pub fn new(data: Vec<T>, rows: usize, cols: usize) -> DynMatrixResult<Self> {
        if data.len() != rows * cols {
            return Err(DynMatrixError::InvalidShape);
        }
        Ok(Self {
            data,
            rows_: rows,
            cols_: cols,
        })
    }

    /// Creates a new `DynMatrix` from the given data in row-major order.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector containing the matrix elements in row-major order.
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// A new `DynMatrix` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = DynMatrix::<f64>::from_vec_row_major(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// assert_eq!(m.shape(), (2, 2));
    /// ```
    pub fn from_vec_row_major(data: Vec<T>, rows: usize, cols: usize) -> DynMatrixResult<Self> {
        if data.len() != rows * cols {
            return Err(DynMatrixError::InvalidShape);
        }
        Ok(Self {
            data,
            rows_: cols,
            cols_: rows,
        }
        .transpose())
    }

    /// Create a new `DynMatrix` filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// A new `DynMatrix` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = DynMatrix::<f64>::zeros(2, 3);
    /// assert_eq!(m.shape(), (2, 3));
    /// ```
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![T::zero(); rows * cols],
            rows_: rows,
            cols_: cols,
        }
    }

    /// Creates a new `DynMatrix` filled with ones.
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// A new `DynMatrix` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = DynMatrix::<f64>::ones(2, 3);
    /// assert_eq!(m.shape(), (2, 3));
    /// ```
    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![T::one(); rows * cols],
            rows_: rows,
            cols_: cols,
        }
    }

    /// Take matrix transpose
    ///
    /// # Returns
    ///
    /// A new `DynMatrix` instance representing the transposed matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = DynMatrix::<f64>::ones(2, 3);
    /// let m_t = m.transpose();
    /// assert_eq!(m_t.shape(), (3, 2));
    /// ```
    pub fn transpose(&self) -> DynMatrix<T> {
        let mut transposed = Vec::with_capacity(self.data.len());
        for col in 0..self.cols_ {
            for row in 0..self.rows_ {
                transposed.push(self.data[col * self.rows_ + row]);
            }
        }
        DynMatrix {
            data: transposed,
            rows_: self.cols_,
            cols_: self.rows_,
        }
    }

    #[inline(always)]
    fn flat_index(&self, row: usize, col: usize) -> DynMatrixResult<usize> {
        if row >= self.rows() || col >= self.cols() {
            return Err(DynMatrixError::IndexOutOfBounds);
        }
        Ok(col * self.rows_ + row)
    }

    /// Returns a reference to the element at the specified position.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index of the element.
    /// * `col` - The column index of the element.
    ///
    /// # Returns
    ///
    /// A reference to the element at the specified position, or an error if the position is out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = DynMatrix::<f64>::ones(2, 2);
    /// let elem = m.at(0, 0);
    /// assert_eq!(elem, Ok(&1.0));
    /// ```
    pub fn at(&self, row: usize, col: usize) -> DynMatrixResult<&T> {
        let flat_index = self.flat_index(row, col)?;
        Ok(&self.data[flat_index])
    }

    /// Returns a mutable reference to the element at the specified position.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index of the element.
    /// * `col` - The column index of the element.
    ///
    /// # Returns
    ///
    /// A mutable reference to the element at the specified position, or an error if the position is out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let mut m = DynMatrix::<f64>::ones(2, 2);
    /// let elem = m.at_mut(0, 0);
    /// assert_eq!(elem, Ok(&mut 1.0));
    /// ```
    pub fn at_mut(&mut self, row: usize, col: usize) -> DynMatrixResult<&mut T> {
        let flat_index = self.flat_index(row, col)?;
        Ok(&mut self.data[flat_index])
    }

    /// Create a dynamic matrix from function of matrix indices
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    /// * `f` - A closure that takes the row and column indices as arguments and returns the value for that position.
    ///
    /// # Returns
    ///
    /// A new `DynMatrix` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = DynMatrix::from_fn(2, 2, |row, col| (row + col) as f64);
    /// assert_eq!(m[(0, 0)], 0.0);
    /// assert_eq!(m[(0, 1)], 1.0);
    /// assert_eq!(m[(1, 0)], 1.0);
    /// assert_eq!(m[(1, 1)], 2.0);
    /// ```
    pub fn from_fn<F>(rows: usize, cols: usize, f: F) -> Self
    where
        F: Fn(usize, usize) -> T,
    {
        let mut data = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                data.push(f(row, col));
            }
        }
        Self {
            data,
            rows_: rows,
            cols_: cols,
        }
    }

    /// Create a dynamic matrix from an iterator
    ///
    /// # Errors
    ///
    /// Returns an error if the iterator does not produce enough elements to fill the matrix.
    ///
    /// # Arguments:
    ///
    /// * `iter` - An iterator that produces elements of type T. Iteration is column major.
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// A new `DynMatrix` instance if the iterator produces enough elements, or an error if it does not.
    ///
    /// # Example:
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = DynMatrix::from_iter(0..4, 2, 2).unwrap();
    /// assert_eq!(m.shape(), (2, 2));
    /// ```
    pub fn from_iter<I>(iter: I, rows: usize, cols: usize) -> DynMatrixResult<Self>
    where
        I: IntoIterator<Item = T>,
    {
        let data: Vec<T> = iter.into_iter().collect();
        if data.len() != rows * cols {
            return Err(DynMatrixError::InvalidShape);
        }
        Ok(Self {
            data,
            rows_: rows,
            cols_: cols,
        })
    }

    /// Element-wise mapping of matrix elements
    ///
    /// # Arguments
    ///
    /// * `f` - A closure that takes a reference to an element of the matrix and returns a new element.
    ///
    /// # Returns
    ///
    /// A new `DynMatrix` instance with the mapped elements.
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = DynMatrix::<f64>::ones(2, 2);
    /// let m2 = m.map(|x| x * 2.0);
    /// assert_eq!(m2[(0,0)], 2.0);
    /// ```
    pub fn map<F, R>(&self, mut f: F) -> DynMatrix<R>
    where
        F: FnMut(&T) -> R,
        R: crate::matrix::MatrixElem,
    {
        let mut result = Vec::with_capacity(self.data.len());
        for elem in &self.data {
            result.push(f(elem));
        }
        DynMatrix {
            data: result,
            rows_: self.rows_,
            cols_: self.cols_,
        }
    }

    /// Matrix minor
    ///
    /// # Arguments
    ///
    /// * `row` - The row index of the element to remove.
    /// * `col` - The column index of the element to remove.
    ///
    /// # Returns
    ///
    /// The minor of the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = DynMatrix::from_fn(3, 3, |row, col| (row + col) as f64);
    /// let minor = m.minor(1, 1).unwrap();
    /// assert_eq!(minor.shape(), (2, 2));
    /// ```
    pub fn minor(&self, row: usize, col: usize) -> DynMatrixResult<Self> {
        let mut minor = Vec::with_capacity((self.rows_ - 1) * (self.cols_ - 1));
        for r in 0..self.rows_ {
            for c in 0..self.cols_ {
                if r != row && c != col {
                    minor.push(self.at(r, c)?);
                }
            }
        }
        Ok(DynMatrix {
            data: minor.into_iter().cloned().collect(),
            rows_: self.rows_ - 1,
            cols_: self.cols_ - 1,
        })
    }

    /// Compute the determinant of the matrix
    ///
    /// # Returns
    ///
    /// The determinant of the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = DynMatrix::from_fn(2, 2, |row, col| (row + col) as f64);
    /// assert_eq!(m.determinant(), Ok(-1.0));
    /// ```
    pub fn determinant(&self) -> DynMatrixResult<T> {
        if self.rows_ != self.cols_ {
            Err(DynMatrixError::MustBeSquare)
        } else if self.rows_ == 1 {
            Ok(self.data[0])
        }
        // Base case for 2x2 matrix
        else if self.rows_ == 2 {
            Ok(*self.at(0, 0)? * *self.at(1, 1)? - *self.at(0, 1)? * *self.at(1, 0)?)
        }
        // We'll also explicitly do 3x3
        else if self.rows_ == 3 {
            Ok(*self.at(0, 0)? * *self.at(1, 1)? * *self.at(2, 2)?
                + *self.at(0, 1)? * *self.at(1, 2)? * *self.at(2, 0)?
                + *self.at(0, 2)? * *self.at(1, 0)? * *self.at(2, 1)?
                - *self.at(0, 2)? * *self.at(1, 1)? * *self.at(2, 0)?
                - *self.at(0, 0)? * *self.at(1, 2)? * *self.at(2, 1)?
                - *self.at(0, 1)? * *self.at(1, 0)? * *self.at(2, 2)?)
        } else {
            // Recursive case for larger matrices
            let mut det = T::zero();
            for col in 0..self.cols_ {
                let sign = if col % 2 == 0 { T::one() } else { -T::one() };
                let minor = self.minor(0, col)?;
                det += sign * *self.at(0, col)? * minor.determinant()?;
            }
            Ok(det)
        }
    }

    /// Explicitly compute matrix inverse
    ///
    /// # Returns
    ///
    /// The inverse of the matrix, if it exists.
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = DynMatrix::from_vec_row_major(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// assert_eq!(m.inverse(), Ok(DynMatrix::from_vec_row_major(vec![-2.0, 1.0, 1.5, -0.5], 2, 2).unwrap()));
    /// ```
    ///
    pub fn inverse(&self) -> DynMatrixResult<Self> {
        if self.rows_ != self.cols_ {
            return Err(DynMatrixError::MustBeSquare);
        }
        let det = self.determinant()?;
        if det.is_zero() {
            return Err(DynMatrixError::SingularMatrix);
        }
        let mut adjugate = DynMatrix::zeros(self.rows_, self.cols_);
        for row in 0..self.rows_ {
            for col in 0..self.cols_ {
                let minor = self.minor(row, col)?;
                adjugate[(row, col)] = minor.determinant()?
                    * if (row + col) % 2 == 0 {
                        T::one()
                    } else {
                        -T::one()
                    };
            }
        }
        Ok(adjugate * (T::one() / det))
    }

    /// Consume self and convert to a dynamic array
    ///
    /// # Returns
    ///
    /// A 2D dynamic array representation of the matrix.
    /// (arrays support more element-wise operations)
    ///
    /// # Example
    ///
    /// ```
    /// use tiny_matrix::prelude::*;
    /// let m = DynMatrix::from_fn(2, 2, |row, col| (row + col) as f64);
    /// let arr = m.as_dynarray();
    /// assert_eq!(arr.shape(), &[2, 2]);
    /// ```
    pub fn as_dynarray(self) -> crate::prelude::DynArray<T> {
        crate::prelude::DynArray {
            data: self.data,
            shape_: vec![self.rows_, self.cols_],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_determinant() {
        let m = DynMatrix::from_fn(2, 2, |row, col| (row + col) as f64);
        assert_eq!(m.determinant(), Ok(-1.0));
    }

    #[test]
    fn test_inverse() {
        let m = DynMatrix::from_vec_row_major(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let inv = m.inverse().unwrap();
        assert_eq!(
            inv,
            DynMatrix::from_vec_row_major(vec![-2.0, 1.0, 1.5, -0.5], 2, 2).unwrap()
        );
    }
}
