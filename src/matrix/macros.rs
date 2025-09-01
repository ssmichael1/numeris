/// Macro for matrix creation
/// Creates a new matrix with the given column-major data.
#[macro_export]
macro_rules! mat {
    ($($row:expr),* $(,)?) => {{
        let data = [$($row),*];
        $crate::matrix::Matrix::new(data)
    }};
}

/// Macro for matrix creation
/// Creates a new matrix with the given column-major data
///
/// # Example
/// ```
/// use numeris::prelude::*;
/// let m = mat![[1.0, 2.0], [3.0, 4.0]];
/// assert!(m == Matrix::new([[1.0, 2.0], [3.0, 4.0]]));
/// ```
#[macro_export]
macro_rules! mat_col_major {
    ($($row:expr),* $(,)?) => {{
        let data = [$($row),*];
        $crate::Matrix::new(data)
    }};
}

/// Macro for matrix creation
/// Creates a new matrix with the given row-major data
///
/// # Example
/// ```
/// use numeris::prelude::*;
/// let m = mat_row_major![[1.0, 2.0], [3.0, 4.0]];
/// // Compare with explicitly created column-major matrix
/// assert!(m == Matrix::new([[1.0, 3.0], [2.0, 4.0]]));
/// ```
#[macro_export]
macro_rules! mat_row_major {
    ($($row:expr),* $(,)?) => {{
        let data = [$($row),*];
        $crate::matrix::Matrix::new(data).transpose()
    }};
}

/// Macro for vector creation (a row matrix)
/// Creates a new vector (row matrix) with the given data.
///
/// # Example
///
/// ```rust
/// use numeris::prelude::*;
/// let v = rowmat![1.0, 2.0, 3.0];
/// assert!(v == Matrix::new([[1.0, 2.0, 3.0]]));
/// ```
#[macro_export]
macro_rules! rowmat {
    // Accepts vector!([1.0, 2.0, 3.0])
    ([$($elem:expr),* $(,)?]) => {{
        $crate::matrix::Matrix::new([[$($elem),*]])
    }};
    // Accepts vector!(1.0, 2.0, 3.0)
    ($($elem:expr),* $(,)?) => {{
        $crate::matrix::Matrix::new([[$($elem),*]])
    }};
}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    #[test]
    fn test_matrix_creation() {
        let m = mat![[1.0, 2.0], [3.0, 4.0]];
        let expected = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(m, expected);
    }

    #[test]
    fn test_vector_creation() {
        let v = rowmat![1.0, 2.0, 3.0];
        let expected = Matrix::new([[1.0, 2.0, 3.0]]);
        assert_eq!(v, expected);
    }
}
