/// Construct a [`Matrix`](crate::Matrix) using MATLAB-like syntax.
///
/// Rows are separated by semicolons, elements by commas.
///
/// ```
/// use numeris::matrix;
///
/// // 2×3 matrix
/// let m = matrix![1.0, 2.0, 3.0; 4.0, 5.0, 6.0];
/// assert_eq!(m[(0, 0)], 1.0);
/// assert_eq!(m[(1, 2)], 6.0);
///
/// // 1×1 matrix
/// let s = matrix![42.0];
/// assert_eq!(s[(0, 0)], 42.0);
/// ```
#[macro_export]
macro_rules! matrix {
    // Multiple rows: a, b; c, d
    ($($($val:expr),+ $(,)?);+ $(;)?) => {
        $crate::Matrix::new([$([$($val),+]),+])
    };
}

/// Construct a column [`Vector`](crate::Vector) (N×1 matrix).
///
/// ```
/// use numeris::vector;
///
/// let v = vector![1.0, 2.0, 3.0];
/// assert_eq!(v[0], 1.0);
/// assert_eq!(v[2], 3.0);
/// ```
#[macro_export]
macro_rules! vector {
    ($($val:expr),+ $(,)?) => {
        $crate::Vector::from_array([$($val),+])
    };
}
