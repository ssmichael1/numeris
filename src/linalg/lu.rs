use crate::linalg::LinalgError;
use crate::matrix::vector::Vector;
use crate::traits::{LinalgScalar, MatrixMut, MatrixRef};
use crate::Matrix;

/// Perform LU decomposition with partial pivoting, in place.
///
/// On return, `a` contains both L and U packed together:
/// - Upper triangle (including diagonal): U
/// - Lower triangle (excluding diagonal): L (diagonal of L is implicitly 1)
///
/// `perm` is filled with the row permutation indices.
/// Returns `true` if the number of row swaps was even.
pub fn lu_in_place<T: LinalgScalar>(
    a: &mut impl MatrixMut<T>,
    perm: &mut [usize],
) -> Result<bool, LinalgError> {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "LU decomposition requires a square matrix");
    assert_eq!(n, perm.len(), "permutation slice length must match matrix size");

    for i in 0..n {
        perm[i] = i;
    }

    let mut even = true;

    for col in 0..n {
        // Partial pivoting: find row with largest modulus in this column
        let mut max_row = col;
        let mut max_val = a.get(col, col).modulus();
        for row in (col + 1)..n {
            let val = a.get(row, col).modulus();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < T::lepsilon() {
            return Err(LinalgError::Singular);
        }

        // Swap rows if needed
        if max_row != col {
            perm.swap(col, max_row);
            for j in 0..n {
                let tmp = *a.get(col, j);
                *a.get_mut(col, j) = *a.get(max_row, j);
                *a.get_mut(max_row, j) = tmp;
            }
            even = !even;
        }

        // Eliminate below pivot, storing L factors in lower triangle
        let pivot = *a.get(col, col);
        for row in (col + 1)..n {
            let factor = *a.get(row, col) / pivot;
            *a.get_mut(row, col) = factor;
            for j in (col + 1)..n {
                let u_val = *a.get(col, j);
                let cur = *a.get(row, j);
                *a.get_mut(row, j) = cur - factor * u_val;
            }
        }
    }

    Ok(even)
}

/// Solve Ax = b given the packed LU decomposition and permutation.
///
/// `lu` is the packed L/U matrix from `lu_in_place`.
/// `perm` is the row permutation from `lu_in_place`.
/// `b` (input) and `x` (output) are separate slices of length n.
pub fn lu_solve<T: LinalgScalar>(
    lu: &impl MatrixRef<T>,
    perm: &[usize],
    b: &[T],
    x: &mut [T],
) {
    let n = lu.nrows();

    // Apply permutation and forward substitution (solve Ly = Pb)
    for i in 0..n {
        let mut sum = b[perm[i]];
        for j in 0..i {
            sum = sum - *lu.get(i, j) * x[j];
        }
        x[i] = sum;
    }

    // Back substitution (solve Ux = y)
    for i in (0..n).rev() {
        let mut sum = x[i];
        for j in (i + 1)..n {
            sum = sum - *lu.get(i, j) * x[j];
        }
        x[i] = sum / *lu.get(i, i);
    }
}

/// LU decomposition of a fixed-size square matrix.
///
/// Stores the packed L/U factors and permutation vector.
/// Use `solve()`, `inverse()`, or `det()` to work with the decomposition.
#[derive(Debug)]
pub struct LuDecomposition<T, const N: usize> {
    lu: Matrix<T, N, N>,
    perm: [usize; N],
    even: bool,
}

impl<T: LinalgScalar, const N: usize> LuDecomposition<T, N> {
    /// Decompose a matrix. Returns an error if the matrix is singular.
    pub fn new(a: &Matrix<T, N, N>) -> Result<Self, LinalgError> {
        let mut lu = *a;
        let mut perm = [0usize; N];
        let even = lu_in_place(&mut lu, &mut perm)?;
        Ok(Self { lu, perm, even })
    }

    /// Solve Ax = b for x.
    pub fn solve(&self, b: &Vector<T, N>) -> Vector<T, N> {
        // Extract b into a flat array
        let mut b_flat = [T::zero(); N];
        for i in 0..N {
            b_flat[i] = b[i];
        }

        let mut x_flat = [T::zero(); N];
        lu_solve(&self.lu, &self.perm, &b_flat, &mut x_flat);

        Vector::from_array(x_flat)
    }

    /// Compute the matrix inverse.
    pub fn inverse(&self) -> Matrix<T, N, N> {
        let mut inv = Matrix::<T, N, N>::zeros();
        let mut col_buf = [T::zero(); N];
        let mut e = [T::zero(); N];

        for col in 0..N {
            // Set up unit vector
            if col > 0 {
                e[col - 1] = T::zero();
            }
            e[col] = T::one();

            lu_solve(&self.lu, &self.perm, &e, &mut col_buf);

            for row in 0..N {
                inv[(row, col)] = col_buf[row];
            }
        }

        inv
    }

    /// Compute the determinant.
    pub fn det(&self) -> T {
        let mut d = if self.even { T::one() } else { T::zero() - T::one() };
        for i in 0..N {
            d = d * self.lu[(i, i)];
        }
        d
    }
}

/// Convenience methods on square matrices.
impl<T: LinalgScalar, const N: usize> Matrix<T, N, N> {
    /// LU decomposition with partial pivoting.
    pub fn lu(&self) -> Result<LuDecomposition<T, N>, LinalgError> {
        LuDecomposition::new(self)
    }

    /// Solve Ax = b for x.
    pub fn solve(&self, b: &Vector<T, N>) -> Result<Vector<T, N>, LinalgError> {
        Ok(self.lu()?.solve(b))
    }

    /// Compute the matrix inverse.
    pub fn inverse(&self) -> Result<Self, LinalgError> {
        Ok(self.lu()?.inverse())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_solve_2x2() {
        // 3x + 2y = 7
        // x + 4y = 9
        let a = Matrix::new([[3.0_f64, 2.0], [1.0, 4.0]]);
        let b = Vector::from_array([7.0, 9.0]);

        let x = a.solve(&b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn lu_solve_3x3() {
        let a = Matrix::new([
            [2.0_f64, 1.0, -1.0],
            [-3.0, -1.0, 2.0],
            [-2.0, 1.0, 2.0],
        ]);
        let b = Vector::from_array([8.0, -11.0, -3.0]);

        let x = a.solve(&b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-12);
        assert!((x[1] - 3.0).abs() < 1e-12);
        assert!((x[2] - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn lu_inverse_2x2() {
        let a = Matrix::new([[4.0_f64, 7.0], [2.0, 6.0]]);
        let a_inv = a.inverse().unwrap();
        let id = a * a_inv;

        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((id[(i, j)] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn lu_inverse_3x3() {
        let a = Matrix::new([
            [1.0_f64, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [5.0, 6.0, 0.0],
        ]);
        let a_inv = a.inverse().unwrap();
        let id = a * a_inv;

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (id[(i, j)] - expected).abs() < 1e-10,
                    "id[({},{})] = {}, expected {}",
                    i,
                    j,
                    id[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn lu_det() {
        let a = Matrix::new([[3.0_f64, 8.0], [4.0, 6.0]]);
        let lu = a.lu().unwrap();
        assert!((lu.det() - (-14.0)).abs() < 1e-12);
    }

    #[test]
    fn lu_det_3x3() {
        let a = Matrix::new([
            [6.0_f64, 1.0, 1.0],
            [4.0, -2.0, 5.0],
            [2.0, 8.0, 7.0],
        ]);
        let lu = a.lu().unwrap();
        assert!((lu.det() - (-306.0)).abs() < 1e-10);
    }

    #[test]
    fn lu_singular() {
        let a = Matrix::new([[1.0_f64, 2.0], [2.0, 4.0]]);
        assert_eq!(a.lu().unwrap_err(), LinalgError::Singular);
    }

    #[test]
    fn lu_in_place_generic() {
        // Verify the free function works via the MatrixMut trait
        let mut a = Matrix::new([[2.0_f64, 1.0], [4.0, 3.0]]);
        let mut perm = [0usize; 2];
        let result = lu_in_place(&mut a, &mut perm);
        assert!(result.is_ok());
    }

    #[test]
    fn solve_verify_residual() {
        // Solve and verify A*x == b by computing residual row-by-row
        let a = Matrix::new([
            [1.0_f64, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [2.0, 6.0, 4.0, 1.0],
            [3.0, 1.0, 9.0, 2.0],
        ]);
        let b = Vector::from_array([10.0, 26.0, 13.0, 15.0]);

        let x = a.solve(&b).unwrap();

        // Check each row: sum_j(a[i][j] * x[j]) == b[i]
        for i in 0..4 {
            let mut row_sum = 0.0;
            for j in 0..4 {
                row_sum += a[(i, j)] * x[j];
            }
            assert!(
                (row_sum - b[i]).abs() < 1e-10,
                "residual[{}] = {}",
                i,
                row_sum - b[i]
            );
        }
    }
}
