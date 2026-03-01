use crate::linalg::{split_two_col_slices, LinalgError};
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

        // Column-major LAPACK dgetf2 style elimination:
        // 1. Scale sub-column by 1/pivot (contiguous in col-major)
        // 2. AXPY: for each column j > col, a[col+1:n, j] -= a[col, j] * a[col+1:n, col]
        //    Both source and destination are contiguous column slices → SIMD-friendly.
        let pivot = *a.get(col, col);
        let inv_pivot = T::one() / pivot;

        // Scale sub-column: a[col+1:n, col] /= pivot
        {
            let sub_col = a.col_as_mut_slice(col, col + 1);
            for x in sub_col.iter_mut() {
                *x = *x * inv_pivot;
            }
        }

        // Rank-1 update: a[col+1:n, j] -= a[col, j] * a[col+1:n, col]
        for j in (col + 1)..n {
            let a_col_j = *a.get(col, j);
            // Both slices are contiguous column data — use SIMD AXPY
            let (left, right) = split_two_col_slices(a, col, j, col + 1);
            crate::simd::axpy_neg_dispatch(right, a_col_j, left);
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

/// LU decomposition in place using direct array access (no trait dispatch).
///
/// Optimized for small N (≤6) where the compiler fully unrolls all loops.
/// Uses `data[col][row]` access directly, bypassing `MatrixMut` trait methods
/// and SIMD dispatch overhead.
#[inline(always)]
fn lu_in_place_small<T: LinalgScalar, const N: usize>(
    data: &mut [[T; N]; N],
    perm: &mut [usize; N],
) -> Result<bool, LinalgError> {
    for i in 0..N {
        perm[i] = i;
    }

    let mut even = true;

    for col in 0..N {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = data[col][col].modulus();
        for row in (col + 1)..N {
            let val = data[col][row].modulus();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < T::lepsilon() {
            return Err(LinalgError::Singular);
        }

        if max_row != col {
            perm.swap(col, max_row);
            for j in 0..N {
                let tmp = data[j][col];
                data[j][col] = data[j][max_row];
                data[j][max_row] = tmp;
            }
            even = !even;
        }

        let pivot = data[col][col];
        let inv_pivot = T::one() / pivot;

        // Scale sub-column
        for i in (col + 1)..N {
            data[col][i] = data[col][i] * inv_pivot;
        }

        // Rank-1 update
        for j in (col + 1)..N {
            let factor = data[j][col];
            for i in (col + 1)..N {
                data[j][i] = data[j][i] - factor * data[col][i];
            }
        }
    }

    Ok(even)
}

/// LU solve using direct array access (no trait dispatch).
///
/// Optimized for small N (≤6). The LU data is `data[col][row]` column-major.
#[inline(always)]
fn lu_solve_small<T: LinalgScalar, const N: usize>(
    data: &[[T; N]; N],
    perm: &[usize; N],
    b: &[T; N],
    x: &mut [T; N],
) {
    // Forward substitution: solve Ly = Pb
    for i in 0..N {
        let mut sum = b[perm[i]];
        for j in 0..i {
            sum = sum - data[j][i] * x[j];
        }
        x[i] = sum;
    }

    // Back substitution: solve Ux = y
    for i in (0..N).rev() {
        let mut sum = x[i];
        for j in (i + 1)..N {
            sum = sum - data[j][i] * x[j];
        }
        x[i] = sum / data[i][i];
    }
}

/// LU decomposition of a fixed-size square matrix.
///
/// Stores the packed L/U factors and permutation vector.
/// Use `solve()`, `inverse()`, or `det()` to work with the decomposition.
///
/// # Example
///
/// ```
/// use numeris::{Matrix, Vector};
///
/// let a = Matrix::new([[2.0_f64, 1.0], [5.0, 3.0]]);
/// let lu = a.lu().unwrap();
///
/// let b = Vector::from_array([4.0, 11.0]);
/// let x = lu.solve(&b);
/// assert!((x[0] - 1.0).abs() < 1e-12);
/// assert!((x[1] - 2.0).abs() < 1e-12);
///
/// let det = lu.det();
/// assert!((det - 1.0).abs() < 1e-12);
/// ```
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
        let even = if N <= 6 {
            lu_in_place_small(&mut lu.data, &mut perm)?
        } else {
            lu_in_place(&mut lu, &mut perm)?
        };
        Ok(Self { lu, perm, even })
    }

    /// Solve Ax = b for x.
    pub fn solve(&self, b: &Vector<T, N>) -> Vector<T, N> {
        let mut b_flat = [T::zero(); N];
        for i in 0..N {
            b_flat[i] = b[i];
        }

        let mut x_flat = [T::zero(); N];
        if N <= 6 {
            lu_solve_small(&self.lu.data, &self.perm, &b_flat, &mut x_flat);
        } else {
            lu_solve(&self.lu, &self.perm, &b_flat, &mut x_flat);
        }

        Vector::from_array(x_flat)
    }

    /// Compute the matrix inverse.
    pub fn inverse(&self) -> Matrix<T, N, N> {
        let mut inv = Matrix::<T, N, N>::zeros();
        let mut e = [T::zero(); N];
        let mut col_buf = [T::zero(); N];

        for col in 0..N {
            if col > 0 {
                e[col - 1] = T::zero();
            }
            e[col] = T::one();

            if N <= 6 {
                lu_solve_small(&self.lu.data, &self.perm, &e, &mut col_buf);
            } else {
                lu_solve(&self.lu, &self.perm, &e, &mut col_buf);
            }

            for row in 0..N {
                inv.data[col][row] = col_buf[row];
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

/// Direct inverse for small matrices (N <= 4) using closed-form adjugate formulas.
///
/// Returns `Err(Singular)` if the determinant modulus is below machine epsilon.
#[inline(always)]
fn inverse_direct<T: LinalgScalar, const N: usize>(
    m: &Matrix<T, N, N>,
) -> Result<Matrix<T, N, N>, LinalgError> {
    // Column-major: m.data[col][row]
    if N == 1 {
        let a = m.data[0][0];
        if a.modulus() < T::lepsilon() {
            return Err(LinalgError::Singular);
        }
        let mut out = Matrix::<T, N, N>::zeros();
        out.data[0][0] = T::one() / a;
        return Ok(out);
    }
    if N == 2 {
        let a = m.data[0][0]; let b = m.data[1][0];
        let c = m.data[0][1]; let d = m.data[1][1];
        let det = a * d - b * c;
        if det.modulus() < T::lepsilon() {
            return Err(LinalgError::Singular);
        }
        let inv_det = T::one() / det;
        let mut out = Matrix::<T, N, N>::zeros();
        out.data[0][0] = d * inv_det;
        out.data[1][0] = (T::zero() - b) * inv_det;
        out.data[0][1] = (T::zero() - c) * inv_det;
        out.data[1][1] = a * inv_det;
        return Ok(out);
    }
    if N == 3 {
        let a = m.data[0][0]; let b = m.data[1][0]; let c = m.data[2][0];
        let d = m.data[0][1]; let e = m.data[1][1]; let f = m.data[2][1];
        let g = m.data[0][2]; let h = m.data[1][2]; let i = m.data[2][2];

        // Cofactors (adjugate transposed)
        let c00 = e * i - f * h;
        let c01 = f * g - d * i;
        let c02 = d * h - e * g;
        let c10 = c * h - b * i;
        let c11 = a * i - c * g;
        let c12 = b * g - a * h;
        let c20 = b * f - c * e;
        let c21 = c * d - a * f;
        let c22 = a * e - b * d;

        let det = a * c00 + b * c01 + c * c02;
        if det.modulus() < T::lepsilon() {
            return Err(LinalgError::Singular);
        }
        let inv_det = T::one() / det;

        let mut out = Matrix::<T, N, N>::zeros();
        out.data[0][0] = c00 * inv_det; out.data[1][0] = c10 * inv_det; out.data[2][0] = c20 * inv_det;
        out.data[0][1] = c01 * inv_det; out.data[1][1] = c11 * inv_det; out.data[2][1] = c21 * inv_det;
        out.data[0][2] = c02 * inv_det; out.data[1][2] = c12 * inv_det; out.data[2][2] = c22 * inv_det;
        return Ok(out);
    }
    if N == 4 {
        let a00 = m.data[0][0]; let a01 = m.data[1][0]; let a02 = m.data[2][0]; let a03 = m.data[3][0];
        let a10 = m.data[0][1]; let a11 = m.data[1][1]; let a12 = m.data[2][1]; let a13 = m.data[3][1];
        let a20 = m.data[0][2]; let a21 = m.data[1][2]; let a22 = m.data[2][2]; let a23 = m.data[3][2];
        let a30 = m.data[0][3]; let a31 = m.data[1][3]; let a32 = m.data[2][3]; let a33 = m.data[3][3];

        // 2x2 sub-determinants from rows 0-1
        let s0 = a00 * a11 - a01 * a10;
        let s1 = a00 * a12 - a02 * a10;
        let s2 = a00 * a13 - a03 * a10;
        let s3 = a01 * a12 - a02 * a11;
        let s4 = a01 * a13 - a03 * a11;
        let s5 = a02 * a13 - a03 * a12;

        // 2x2 sub-determinants from rows 2-3
        let c0 = a20 * a31 - a21 * a30;
        let c1 = a20 * a32 - a22 * a30;
        let c2 = a20 * a33 - a23 * a30;
        let c3 = a21 * a32 - a22 * a31;
        let c4 = a21 * a33 - a23 * a31;
        let c5 = a22 * a33 - a23 * a32;

        let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
        if det.modulus() < T::lepsilon() {
            return Err(LinalgError::Singular);
        }
        let inv_det = T::one() / det;

        let mut out = Matrix::<T, N, N>::zeros();
        // Adjugate / det — each element is a 3x3 cofactor
        out.data[0][0] = ( a11 * c5 - a12 * c4 + a13 * c3) * inv_det;
        out.data[1][0] = (T::zero() - a01 * c5 + a02 * c4 - a03 * c3) * inv_det;
        out.data[2][0] = ( a31 * s5 - a32 * s4 + a33 * s3) * inv_det;
        out.data[3][0] = (T::zero() - a21 * s5 + a22 * s4 - a23 * s3) * inv_det;

        out.data[0][1] = (T::zero() - a10 * c5 + a12 * c2 - a13 * c1) * inv_det;
        out.data[1][1] = ( a00 * c5 - a02 * c2 + a03 * c1) * inv_det;
        out.data[2][1] = (T::zero() - a30 * s5 + a32 * s2 - a33 * s1) * inv_det;
        out.data[3][1] = ( a20 * s5 - a22 * s2 + a23 * s1) * inv_det;

        out.data[0][2] = ( a10 * c4 - a11 * c2 + a13 * c0) * inv_det;
        out.data[1][2] = (T::zero() - a00 * c4 + a01 * c2 - a03 * c0) * inv_det;
        out.data[2][2] = ( a30 * s4 - a31 * s2 + a33 * s0) * inv_det;
        out.data[3][2] = (T::zero() - a20 * s4 + a21 * s2 - a23 * s0) * inv_det;

        out.data[0][3] = (T::zero() - a10 * c3 + a11 * c1 - a12 * c0) * inv_det;
        out.data[1][3] = ( a00 * c3 - a01 * c1 + a02 * c0) * inv_det;
        out.data[2][3] = (T::zero() - a30 * s3 + a31 * s1 - a32 * s0) * inv_det;
        out.data[3][3] = ( a20 * s3 - a21 * s1 + a22 * s0) * inv_det;

        return Ok(out);
    }
    // Should not be called for N > 4, but fall back to LU
    Ok(LuDecomposition::new(m)?.inverse())
}

/// Convenience methods on square matrices.
impl<T: LinalgScalar, const N: usize> Matrix<T, N, N> {
    /// LU decomposition with partial pivoting.
    pub fn lu(&self) -> Result<LuDecomposition<T, N>, LinalgError> {
        LuDecomposition::new(self)
    }

    /// Solve `Ax = b` for `x` via LU decomposition.
    ///
    /// ```
    /// use numeris::{Matrix, Vector};
    /// let a = Matrix::new([
    ///     [2.0_f64, 1.0, -1.0],
    ///     [-3.0, -1.0, 2.0],
    ///     [-2.0, 1.0, 2.0],
    /// ]);
    /// let b = Vector::from_array([8.0, -11.0, -3.0]);
    /// let x = a.solve(&b).unwrap();
    /// assert!((x[0] - 2.0).abs() < 1e-12);
    /// assert!((x[1] - 3.0).abs() < 1e-12);
    /// assert!((x[2] - (-1.0)).abs() < 1e-12);
    /// ```
    pub fn solve(&self, b: &Vector<T, N>) -> Result<Vector<T, N>, LinalgError> {
        Ok(self.lu()?.solve(b))
    }

    /// Compute the matrix inverse via direct formulas (N<=4) or LU decomposition.
    ///
    /// ```
    /// use numeris::Matrix;
    /// let a = Matrix::new([[4.0_f64, 7.0], [2.0, 6.0]]);
    /// let a_inv = a.inverse().unwrap();
    /// let id = a * a_inv;
    /// assert!((id[(0, 0)] - 1.0).abs() < 1e-12);
    /// assert!((id[(0, 1)]).abs() < 1e-12);
    /// ```
    pub fn inverse(&self) -> Result<Self, LinalgError> {
        if N <= 4 {
            return inverse_direct(self);
        }
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
