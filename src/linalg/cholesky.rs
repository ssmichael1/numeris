use crate::linalg::LinalgError;
use crate::matrix::vector::Vector;
use crate::traits::{LinalgScalar, MatrixMut, MatrixRef};
use crate::Matrix;

/// Cholesky decomposition in place: A = L * L^H.
///
/// For real matrices, L^H = L^T (standard Cholesky).
/// For complex matrices, this is the Hermitian Cholesky decomposition.
///
/// On return, the lower triangle of `a` (including diagonal) contains L.
/// The upper triangle is left unchanged.
///
/// Returns an error if the matrix is not (Hermitian) positive definite.
///
/// Uses a left-looking column algorithm with SIMD-accelerated AXPY for
/// the rank-1 column updates (same pattern as LU elimination).
pub fn cholesky_in_place<T: LinalgScalar>(a: &mut impl MatrixMut<T>) -> Result<(), LinalgError> {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "Cholesky decomposition requires a square matrix");

    for j in 0..n {
        // Left-looking: subtract contributions from previous columns.
        // a[j:n, j] -= conj(L[j, k]) * L[j:n, k]  for each k < j.
        // This is an AXPY on contiguous column slices.
        for k in 0..j {
            let ljk_conj = (*a.get(j, k)).conj();
            let (col_j, col_k) = super::split_two_col_slices(a, j, k, j);
            crate::simd::axpy_neg_dispatch(col_j, ljk_conj, col_k);
        }

        // Diagonal: a[j,j] has been updated by the AXPY loop above
        let diag = *a.get(j, j);
        if diag.re() <= <T::Real as num_traits::Zero>::zero() {
            return Err(LinalgError::NotPositiveDefinite);
        }
        let ljj = diag.re().lsqrt();
        let ljj_t = T::from_real(ljj);
        *a.get_mut(j, j) = ljj_t;

        // Scale below-diagonal: L[j+1:n, j] /= L[j, j]
        let inv_ljj = T::one() / ljj_t;
        let col = a.col_as_mut_slice(j, j + 1);
        for x in col.iter_mut() {
            *x = *x * inv_ljj;
        }
    }

    Ok(())
}

/// Solve L*x = b by forward substitution, where L is lower triangular.
///
/// `l` contains L in its lower triangle (as produced by `cholesky_in_place`).
/// `b` is the right-hand side, `x` receives the solution.
pub fn forward_substitute<T: LinalgScalar>(
    l: &impl MatrixRef<T>,
    b: &[T],
    x: &mut [T],
) {
    let n = l.nrows();
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum = sum - *l.get(i, j) * x[j];
        }
        x[i] = sum / *l.get(i, i);
    }
}

/// Solve L^H * x = b by back substitution, where L is lower triangular.
///
/// For real matrices, L^H = L^T. For complex matrices, this uses
/// the conjugate transpose.
///
/// `l` contains L in its lower triangle (as produced by `cholesky_in_place`).
/// `b` is the right-hand side, `x` receives the solution.
pub fn back_substitute_lt<T: LinalgScalar>(
    l: &impl MatrixRef<T>,
    b: &[T],
    x: &mut [T],
) {
    let n = l.nrows();
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum = sum - (*l.get(j, i)).conj() * x[j]; // L^H[i][j] = conj(L[j][i])
        }
        x[i] = sum / (*l.get(i, i)).conj(); // L diagonal is real, so conj is identity
    }
}

/// Cholesky decomposition in place using direct array access (no trait dispatch).
///
/// Optimized for small N (≤6) where the compiler fully unrolls all loops.
/// Same left-looking column algorithm but with `data[col][row]` access directly.
#[inline(always)]
fn cholesky_in_place_small<T: LinalgScalar, const N: usize>(
    data: &mut [[T; N]; N],
) -> Result<(), LinalgError> {
    for j in 0..N {
        // Left-looking: subtract contributions from previous columns
        for k in 0..j {
            let ljk = data[k][j].conj();
            for i in j..N {
                data[j][i] = data[j][i] - ljk * data[k][i];
            }
        }

        // Diagonal check and sqrt
        let diag = data[j][j];
        if diag.re() <= <T::Real as num_traits::Zero>::zero() {
            return Err(LinalgError::NotPositiveDefinite);
        }
        let ljj = diag.re().lsqrt();
        let ljj_t = T::from_real(ljj);
        data[j][j] = ljj_t;

        // Scale below-diagonal
        let inv_ljj = T::one() / ljj_t;
        for i in (j + 1)..N {
            data[j][i] = data[j][i] * inv_ljj;
        }
    }

    Ok(())
}

/// Forward substitution using direct array access for small matrices.
#[inline(always)]
fn forward_substitute_small<T: LinalgScalar, const N: usize>(
    data: &[[T; N]; N],
    b: &[T; N],
    x: &mut [T; N],
) {
    for i in 0..N {
        let mut sum = b[i];
        for j in 0..i {
            sum = sum - data[j][i] * x[j];
        }
        x[i] = sum / data[i][i];
    }
}

/// Back substitution L^H x = b using direct array access for small matrices.
#[inline(always)]
fn back_substitute_lt_small<T: LinalgScalar, const N: usize>(
    data: &[[T; N]; N],
    b: &[T; N],
    x: &mut [T; N],
) {
    for i in (0..N).rev() {
        let mut sum = b[i];
        for j in (i + 1)..N {
            sum = sum - data[i][j].conj() * x[j]; // L^H[i][j] = conj(L[j][i]) = conj(data[i][j] in col-major... wait
            // L is lower triangular stored in data[col][row] where col <= row
            // L^H[i][j] = conj(L[j][i]) = conj(data[i][j])
        }
        x[i] = sum / data[i][i].conj();
    }
}

/// Cholesky decomposition of a fixed-size (Hermitian) positive-definite matrix.
///
/// Stores the lower triangular factor L where `A = L * L^H`.
/// For real matrices, `L^H = L^T`.
///
/// # Example
///
/// ```
/// use numeris::{Matrix, Vector};
///
/// let a = Matrix::new([[4.0_f64, 2.0], [2.0, 3.0]]);
/// let chol = a.cholesky().unwrap();
///
/// let b = Vector::from_array([8.0, 7.0]);
/// let x = chol.solve(&b); // solve Ax = b
///
/// let inv = chol.inverse(); // A^{-1}
/// let det = chol.det();     // det(A)
/// assert!((det - 8.0).abs() < 1e-12);
/// ```
#[derive(Debug)]
pub struct CholeskyDecomposition<T, const N: usize> {
    l: Matrix<T, N, N>,
}

impl<T: LinalgScalar, const N: usize> CholeskyDecomposition<T, N> {
    /// Decompose a (Hermitian) positive-definite matrix.
    ///
    /// Returns an error if the matrix is not positive definite.
    pub fn new(a: &Matrix<T, N, N>) -> Result<Self, LinalgError> {
        let mut l = *a;
        if N <= 6 {
            cholesky_in_place_small(&mut l.data)?;
        } else {
            cholesky_in_place(&mut l)?;
        }
        Ok(Self { l })
    }

    /// Reference to the lower triangular factor L.
    #[inline]
    pub fn l(&self) -> &Matrix<T, N, N> {
        &self.l
    }

    /// Extract the full lower triangular factor (zeros above diagonal).
    pub fn l_full(&self) -> Matrix<T, N, N> {
        let mut out = Matrix::zeros();
        for i in 0..N {
            for j in 0..=i {
                out[(i, j)] = self.l[(i, j)];
            }
        }
        out
    }

    /// Solve A*x = b for x, where A = L*L^H.
    pub fn solve(&self, b: &Vector<T, N>) -> Vector<T, N> {
        let mut b_flat = [T::zero(); N];
        for i in 0..N {
            b_flat[i] = b[i];
        }

        let mut y = [T::zero(); N];
        let mut x = [T::zero(); N];

        if N <= 6 {
            forward_substitute_small(&self.l.data, &b_flat, &mut y);
            back_substitute_lt_small(&self.l.data, &y, &mut x);
        } else {
            forward_substitute(&self.l, &b_flat, &mut y);
            back_substitute_lt(&self.l, &y, &mut x);
        }

        Vector::from_array(x)
    }

    /// Compute the determinant: det(A) = |det(L)|^2.
    ///
    /// For Hermitian PD matrices, the diagonal of L is real and positive,
    /// so det(A) = product(L\[i\]\[i\])^2.
    pub fn det(&self) -> T {
        let mut prod = T::one();
        for i in 0..N {
            prod = prod * self.l[(i, i)];
        }
        prod * prod
    }

    /// Compute the log-determinant: ln(det(A)) = 2 * sum(ln(L\[i\]\[i\])).
    ///
    /// More numerically stable than `det()` for large matrices.
    pub fn ln_det(&self) -> T {
        let mut sum = T::zero();
        for i in 0..N {
            sum = sum + self.l[(i, i)].lln();
        }
        sum + sum
    }

    /// Compute the matrix inverse using the Cholesky factorization.
    pub fn inverse(&self) -> Matrix<T, N, N> {
        let mut inv = Matrix::<T, N, N>::zeros();
        let mut e = [T::zero(); N];
        let mut y = [T::zero(); N];
        let mut x = [T::zero(); N];

        for col in 0..N {
            if col > 0 {
                e[col - 1] = T::zero();
            }
            e[col] = T::one();

            if N <= 6 {
                forward_substitute_small(&self.l.data, &e, &mut y);
                back_substitute_lt_small(&self.l.data, &y, &mut x);
            } else {
                forward_substitute(&self.l, &e, &mut y);
                back_substitute_lt(&self.l, &y, &mut x);
            }

            for row in 0..N {
                inv.data[col][row] = x[row];
            }
        }

        inv
    }
}

/// Convenience methods on square matrices.
impl<T: LinalgScalar, const N: usize> Matrix<T, N, N> {
    /// Cholesky decomposition (`A = L * L^H`).
    ///
    /// For real matrices, this is the standard `A = L * L^T`.
    /// Returns an error if the matrix is not (Hermitian) positive definite.
    ///
    /// ```
    /// use numeris::Matrix;
    /// let spd = Matrix::new([[4.0_f64, 2.0], [2.0, 3.0]]);
    /// let chol = spd.cholesky().unwrap();
    /// let l = chol.l_full();
    /// let reconstructed = l * l.transpose();
    /// assert!((reconstructed[(0, 0)] - 4.0).abs() < 1e-12);
    /// ```
    pub fn cholesky(&self) -> Result<CholeskyDecomposition<T, N>, LinalgError> {
        CholeskyDecomposition::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd_2x2() -> Matrix<f64, 2, 2> {
        // A = [[4, 2], [2, 3]] — symmetric positive definite
        Matrix::new([[4.0, 2.0], [2.0, 3.0]])
    }

    fn spd_3x3() -> Matrix<f64, 3, 3> {
        // A = L * L^T where L = [[2,0,0],[1,3,0],[0.5,0.5,2]]
        Matrix::new([
            [4.0, 2.0, 1.0],
            [2.0, 10.0, 3.5],
            [1.0, 3.5, 4.5],
        ])
    }

    #[test]
    fn cholesky_2x2() {
        let a = spd_2x2();
        let chol = a.cholesky().unwrap();
        let l = chol.l_full();

        // Verify L * L^T == A
        let reconstructed = l * l.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed[(i, j)] - a[(i, j)]).abs() < 1e-12,
                    "mismatch at ({},{})",
                    i, j
                );
            }
        }
    }

    #[test]
    fn cholesky_3x3() {
        let a = spd_3x3();
        let chol = a.cholesky().unwrap();
        let l = chol.l_full();

        let reconstructed = l * l.transpose();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (reconstructed[(i, j)] - a[(i, j)]).abs() < 1e-12,
                    "mismatch at ({},{})",
                    i, j
                );
            }
        }
    }

    #[test]
    fn cholesky_solve() {
        let a = spd_2x2();
        let b = Vector::from_array([8.0, 7.0]);
        let chol = a.cholesky().unwrap();
        let x = chol.solve(&b);

        // Verify A*x == b
        for i in 0..2 {
            let mut sum = 0.0;
            for j in 0..2 {
                sum += a[(i, j)] * x[j];
            }
            assert!((sum - b[i]).abs() < 1e-12, "residual[{}] = {}", i, sum - b[i]);
        }
    }

    #[test]
    fn cholesky_solve_3x3() {
        let a = spd_3x3();
        let b = Vector::from_array([1.0, 2.0, 3.0]);
        let chol = a.cholesky().unwrap();
        let x = chol.solve(&b);

        for i in 0..3 {
            let mut sum = 0.0;
            for j in 0..3 {
                sum += a[(i, j)] * x[j];
            }
            assert!((sum - b[i]).abs() < 1e-10, "residual[{}] = {}", i, sum - b[i]);
        }
    }

    #[test]
    fn cholesky_det() {
        let a = spd_2x2();
        let chol = a.cholesky().unwrap();
        let det_chol = chol.det();
        let det_lu = a.det();
        assert!((det_chol - det_lu).abs() < 1e-12);
    }

    #[test]
    fn cholesky_ln_det() {
        let a = spd_2x2();
        let chol = a.cholesky().unwrap();
        let ln_det = chol.ln_det();
        let expected = chol.det().ln();
        assert!((ln_det - expected).abs() < 1e-12);
    }

    #[test]
    fn cholesky_inverse() {
        let a = spd_3x3();
        let chol = a.cholesky().unwrap();
        let a_inv = chol.inverse();

        let id = a * a_inv;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (id[(i, j)] - expected).abs() < 1e-10,
                    "id[({},{})] = {}, expected {}",
                    i, j, id[(i, j)], expected
                );
            }
        }
    }

    #[test]
    fn cholesky_not_positive_definite() {
        // Not positive definite (negative eigenvalue)
        let a = Matrix::new([[1.0_f64, 5.0], [5.0, 1.0]]);
        assert_eq!(a.cholesky().unwrap_err(), LinalgError::NotPositiveDefinite);
    }

    #[test]
    fn cholesky_in_place_generic() {
        // Verify the free function works via MatrixMut trait
        let mut a = spd_2x2();
        let result = cholesky_in_place(&mut a);
        assert!(result.is_ok());
    }

    #[test]
    fn cholesky_identity() {
        let id: Matrix<f64, 3, 3> = Matrix::eye();
        let chol = id.cholesky().unwrap();
        let l = chol.l_full();
        assert_eq!(l, id);
    }
}
