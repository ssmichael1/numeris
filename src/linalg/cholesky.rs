use crate::linalg::LinalgError;
use crate::matrix::vector::Vector;
use crate::traits::{FloatScalar, LinalgScalar, MatrixMut, MatrixRef};
use crate::Matrix;

// ---------------------------------------------------------------------------
// Public functions (used by DynCholesky — semantics unchanged)
// ---------------------------------------------------------------------------

/// Cholesky decomposition in place: A = L * L^H.
///
/// For real matrices, L^H = L^T (standard Cholesky).
/// For complex matrices, this is the Hermitian Cholesky decomposition.
///
/// On return, the lower triangle of `a` (including diagonal) contains L.
/// The upper triangle is left unchanged.
///
/// Returns an error if the matrix is not (Hermitian) positive definite.
#[inline]
pub fn cholesky_in_place<T: LinalgScalar>(a: &mut impl MatrixMut<T>) -> Result<(), LinalgError> {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "Cholesky decomposition requires a square matrix");

    for j in 0..n {
        for k in 0..j {
            let ljk_conj = (*a.get(j, k)).conj();
            let (col_j, col_k) = super::split_two_col_slices(a, j, k, j);
            crate::simd::axpy_neg_dispatch(col_j, ljk_conj, col_k);
        }

        let diag = *a.get(j, j);
        if diag.re() <= <T::Real as num_traits::Zero>::zero() {
            return Err(LinalgError::NotPositiveDefinite);
        }
        let ljj = diag.re().lsqrt();
        let ljj_t = T::from_real(ljj);
        *a.get_mut(j, j) = ljj_t;

        let inv_ljj = T::one() / ljj_t;
        let col = a.col_as_mut_slice(j, j + 1);
        for x in col.iter_mut() {
            *x = *x * inv_ljj;
        }
    }

    Ok(())
}

/// Solve L*x = b by forward substitution, where L is lower triangular.
#[inline]
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
#[inline]
pub fn back_substitute_lt<T: LinalgScalar>(
    l: &impl MatrixRef<T>,
    b: &[T],
    x: &mut [T],
) {
    let n = l.nrows();
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum = sum - (*l.get(j, i)).conj() * x[j];
        }
        x[i] = sum / (*l.get(i, i)).conj();
    }
}

// ---------------------------------------------------------------------------
// Rank-1 update / downdate
// ---------------------------------------------------------------------------

/// Rank-1 update: given lower-triangular `L` where `A = L·Lᵀ`, compute
/// `L'` in-place such that `A + v·vᵀ = L'·L'ᵀ`.
///
/// `v` is used as workspace and modified in place.
///
/// Works with both fixed-size [`Matrix`] and [`DynMatrix`](crate::DynMatrix)
/// via the [`MatrixMut`] trait.
///
/// # Errors
///
/// Returns [`LinalgError::NotPositiveDefinite`] if a diagonal element of `L`
/// is zero (matrix is singular).
///
/// # Example
///
/// ```
/// use numeris::Matrix;
/// use numeris::linalg::cholesky_rank1_update;
///
/// let a = Matrix::new([[4.0_f64, 2.0], [2.0, 3.0]]);
/// let mut l = a.cholesky().unwrap().l_full();
/// let v_orig = [1.0_f64, 0.5];
/// let mut v = v_orig;
///
/// cholesky_rank1_update(&mut l, &mut v).unwrap();
///
/// // L'·L'ᵀ should equal A + v·vᵀ
/// let result = l * l.transpose();
/// assert!((result[(0, 0)] - 5.0).abs() < 1e-12);  // 4 + 1*1
/// assert!((result[(1, 1)] - 3.25).abs() < 1e-12);  // 3 + 0.5*0.5
/// ```
pub fn cholesky_rank1_update<T: FloatScalar>(
    l: &mut impl MatrixMut<T>,
    v: &mut [T],
) -> Result<(), LinalgError> {
    cholesky_rank1_impl(l, v, T::one())
}

/// Rank-1 downdate: given lower-triangular `L` where `A = L·Lᵀ`, compute
/// `L'` in-place such that `A - v·vᵀ = L'·L'ᵀ`.
///
/// `v` is used as workspace and modified in place.
///
/// Works with both fixed-size [`Matrix`] and [`DynMatrix`](crate::DynMatrix)
/// via the [`MatrixMut`] trait.
///
/// # Errors
///
/// Returns [`LinalgError::NotPositiveDefinite`] if the result would not be
/// positive definite.
///
/// # Example
///
/// ```
/// use numeris::Matrix;
/// use numeris::linalg::cholesky_rank1_downdate;
///
/// // Start with A + v·vᵀ, downdate by v to recover A
/// let a = Matrix::new([[4.0_f64, 2.0], [2.0, 3.0]]);
/// let v_col = Matrix::new([[0.5], [0.3_f64]]);
/// let a_aug = a + v_col * v_col.transpose();
///
/// let mut l = a_aug.cholesky().unwrap().l_full();
/// let mut v = [0.5, 0.3_f64];
///
/// cholesky_rank1_downdate(&mut l, &mut v).unwrap();
///
/// let recovered = l * l.transpose();
/// for i in 0..2 {
///     for j in 0..2 {
///         assert!((recovered[(i, j)] - a[(i, j)]).abs() < 1e-10);
///     }
/// }
/// ```
pub fn cholesky_rank1_downdate<T: FloatScalar>(
    l: &mut impl MatrixMut<T>,
    v: &mut [T],
) -> Result<(), LinalgError> {
    cholesky_rank1_impl(l, v, T::one().neg())
}

/// Internal implementation for rank-1 update (`sign = +1`) or downdate (`sign = -1`).
///
/// Algorithm (direct formulation, cf. LINPACK `dchud`/`dchdd`):
///
/// For j = 0..N:
///   1. `r = hypot(L[j,j], v[j])` (update) or `sqrt(L[j,j]² - v[j]²)` (downdate)
///   2. `c = r / L[j,j]`, `s = v[j] / L[j,j]`
///   3. `L[j,j] = r`
///   4. For i = j+1..N:
///        `L[i,j] = (L[i,j] + sign·s·v[i]) / c`
///        `v[i]   = c·v[i] - s·L[i,j]_new`
fn cholesky_rank1_impl<T: FloatScalar>(
    l: &mut impl MatrixMut<T>,
    v: &mut [T],
    sign: T,
) -> Result<(), LinalgError> {
    let n = l.nrows();
    debug_assert_eq!(n, l.ncols());
    debug_assert_eq!(v.len(), n);

    for j in 0..n {
        let ljj = *l.get(j, j);
        let vj = v[j];

        let r = if sign > T::zero() {
            // Update: use hypot for numerical stability
            ljj.hypot(vj)
        } else {
            // Downdate: need sqrt(ljj² - vj²)
            let arg = ljj * ljj + sign * vj * vj;
            if arg <= T::zero() {
                return Err(LinalgError::NotPositiveDefinite);
            }
            arg.sqrt()
        };

        let c = r / ljj;
        let s = vj / ljj;
        *l.get_mut(j, j) = r;

        for i in (j + 1)..n {
            let lij = *l.get(i, j);
            let new_lij = (lij + sign * s * v[i]) / c;
            *l.get_mut(i, j) = new_lij;
            v[i] = c * v[i] - s * new_lij;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Small-size specialization (N <= 6, direct data access, compiler unrolls)
// ---------------------------------------------------------------------------

/// Cholesky decomposition using direct `data[col][row]` access.
/// For small const N the compiler fully unrolls all loops.
#[inline(always)]
fn cholesky_direct<T: LinalgScalar, const N: usize>(
    l: &mut Matrix<T, N, N>,
) -> Result<(), LinalgError> {
    let zero_r = <T::Real as num_traits::Zero>::zero();

    for j in 0..N {
        for k in 0..j {
            let ljk_conj = l.data[k][j].conj();
            for i in j..N {
                l.data[j][i] = l.data[j][i] - ljk_conj * l.data[k][i];
            }
        }

        let diag = l.data[j][j];
        if diag.re() <= zero_r {
            return Err(LinalgError::NotPositiveDefinite);
        }
        let ljj = diag.re().lsqrt();
        let ljj_t = T::from_real(ljj);
        l.data[j][j] = ljj_t;

        let inv_ljj = T::one() / ljj_t;
        for i in (j + 1)..N {
            l.data[j][i] = l.data[j][i] * inv_ljj;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Public CholeskyDecomposition
// ---------------------------------------------------------------------------

/// Cholesky decomposition of a fixed-size (Hermitian) positive-definite matrix.
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
    /// Lower triangular Cholesky factor L (A = L·L^H).
    l: Matrix<T, N, N>,
}

impl<T: LinalgScalar, const N: usize> CholeskyDecomposition<T, N> {
    /// Decompose a (Hermitian) positive-definite matrix.
    ///
    /// Returns an error if the matrix is not positive definite.
    #[inline]
    pub fn new(a: &Matrix<T, N, N>) -> Result<Self, LinalgError> {
        let mut l = *a;
        if N <= 6 {
            cholesky_direct(&mut l)?;
        } else {
            cholesky_in_place(&mut l)?;
        }
        Ok(Self { l })
    }

    /// Return a reference to the lower triangular Cholesky factor L.
    pub fn l(&self) -> &Matrix<T, N, N> {
        &self.l
    }

    /// Extract the full lower triangular factor (zeros above diagonal).
    pub fn l_full(&self) -> Matrix<T, N, N> {
        let mut out = self.l;
        for j in 0..N {
            for i in 0..j {
                out.data[j][i] = T::zero();
            }
        }
        out
    }

    /// Solve A*x = b for x, where A = L·L^H.
    pub fn solve(&self, b: &Vector<T, N>) -> Vector<T, N> {
        let b_flat: [T; N] = core::array::from_fn(|i| b[i]);
        let mut y = [T::zero(); N];
        let mut x = [T::zero(); N];

        forward_substitute(&self.l, &b_flat, &mut y);
        back_substitute_lt(&self.l, &y, &mut x);

        Vector::from_array(x)
    }

    /// Compute the determinant: det(A) = (Π L\[i,i\])².
    pub fn det(&self) -> T {
        let mut prod = T::one();
        for i in 0..N {
            prod = prod * self.l[(i, i)];
        }
        prod * prod
    }

    /// Compute the log-determinant: ln(det(A)) = 2 · Σ ln(L\[i,i\]).
    ///
    /// More numerically stable than `det()` for large matrices.
    pub fn ln_det(&self) -> T {
        let two = T::one() + T::one();
        let mut sum = T::zero();
        for i in 0..N {
            sum = sum + self.l[(i, i)].lln();
        }
        sum * two
    }

    /// Apply a rank-1 update in place: `A + v·vᵀ`.
    ///
    /// After the update, `self` holds the Cholesky factor of the updated matrix.
    /// The vector `v` is used as workspace and modified.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::{Matrix, Vector};
    ///
    /// let a = Matrix::new([[4.0_f64, 2.0], [2.0, 3.0]]);
    /// let mut chol = a.cholesky().unwrap();
    /// let mut v = Vector::from_array([1.0, 0.5_f64]);
    ///
    /// chol.rank1_update(&mut v).unwrap();
    ///
    /// let l = chol.l_full();
    /// let result = l * l.transpose();
    /// assert!((result[(0, 0)] - 5.0).abs() < 1e-12);
    /// ```
    pub fn rank1_update(&mut self, v: &mut Vector<T, N>) -> Result<(), LinalgError>
    where
        T: FloatScalar,
    {
        cholesky_rank1_update(&mut self.l, v.as_mut_slice())
    }

    /// Apply a rank-1 downdate in place: `A - v·vᵀ`.
    ///
    /// After the downdate, `self` holds the Cholesky factor of the downdated
    /// matrix. Returns an error if the result would not be positive definite.
    /// The vector `v` is used as workspace and modified.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::{Matrix, Vector};
    ///
    /// let a = Matrix::new([[4.0_f64, 2.0], [2.0, 3.0]]);
    /// let v_col = Matrix::new([[0.5], [0.3_f64]]);
    /// let a_aug = a + v_col * v_col.transpose();
    /// let mut chol = a_aug.cholesky().unwrap();
    /// let mut v = Vector::from_array([0.5, 0.3_f64]);
    ///
    /// chol.rank1_downdate(&mut v).unwrap();
    ///
    /// let l = chol.l_full();
    /// let recovered = l * l.transpose();
    /// for i in 0..2 {
    ///     for j in 0..2 {
    ///         assert!((recovered[(i, j)] - a[(i, j)]).abs() < 1e-10);
    ///     }
    /// }
    /// ```
    pub fn rank1_downdate(&mut self, v: &mut Vector<T, N>) -> Result<(), LinalgError>
    where
        T: FloatScalar,
    {
        cholesky_rank1_downdate(&mut self.l, v.as_mut_slice())
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

            forward_substitute(&self.l, &e, &mut y);
            back_substitute_lt(&self.l, &y, &mut x);

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
    #[inline]
    pub fn cholesky(&self) -> Result<CholeskyDecomposition<T, N>, LinalgError> {
        CholeskyDecomposition::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd_2x2() -> Matrix<f64, 2, 2> {
        Matrix::new([[4.0, 2.0], [2.0, 3.0]])
    }

    fn spd_3x3() -> Matrix<f64, 3, 3> {
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

        let reconstructed = l * l.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed[(i, j)] - a[(i, j)]).abs() < 1e-12,
                    "mismatch at ({},{})",
                    i,
                    j
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
                    i,
                    j
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
                    i,
                    j,
                    id[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn cholesky_not_positive_definite() {
        let a = Matrix::new([[1.0_f64, 5.0], [5.0, 1.0]]);
        assert_eq!(a.cholesky().unwrap_err(), LinalgError::NotPositiveDefinite);
    }

    #[test]
    fn cholesky_in_place_generic() {
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

    // ── rank-1 update/downdate tests ─────────────────────────────────

    #[test]
    fn rank1_update_2x2() {
        let a = spd_2x2();
        let mut l = a.cholesky().unwrap().l_full();
        let v_orig = [1.0_f64, 0.5];
        let mut v = v_orig;

        cholesky_rank1_update(&mut l, &mut v).unwrap();

        let p_new = l * l.transpose();
        // Expected: A + v*v^T
        assert!((p_new[(0, 0)] - 5.0).abs() < 1e-12);   // 4 + 1
        assert!((p_new[(0, 1)] - 2.5).abs() < 1e-12);   // 2 + 0.5
        assert!((p_new[(1, 0)] - 2.5).abs() < 1e-12);
        assert!((p_new[(1, 1)] - 3.25).abs() < 1e-12);  // 3 + 0.25
    }

    #[test]
    fn rank1_downdate_roundtrip() {
        let a = spd_2x2();
        let v_orig = [0.5_f64, 0.3];
        // Build A + v*v^T
        let v_col = Matrix::new([[0.5], [0.3_f64]]);
        let a_aug = a + v_col * v_col.transpose();

        let mut l = a_aug.cholesky().unwrap().l_full();
        let mut v = v_orig;

        cholesky_rank1_downdate(&mut l, &mut v).unwrap();

        let recovered = l * l.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (recovered[(i, j)] - a[(i, j)]).abs() < 1e-10,
                    "mismatch at ({},{}): {} vs {}",
                    i, j, recovered[(i, j)], a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn rank1_downdate_fails_non_pd() {
        let mut l = Matrix::<f64, 2, 2>::eye();
        let mut v = [1.5_f64, 0.0];

        let result = cholesky_rank1_downdate(&mut l, &mut v);
        assert_eq!(result.unwrap_err(), LinalgError::NotPositiveDefinite);
    }

    #[test]
    fn rank1_update_3x3() {
        let a = spd_3x3();
        let mut l = a.cholesky().unwrap().l_full();
        let v_orig = [0.3_f64, 0.7, 0.1];
        let mut v = v_orig;

        cholesky_rank1_update(&mut l, &mut v).unwrap();

        let p_new = l * l.transpose();
        let v_col = Matrix::new([[0.3], [0.7], [0.1_f64]]);
        let p_expected = a + v_col * v_col.transpose();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (p_new[(i, j)] - p_expected[(i, j)]).abs() < 1e-10,
                    "mismatch at ({},{})",
                    i, j
                );
            }
        }
    }

    #[test]
    fn rank1_update_via_decomposition() {
        let a = spd_2x2();
        let mut chol = a.cholesky().unwrap();
        let mut v = Vector::from_array([1.0_f64, 0.5]);

        chol.rank1_update(&mut v).unwrap();

        let l = chol.l_full();
        let result = l * l.transpose();
        assert!((result[(0, 0)] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn rank1_downdate_via_decomposition() {
        let a = spd_2x2();
        let v_col = Matrix::new([[0.5], [0.3_f64]]);
        let a_aug = a + v_col * v_col.transpose();
        let mut chol = a_aug.cholesky().unwrap();
        let mut v = Vector::from_array([0.5_f64, 0.3]);

        chol.rank1_downdate(&mut v).unwrap();

        let l = chol.l_full();
        let recovered = l * l.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert!((recovered[(i, j)] - a[(i, j)]).abs() < 1e-10);
            }
        }
    }
}
