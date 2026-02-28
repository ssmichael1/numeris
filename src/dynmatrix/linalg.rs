use alloc::vec;
use alloc::vec::Vec;

use crate::linalg::LinalgError;
use crate::linalg::lu::{lu_in_place, lu_solve};
use crate::linalg::cholesky::{cholesky_in_place, forward_substitute, back_substitute_lt};
use crate::linalg::qr::qr_in_place;
use crate::traits::LinalgScalar;

use super::vector::DynVector;
use super::DynMatrix;

// ── DynLu ───────────────────────────────────────────────────────────

/// LU decomposition of a dynamically-sized square matrix.
///
/// Stores the packed L/U factors and permutation vector.
///
/// # Example
///
/// ```
/// use numeris::{DynMatrix, DynVector};
///
/// let a = DynMatrix::from_slice(2, 2, &[2.0_f64, 1.0, 5.0, 3.0]);
/// let lu = a.lu().unwrap();
///
/// let b = DynVector::from_slice(&[4.0, 11.0]);
/// let x = lu.solve(&b);
/// assert!((x[0] - 1.0).abs() < 1e-12);
/// assert!((x[1] - 2.0).abs() < 1e-12);
/// ```
#[derive(Debug)]
pub struct DynLu<T> {
    lu: DynMatrix<T>,
    perm: Vec<usize>,
    even: bool,
}

impl<T: LinalgScalar> DynLu<T> {
    /// Decompose a matrix. Returns an error if the matrix is singular.
    pub fn new(a: &DynMatrix<T>) -> Result<Self, LinalgError> {
        assert!(a.is_square(), "LU decomposition requires a square matrix");
        let n = a.nrows();
        let mut lu = a.clone();
        let mut perm = vec![0usize; n];
        let even = lu_in_place(&mut lu, &mut perm)?;
        Ok(Self { lu, perm, even })
    }

    /// Solve Ax = b for x.
    pub fn solve(&self, b: &DynVector<T>) -> DynVector<T> {
        let n = self.lu.nrows();
        assert_eq!(b.len(), n, "rhs length mismatch");
        let b_flat: Vec<T> = (0..n).map(|i| b[i]).collect();
        let mut x_flat = vec![T::zero(); n];
        lu_solve(&self.lu, &self.perm, &b_flat, &mut x_flat);
        DynVector::from_vec(x_flat)
    }

    /// Compute the matrix inverse.
    pub fn inverse(&self) -> DynMatrix<T> {
        let n = self.lu.nrows();
        let mut inv = DynMatrix::zeros(n, n, T::zero());
        let mut col_buf = vec![T::zero(); n];
        let mut e = vec![T::zero(); n];

        for col in 0..n {
            if col > 0 {
                e[col - 1] = T::zero();
            }
            e[col] = T::one();

            lu_solve(&self.lu, &self.perm, &e, &mut col_buf);

            for row in 0..n {
                inv[(row, col)] = col_buf[row];
            }
        }

        inv
    }

    /// Compute the determinant.
    pub fn det(&self) -> T {
        let n = self.lu.nrows();
        let mut d = if self.even {
            T::one()
        } else {
            T::zero() - T::one()
        };
        for i in 0..n {
            d = d * self.lu[(i, i)];
        }
        d
    }
}

// ── DynCholesky ─────────────────────────────────────────────────────

/// Cholesky decomposition of a dynamically-sized (Hermitian) positive-definite matrix.
///
/// Stores the lower triangular factor L where `A = L * L^H`.
///
/// # Example
///
/// ```
/// use numeris::{DynMatrix, DynVector};
///
/// let a = DynMatrix::from_slice(2, 2, &[4.0_f64, 2.0, 2.0, 3.0]);
/// let chol = a.cholesky().unwrap();
///
/// let b = DynVector::from_slice(&[8.0, 7.0]);
/// let x = chol.solve(&b);
/// let det = chol.det();
/// assert!((det - 8.0).abs() < 1e-12);
/// ```
#[derive(Debug)]
pub struct DynCholesky<T> {
    l: DynMatrix<T>,
}

impl<T: LinalgScalar> DynCholesky<T> {
    /// Decompose a (Hermitian) positive-definite matrix.
    pub fn new(a: &DynMatrix<T>) -> Result<Self, LinalgError> {
        assert!(
            a.is_square(),
            "Cholesky decomposition requires a square matrix"
        );
        let mut l = a.clone();
        cholesky_in_place(&mut l)?;
        Ok(Self { l })
    }

    /// Reference to the lower triangular factor.
    #[inline]
    pub fn l(&self) -> &DynMatrix<T> {
        &self.l
    }

    /// Extract the full lower triangular factor (zeros above diagonal).
    pub fn l_full(&self) -> DynMatrix<T> {
        let n = self.l.nrows();
        let mut out = DynMatrix::zeros(n, n, T::zero());
        for i in 0..n {
            for j in 0..=i {
                out[(i, j)] = self.l[(i, j)];
            }
        }
        out
    }

    /// Solve A*x = b for x, where A = L*L^H.
    pub fn solve(&self, b: &DynVector<T>) -> DynVector<T> {
        let n = self.l.nrows();
        assert_eq!(b.len(), n, "rhs length mismatch");
        let b_flat: Vec<T> = (0..n).map(|i| b[i]).collect();
        let mut y = vec![T::zero(); n];
        forward_substitute(&self.l, &b_flat, &mut y);
        let mut x = vec![T::zero(); n];
        back_substitute_lt(&self.l, &y, &mut x);
        DynVector::from_vec(x)
    }

    /// Determinant: det(A) = product(L[i][i])^2.
    pub fn det(&self) -> T {
        let n = self.l.nrows();
        let mut prod = T::one();
        for i in 0..n {
            prod = prod * self.l[(i, i)];
        }
        prod * prod
    }

    /// Log-determinant: ln(det(A)) = 2 * sum(ln(L[i][i])).
    pub fn ln_det(&self) -> T {
        let n = self.l.nrows();
        let mut sum = T::zero();
        for i in 0..n {
            sum = sum + self.l[(i, i)].lln();
        }
        sum + sum
    }

    /// Matrix inverse using the Cholesky factorization.
    pub fn inverse(&self) -> DynMatrix<T> {
        let n = self.l.nrows();
        let mut inv = DynMatrix::zeros(n, n, T::zero());
        let mut e = vec![T::zero(); n];
        let mut y = vec![T::zero(); n];
        let mut x = vec![T::zero(); n];

        for col in 0..n {
            if col > 0 {
                e[col - 1] = T::zero();
            }
            e[col] = T::one();

            forward_substitute(&self.l, &e, &mut y);
            back_substitute_lt(&self.l, &y, &mut x);

            for row in 0..n {
                inv[(row, col)] = x[row];
            }
        }

        inv
    }
}

// ── DynQr ───────────────────────────────────────────────────────────

/// QR decomposition of a dynamically-sized matrix (M >= N).
///
/// Stores the packed Householder vectors, R, and tau scalars.
///
/// # Example
///
/// ```
/// use numeris::{DynMatrix, DynVector};
///
/// let a = DynMatrix::from_slice(3, 2, &[
///     1.0_f64, 0.0,
///     1.0, 1.0,
///     1.0, 2.0,
/// ]);
/// let b = DynVector::from_slice(&[1.0, 2.0, 4.0]);
/// let x = a.qr().unwrap().solve(&b);
/// assert!((x[0] - 5.0 / 6.0).abs() < 1e-10);
/// assert!((x[1] - 3.0 / 2.0).abs() < 1e-10);
/// ```
#[derive(Debug)]
pub struct DynQr<T> {
    qr: DynMatrix<T>,
    tau: Vec<T>,
}

impl<T: LinalgScalar> DynQr<T> {
    /// Decompose a matrix. Returns an error if a column is rank-deficient.
    pub fn new(a: &DynMatrix<T>) -> Result<Self, LinalgError> {
        let m = a.nrows();
        let n = a.ncols();
        assert!(m >= n, "QR decomposition requires M >= N");
        let k = m.min(n);
        let mut qr = a.clone();
        let mut tau = vec![T::zero(); k];
        qr_in_place(&mut qr, &mut tau)?;
        Ok(Self { qr, tau })
    }

    /// Extract the upper-triangular R factor (N × N).
    pub fn r(&self) -> DynMatrix<T> {
        let n = self.qr.ncols();
        let mut r = DynMatrix::zeros(n, n, T::zero());
        for i in 0..n {
            for j in i..n {
                r[(i, j)] = self.qr[(i, j)];
            }
        }
        r
    }

    /// Compute the thin Q factor (M × N).
    pub fn q(&self) -> DynMatrix<T> {
        let m = self.qr.nrows();
        let n = self.qr.ncols();

        let mut q = DynMatrix::zeros(m, n, T::zero());
        for i in 0..n {
            q[(i, i)] = T::one();
        }

        for col in (0..n).rev() {
            let tau_val = self.tau[col];

            for j in col..n {
                let mut dot = q[(col, j)];
                for i in (col + 1)..m {
                    dot = dot + self.qr[(i, col)].conj() * q[(i, j)];
                }
                dot = dot * tau_val;

                q[(col, j)] = q[(col, j)] - dot;
                for i in (col + 1)..m {
                    q[(i, j)] = q[(i, j)] - dot * self.qr[(i, col)];
                }
            }
        }

        q
    }

    /// Solve the least-squares problem min ||Ax - b|| for x.
    pub fn solve(&self, b: &DynVector<T>) -> DynVector<T> {
        let m = self.qr.nrows();
        let n = self.qr.ncols();
        assert_eq!(b.len(), m, "rhs length mismatch");

        let mut qtb: Vec<T> = (0..m).map(|i| b[i]).collect();

        for col in 0..n {
            let tau_val = self.tau[col];
            let mut dot = qtb[col];
            for i in (col + 1)..m {
                dot = dot + self.qr[(i, col)].conj() * qtb[i];
            }
            dot = dot * tau_val;

            qtb[col] = qtb[col] - dot;
            for i in (col + 1)..m {
                qtb[i] = qtb[i] - dot * self.qr[(i, col)];
            }
        }

        let mut x = vec![T::zero(); n];
        for i in (0..n).rev() {
            let mut sum = qtb[i];
            for j in (i + 1)..n {
                sum = sum - self.qr[(i, j)] * x[j];
            }
            x[i] = sum / self.qr[(i, i)];
        }

        DynVector::from_vec(x)
    }

    /// Determinant of the original matrix (square only).
    pub fn det(&self) -> T {
        let m = self.qr.nrows();
        let n = self.qr.ncols();
        assert_eq!(m, n, "determinant requires a square matrix");
        let mut d = T::one();
        for i in 0..n {
            d = d * self.qr[(i, i)];
        }
        d
    }
}

// ── Convenience methods on DynMatrix ────────────────────────────────

impl<T: LinalgScalar> DynMatrix<T> {
    /// LU decomposition with partial pivoting.
    pub fn lu(&self) -> Result<DynLu<T>, LinalgError> {
        DynLu::new(self)
    }

    /// Cholesky decomposition (`A = L * L^H`).
    pub fn cholesky(&self) -> Result<DynCholesky<T>, LinalgError> {
        DynCholesky::new(self)
    }

    /// QR decomposition using Householder reflections.
    pub fn qr(&self) -> Result<DynQr<T>, LinalgError> {
        DynQr::new(self)
    }

    /// Solve `Ax = b` for `x` via LU decomposition.
    ///
    /// ```
    /// use numeris::{DynMatrix, DynVector};
    /// let a = DynMatrix::from_slice(2, 2, &[2.0_f64, 1.0, 5.0, 3.0]);
    /// let b = DynVector::from_slice(&[4.0, 11.0]);
    /// let x = a.solve(&b).unwrap();
    /// assert!((x[0] - 1.0).abs() < 1e-12);
    /// assert!((x[1] - 2.0).abs() < 1e-12);
    /// ```
    pub fn solve(&self, b: &DynVector<T>) -> Result<DynVector<T>, LinalgError> {
        Ok(self.lu()?.solve(b))
    }

    /// Matrix inverse via LU decomposition.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let a = DynMatrix::from_slice(2, 2, &[4.0_f64, 7.0, 2.0, 6.0]);
    /// let a_inv = a.inverse().unwrap();
    /// let id = &a * &a_inv;
    /// assert!((id[(0, 0)] - 1.0).abs() < 1e-12);
    /// assert!((id[(0, 1)]).abs() < 1e-12);
    /// ```
    pub fn inverse(&self) -> Result<DynMatrix<T>, LinalgError> {
        Ok(self.lu()?.inverse())
    }

    /// Solve `Ax = b` via QR decomposition.
    pub fn solve_qr(&self, b: &DynVector<T>) -> Result<DynVector<T>, LinalgError> {
        Ok(self.qr()?.solve(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_solve_2x2() {
        let a = DynMatrix::from_slice(2, 2, &[3.0_f64, 2.0, 1.0, 4.0]);
        let b = DynVector::from_slice(&[7.0, 9.0]);
        let x = a.solve(&b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn lu_solve_3x3() {
        let a = DynMatrix::from_slice(
            3,
            3,
            &[2.0_f64, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0],
        );
        let b = DynVector::from_slice(&[8.0, -11.0, -3.0]);
        let x = a.solve(&b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-12);
        assert!((x[1] - 3.0).abs() < 1e-12);
        assert!((x[2] - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn lu_inverse_2x2() {
        let a = DynMatrix::from_slice(2, 2, &[4.0_f64, 7.0, 2.0, 6.0]);
        let a_inv = a.inverse().unwrap();
        let id = &a * &a_inv;
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((id[(i, j)] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn lu_det() {
        let a = DynMatrix::from_slice(2, 2, &[3.0_f64, 8.0, 4.0, 6.0]);
        let lu = a.lu().unwrap();
        assert!((lu.det() - (-14.0)).abs() < 1e-12);
    }

    #[test]
    fn lu_singular() {
        let a = DynMatrix::from_slice(2, 2, &[1.0_f64, 2.0, 2.0, 4.0]);
        assert_eq!(a.lu().unwrap_err(), LinalgError::Singular);
    }

    #[test]
    fn cholesky_solve() {
        let a = DynMatrix::from_slice(2, 2, &[4.0_f64, 2.0, 2.0, 3.0]);
        let b = DynVector::from_slice(&[8.0, 7.0]);
        let chol = a.cholesky().unwrap();
        let x = chol.solve(&b);

        for i in 0..2 {
            let mut sum = 0.0;
            for j in 0..2 {
                sum += a[(i, j)] * x[j];
            }
            assert!((sum - b[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn cholesky_det() {
        let a = DynMatrix::from_slice(2, 2, &[4.0_f64, 2.0, 2.0, 3.0]);
        let chol = a.cholesky().unwrap();
        let det_chol = chol.det();
        let det_lu = a.lu().unwrap().det();
        assert!((det_chol - det_lu).abs() < 1e-12);
    }

    #[test]
    fn cholesky_inverse() {
        let a = DynMatrix::from_slice(
            3,
            3,
            &[4.0_f64, 2.0, 1.0, 2.0, 10.0, 3.5, 1.0, 3.5, 4.5],
        );
        let chol = a.cholesky().unwrap();
        let a_inv = chol.inverse();
        let id = &a * &a_inv;
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
    fn cholesky_not_pd() {
        let a = DynMatrix::from_slice(2, 2, &[1.0_f64, 5.0, 5.0, 1.0]);
        assert_eq!(a.cholesky().unwrap_err(), LinalgError::NotPositiveDefinite);
    }

    #[test]
    fn qr_solve_square() {
        let a = DynMatrix::from_slice(
            3,
            3,
            &[2.0_f64, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0],
        );
        let b = DynVector::from_slice(&[8.0, -11.0, -3.0]);
        let x_qr = a.solve_qr(&b).unwrap();
        let x_lu = a.solve(&b).unwrap();
        for i in 0..3 {
            assert!((x_qr[i] - x_lu[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn qr_least_squares() {
        let a = DynMatrix::from_slice(3, 2, &[1.0_f64, 0.0, 1.0, 1.0, 1.0, 2.0]);
        let b = DynVector::from_slice(&[1.0, 2.0, 4.0]);
        let qr = a.qr().unwrap();
        let x = qr.solve(&b);
        assert!((x[0] - 5.0 / 6.0).abs() < 1e-10);
        assert!((x[1] - 3.0 / 2.0).abs() < 1e-10);
    }

    #[test]
    fn qr_q_orthogonal() {
        let a = DynMatrix::from_slice(
            3,
            3,
            &[12.0_f64, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
        );
        let qr = a.qr().unwrap();
        let q = qr.q();
        let r = qr.r();

        // Q*R == A
        let qr_prod = &q * &r;
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (qr_prod[(i, j)] - a[(i, j)]).abs() < 1e-10,
                    "QR[({},{})]",
                    i,
                    j
                );
            }
        }

        // Q^T * Q == I
        let qt = q.transpose();
        let qtq = &qt * &q;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[(i, j)] - expected).abs() < 1e-10,
                    "QtQ[({},{})]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn qr_det() {
        let a = DynMatrix::from_slice(
            3,
            3,
            &[6.0_f64, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0],
        );
        let det_qr = a.qr().unwrap().det();
        let det_lu = a.lu().unwrap().det();
        assert!((det_qr.abs() - det_lu.abs()).abs() < 1e-10);
    }

    #[test]
    fn solve_verify_residual() {
        let a = DynMatrix::from_slice(
            4,
            4,
            &[
                1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 6.0, 4.0, 1.0, 3.0, 1.0, 9.0,
                2.0,
            ],
        );
        let b = DynVector::from_slice(&[10.0, 26.0, 13.0, 15.0]);
        let x = a.solve(&b).unwrap();

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
