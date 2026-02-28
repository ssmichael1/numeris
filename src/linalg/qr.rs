use crate::linalg::LinalgError;
use crate::matrix::vector::Vector;
use crate::traits::{LinalgScalar, MatrixMut};
use crate::Matrix;
use num_traits::Zero;

/// QR decomposition in place using Householder reflections.
///
/// On return, `a` contains the packed QR factorization:
/// - Upper triangle (including diagonal): R
/// - Lower triangle (excluding diagonal): Householder vectors (scaled)
///
/// `tau` is filled with the Householder scalar factors (length min(M,N)).
///
/// Works on rectangular matrices (M >= N).
/// Returns `LinalgError::Singular` if a zero column is encountered.
///
/// For complex matrices, uses `H = I - tau * v * v^H` (conjugate transpose).
pub fn qr_in_place<T: LinalgScalar>(
    a: &mut impl MatrixMut<T>,
    tau: &mut [T],
) -> Result<(), LinalgError> {
    let m = a.nrows();
    let n = a.ncols();
    let k = m.min(n);
    assert!(m >= n, "QR decomposition requires M >= N");
    assert_eq!(tau.len(), k, "tau length must equal min(M, N)");

    for col in 0..k {
        // Compute the squared norm of the sub-column a[col:m, col]
        // using |v|^2 = v * conj(v) for complex support
        let mut norm_sq = <T::Real as Zero>::zero();
        for i in col..m {
            let v = *a.get(i, col);
            norm_sq = norm_sq + (v * v.conj()).re();
        }

        if norm_sq < T::lepsilon() {
            return Err(LinalgError::Singular);
        }

        let norm = norm_sq.lsqrt();
        let a_col_col = *a.get(col, col);

        // sigma: for real, sign(a[col,col]) * ||x||
        //        for complex, a[col,col] / |a[col,col]| * ||x||
        // This ensures v0 = a + sigma = (a/|a|)(|a| + ||x||) avoids cancellation.
        let alpha = a_col_col.modulus();
        let sigma = if alpha < T::lepsilon() {
            T::from_real(norm)
        } else {
            T::from_real(norm) * (a_col_col / T::from_real(alpha))
        };

        // v[col] = a[col,col] + sigma; rest of v is a[col+1:m, col] (stored in-place)
        let v0 = a_col_col + sigma;
        *a.get_mut(col, col) = v0;

        // tau = v0 / sigma
        // For real: tau = v0 / sigma (as before)
        // For complex: tau = 2 / ||v||^2 * v0^2 ... but the formula tau = v0/sigma
        // generalizes correctly when sigma = conj(a)/|a| * norm.
        let tau_val = v0 / sigma;
        tau[col] = tau_val;

        // Scale the sub-diagonal entries by 1/v0 for storage
        for i in (col + 1)..m {
            let val = *a.get(i, col) / v0;
            *a.get_mut(i, col) = val;
        }

        // Apply H to trailing columns: A[col:m, col+1:n] -= tau * v * (v^H * A)
        // where v = [1, a[col+1,col], ..., a[m-1,col]] (stored values)
        // v^H * A uses conj(v_i) for complex.
        for j in (col + 1)..n {
            // Compute v^H * A[:, j] over rows col..m
            // v^H[0] = conj(1) = 1 for the implicit leading 1
            let mut dot = *a.get(col, j); // conj(1) * A[col,j] = A[col,j]
            for i in (col + 1)..m {
                dot = dot + (*a.get(i, col)).conj() * *a.get(i, j);
            }
            dot = dot * tau_val;

            // A[:, j] -= dot * v
            *a.get_mut(col, j) = *a.get(col, j) - dot; // v[0] = 1
            for i in (col + 1)..m {
                let vi = *a.get(i, col);
                let old = *a.get(i, j);
                *a.get_mut(i, j) = old - dot * vi;
            }
        }

        // Store -sigma (the R diagonal entry) in a[col, col]
        *a.get_mut(col, col) = T::zero() - sigma;
    }

    Ok(())
}

/// QR decomposition of a fixed-size matrix (M >= N).
///
/// Stores the packed Householder vectors, R, and tau scalars.
/// Use `q()`, `r()`, `solve()`, or `det()` to work with the decomposition.
///
/// # Example
///
/// ```
/// use numeris::{Matrix, Vector};
///
/// // Least-squares fit: y = c0 + c1*x to points (0,1), (1,2), (2,4)
/// let a = Matrix::new([
///     [1.0_f64, 0.0],
///     [1.0, 1.0],
///     [1.0, 2.0],
/// ]);
/// let b = Vector::from_array([1.0, 2.0, 4.0]);
/// let x = a.qr().unwrap().solve(&b);
/// assert!((x[0] - 5.0 / 6.0).abs() < 1e-10);
/// assert!((x[1] - 3.0 / 2.0).abs() < 1e-10);
/// ```
#[derive(Debug)]
pub struct QrDecomposition<T, const M: usize, const N: usize> {
    qr: Matrix<T, M, N>,
    tau: [T; N],
}

impl<T: LinalgScalar, const M: usize, const N: usize> QrDecomposition<T, M, N> {
    /// Decompose a matrix. Returns an error if a column is rank-deficient.
    pub fn new(a: &Matrix<T, M, N>) -> Result<Self, LinalgError> {
        assert!(M >= N, "QR decomposition requires M >= N");
        let mut qr = *a;
        let mut tau = [T::zero(); N];
        qr_in_place(&mut qr, &mut tau)?;
        Ok(Self { qr, tau })
    }

    /// Extract the upper-triangular R factor (N × N).
    ///
    /// ```
    /// use numeris::Matrix;
    /// let a = Matrix::new([[12.0_f64, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]]);
    /// let r = a.qr().unwrap().r();
    /// // R is upper-triangular
    /// assert!((r[(1, 0)]).abs() < 1e-12);
    /// assert!((r[(2, 0)]).abs() < 1e-12);
    /// ```
    pub fn r(&self) -> Matrix<T, N, N> {
        let mut r = Matrix::<T, N, N>::zeros();
        for i in 0..N {
            for j in i..N {
                r[(i, j)] = self.qr[(i, j)];
            }
        }
        r
    }

    /// Compute the thin Q factor (M × N, orthonormal columns).
    ///
    /// Applies Householder reflections in reverse to the first N columns
    /// of the identity matrix.
    ///
    /// For complex matrices, Q is unitary (`Q^H * Q = I`).
    ///
    /// ```
    /// use numeris::Matrix;
    /// let a = Matrix::new([[12.0_f64, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]]);
    /// let qr = a.qr().unwrap();
    /// let q = qr.q();
    /// let qtq = q.transpose() * q;
    /// // Q^T * Q ≈ I
    /// assert!((qtq[(0, 0)] - 1.0).abs() < 1e-10);
    /// assert!((qtq[(0, 1)]).abs() < 1e-10);
    /// ```
    pub fn q(&self) -> Matrix<T, M, N> {
        // Start with the M×N "thin identity": e_0..e_{N-1}
        let mut q = Matrix::<T, M, N>::zeros();
        for i in 0..N {
            q[(i, i)] = T::one();
        }

        // Apply reflections in reverse order
        for col in (0..N).rev() {
            let tau_val = self.tau[col];

            // Apply H_col to Q[col:M, col:N]
            // v = [1, qr[col+1,col], ..., qr[M-1,col]]
            // H = I - tau * v * v^H
            for j in col..N {
                let mut dot = q[(col, j)]; // conj(1) * q[col,j]
                for i in (col + 1)..M {
                    dot = dot + self.qr[(i, col)].conj() * q[(i, j)];
                }
                dot = dot * tau_val;

                q[(col, j)] = q[(col, j)] - dot;
                for i in (col + 1)..M {
                    q[(i, j)] = q[(i, j)] - dot * self.qr[(i, col)];
                }
            }
        }

        q
    }

    /// Solve the least-squares problem min ||Ax - b|| for x.
    ///
    /// Computes x = R^{-1} Q^H b via Householder application + back substitution.
    pub fn solve(&self, b: &Vector<T, M>) -> Vector<T, N> {
        // Apply Q^H to b by applying each Householder reflection in order.
        // Work with a copy of b as a flat array of length M.
        let mut qtb = [T::zero(); M];
        for i in 0..M {
            qtb[i] = b[i];
        }

        for col in 0..N {
            let tau_val = self.tau[col];
            // v = [1, qr[col+1,col], ..., qr[M-1,col]]
            let mut dot = qtb[col]; // conj(1) * qtb[col]
            for i in (col + 1)..M {
                dot = dot + self.qr[(i, col)].conj() * qtb[i];
            }
            dot = dot * tau_val;

            qtb[col] = qtb[col] - dot;
            for i in (col + 1)..M {
                qtb[i] = qtb[i] - dot * self.qr[(i, col)];
            }
        }

        // Back substitution with R (upper triangle of qr, first N rows)
        let mut x = [T::zero(); N];
        for i in (0..N).rev() {
            let mut sum = qtb[i];
            for j in (i + 1)..N {
                sum = sum - self.qr[(i, j)] * x[j];
            }
            x[i] = sum / self.qr[(i, i)];
        }

        Vector::from_array(x)
    }

    /// Determinant of the original matrix (square only, M == N).
    ///
    /// Panics at runtime if M != N.
    pub fn det(&self) -> T {
        assert_eq!(M, N, "determinant requires a square matrix");
        let mut d = T::one();
        for i in 0..N {
            d = d * self.qr[(i, i)];
        }
        d
    }
}

/// Convenience methods on rectangular matrices.
impl<T: LinalgScalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// QR decomposition using Householder reflections.
    ///
    /// Returns an error if the matrix is rank-deficient.
    pub fn qr(&self) -> Result<QrDecomposition<T, M, N>, LinalgError> {
        QrDecomposition::new(self)
    }
}

/// Convenience methods for QR solve on square matrices.
impl<T: LinalgScalar, const N: usize> Matrix<T, N, N> {
    /// Solve Ax = b via QR decomposition.
    pub fn solve_qr(&self, b: &Vector<T, N>) -> Result<Vector<T, N>, LinalgError> {
        Ok(self.qr()?.solve(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn assert_near(a: f64, b: f64, tol: f64, msg: &str) {
        assert!((a - b).abs() < tol, "{}: {} vs {} (diff {})", msg, a, b, (a - b).abs());
    }

    #[test]
    fn qr_square_3x3() {
        let a = Matrix::new([
            [12.0_f64, -51.0, 4.0],
            [6.0, 167.0, -68.0],
            [-4.0, 24.0, -41.0],
        ]);
        let qr = a.qr().unwrap();
        let q = qr.q();
        let r = qr.r();

        // Verify Q*R == A
        let qr_prod = q * r;
        for i in 0..3 {
            for j in 0..3 {
                assert_near(qr_prod[(i, j)], a[(i, j)], TOL,
                    &format!("QR[({},{})]", i, j));
            }
        }

        // Verify Q^T * Q == I
        let qtq = q.transpose() * q;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(qtq[(i, j)], expected, TOL,
                    &format!("QtQ[({},{})]", i, j));
            }
        }
    }

    #[test]
    fn qr_rectangular_4x3() {
        let a = Matrix::new([
            [1.0_f64, -1.0, 4.0],
            [1.0, 4.0, -2.0],
            [1.0, 4.0, 2.0],
            [1.0, -1.0, 0.0],
        ]);
        let qr = a.qr().unwrap();
        let q = qr.q();
        let r = qr.r();

        // Verify Q*R == A (Q is 4×3, R is 3×3)
        let qr_prod: Matrix<f64, 4, 3> = q * r;
        for i in 0..4 {
            for j in 0..3 {
                assert_near(qr_prod[(i, j)], a[(i, j)], TOL,
                    &format!("QR[({},{})]", i, j));
            }
        }

        // Verify Q^T * Q == I_3 (thin Q)
        let qtq: Matrix<f64, 3, 3> = q.transpose() * q;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(qtq[(i, j)], expected, TOL,
                    &format!("QtQ[({},{})]", i, j));
            }
        }
    }

    #[test]
    fn qr_solve_square() {
        // Compare QR solve against LU solve
        let a = Matrix::new([
            [2.0_f64, 1.0, -1.0],
            [-3.0, -1.0, 2.0],
            [-2.0, 1.0, 2.0],
        ]);
        let b = Vector::from_array([8.0, -11.0, -3.0]);

        let x_qr = a.solve_qr(&b).unwrap();
        let x_lu = a.solve(&b).unwrap();

        for i in 0..3 {
            assert_near(x_qr[i], x_lu[i], TOL, &format!("x[{}]", i));
        }
    }

    #[test]
    fn qr_least_squares() {
        // Overdetermined system: 3 equations, 2 unknowns
        // Fit y = c0 + c1*x to points (0,1), (1,2), (2,4)
        // A = [[1, 0], [1, 1], [1, 2]], b = [1, 2, 4]
        let a = Matrix::new([
            [1.0_f64, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ]);
        let b = Vector::from_array([1.0, 2.0, 4.0]);

        let qr = a.qr().unwrap();
        let x = qr.solve(&b);

        // Least-squares solution: A^T A x = A^T b
        // A^T A = [[3, 3], [3, 5]], A^T b = [7, 10]
        // x = [5/6, 3/2]
        assert_near(x[0], 5.0 / 6.0, TOL, "c0");
        assert_near(x[1], 3.0 / 2.0, TOL, "c1");

        // Verify the residual r = b - Ax is orthogonal to column space of A
        // i.e. A^T * r ≈ 0
        let ax = a.vecmul(&x);
        let r = b - ax;
        let at = a.transpose();
        let atr = at.vecmul(&r);
        for i in 0..2 {
            assert_near(atr[i], 0.0, TOL, &format!("A^T r[{}]", i));
        }
    }

    #[test]
    fn qr_det() {
        let a = Matrix::new([
            [6.0_f64, 1.0, 1.0],
            [4.0, -2.0, 5.0],
            [2.0, 8.0, 7.0],
        ]);
        let qr = a.qr().unwrap();
        let det_qr = qr.det();
        let det_lu = a.det();
        assert_near(det_qr.abs(), det_lu.abs(), TOL, "det magnitude");
    }

    #[test]
    fn qr_identity() {
        let id: Matrix<f64, 3, 3> = Matrix::eye();
        let qr = id.qr().unwrap();
        let q = qr.q();
        let r = qr.r();

        // Q should be identity (up to sign flips on columns)
        // R should be identity (up to sign flips on diagonal)
        // Q*R should be identity
        let prod = q * r;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(prod[(i, j)], expected, TOL,
                    &format!("QR[({},{})]", i, j));
            }
        }
    }

    #[test]
    fn qr_in_place_generic() {
        // Verify the free function works via the MatrixMut trait
        let mut a = Matrix::new([[2.0_f64, 1.0], [4.0, 3.0]]);
        let mut tau = [0.0; 2];
        let result = qr_in_place(&mut a, &mut tau);
        assert!(result.is_ok());
    }

    #[test]
    fn qr_rank_deficient() {
        // Matrix with a zero column
        let a = Matrix::new([
            [1.0_f64, 0.0],
            [0.0, 0.0],
        ]);
        assert_eq!(a.qr().unwrap_err(), LinalgError::Singular);
    }
}
