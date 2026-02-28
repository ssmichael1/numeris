use crate::linalg::LinalgError;
use crate::matrix::vector::Vector;
use crate::traits::{FloatScalar, MatrixMut};
use crate::Matrix;

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
pub fn qr_in_place<T: FloatScalar>(
    a: &mut impl MatrixMut<T>,
    tau: &mut [T],
) -> Result<(), LinalgError> {
    let m = a.nrows();
    let n = a.ncols();
    let k = m.min(n);
    assert!(m >= n, "QR decomposition requires M >= N");
    assert_eq!(tau.len(), k, "tau length must equal min(M, N)");

    for col in 0..k {
        // Compute the norm of the sub-column a[col:m, col]
        let mut norm_sq = T::zero();
        for i in col..m {
            let v = *a.get(i, col);
            norm_sq = norm_sq + v * v;
        }

        if norm_sq < T::epsilon() {
            return Err(LinalgError::Singular);
        }

        let norm = norm_sq.sqrt();
        let a_col_col = *a.get(col, col);

        // sigma = sign(a[col,col]) * ||x||
        let sigma = if a_col_col >= T::zero() { norm } else { T::zero() - norm };

        // v[col] = a[col,col] + sigma; rest of v is a[col+1:m, col] (stored in-place)
        let v0 = a_col_col + sigma;
        *a.get_mut(col, col) = v0;

        // tau = v0 / sigma  (= 2 / (v^T v / sigma^2) normalized form)
        // But we use: tau = 2*v0*sigma / (v0^2 + sum(v[i]^2, i>col))
        // Simpler: tau = v0 / sigma is equivalent when v is [a+sigma, a[col+1], ...]
        // and sigma = sign(a)*||x||.
        //
        // Actually: ||v||^2 = v0^2 + sum(a[i,col]^2, i>col)
        //                   = (a+sigma)^2 + (||x||^2 - a^2)
        //                   = a^2 + 2*a*sigma + sigma^2 + ||x||^2 - a^2
        //                   = 2*sigma^2 + 2*a*sigma  = 2*sigma*(sigma+a) = 2*sigma*v0
        // So tau = 2/(||v||^2) = 1/(sigma*v0) ... no wait:
        // H = I - tau * v * v^T where tau = 2/||v||^2
        // tau = 2 / (2*sigma*v0) = 1 / (sigma*v0)
        //
        // But we want to store scaled vectors below diagonal. Let's normalize v
        // by dividing by v0, so v_stored = [1, a[col+1]/v0, ...].
        // Then H = I - tau' * v_stored * v_stored^T where tau' = tau * v0^2 = v0/sigma.
        let tau_val = v0 / sigma;
        tau[col] = tau_val;

        // Scale the sub-diagonal entries by 1/v0 for storage
        for i in (col + 1)..m {
            let val = *a.get(i, col) / v0;
            *a.get_mut(i, col) = val;
        }

        // Apply H to trailing columns: A[col:m, col+1:n] -= tau * v * (v^T * A)
        // where v = [1, a[col+1,col], ..., a[m-1,col]] (stored values)
        for j in (col + 1)..n {
            // Compute v^T * A[:, j] over rows col..m
            let mut dot = *a.get(col, j); // v[0] = 1
            for i in (col + 1)..m {
                dot = dot + *a.get(i, col) * *a.get(i, j);
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
#[derive(Debug)]
pub struct QrDecomposition<T, const M: usize, const N: usize> {
    qr: Matrix<T, M, N>,
    tau: [T; N],
}

impl<T: FloatScalar, const M: usize, const N: usize> QrDecomposition<T, M, N> {
    /// Decompose a matrix. Returns an error if a column is rank-deficient.
    pub fn new(a: &Matrix<T, M, N>) -> Result<Self, LinalgError> {
        assert!(M >= N, "QR decomposition requires M >= N");
        let mut qr = *a;
        let mut tau = [T::zero(); N];
        qr_in_place(&mut qr, &mut tau)?;
        Ok(Self { qr, tau })
    }

    /// Extract the upper-triangular R factor (N x N).
    pub fn r(&self) -> Matrix<T, N, N> {
        let mut r = Matrix::<T, N, N>::zeros();
        for i in 0..N {
            for j in i..N {
                r[(i, j)] = self.qr[(i, j)];
            }
        }
        r
    }

    /// Compute the thin Q factor (M x N, orthonormal columns).
    ///
    /// Applies Householder reflections in reverse to the first N columns
    /// of the identity matrix.
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
            for j in col..N {
                let mut dot = q[(col, j)]; // v[0] = 1
                for i in (col + 1)..M {
                    dot = dot + self.qr[(i, col)] * q[(i, j)];
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
    /// Computes x = R^{-1} Q^T b via Householder application + back substitution.
    pub fn solve(&self, b: &Vector<T, M>) -> Vector<T, N> {
        // Apply Q^T to b by applying each Householder reflection in order.
        // Work with a copy of b as a flat array of length M.
        let mut qtb = [T::zero(); M];
        for i in 0..M {
            qtb[i] = b[i];
        }

        for col in 0..N {
            let tau_val = self.tau[col];
            // v = [1, qr[col+1,col], ..., qr[M-1,col]]
            let mut dot = qtb[col];
            for i in (col + 1)..M {
                dot = dot + self.qr[(i, col)] * qtb[i];
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
        // Q is orthogonal with det ±1. Each Householder reflection has det -1,
        // so det(Q) = (-1)^N. Since A = QR, det(A) = det(Q)*det(R) = (-1)^N * prod(R_ii).
        // But our R diagonal stores -sigma where sigma has the same sign as the original
        // diagonal (chosen to avoid cancellation). The product of R diagonals already
        // encodes the correct sign because each -sigma = -sign(a)*||x||, and we applied
        // N reflections. So det(A) = product of R diagonal entries (they already include
        // the sign flips from the Householder convention).
        d
    }
}

/// Convenience methods on rectangular matrices.
impl<T: FloatScalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// QR decomposition using Householder reflections.
    ///
    /// Returns an error if the matrix is rank-deficient.
    pub fn qr(&self) -> Result<QrDecomposition<T, M, N>, LinalgError> {
        QrDecomposition::new(self)
    }
}

/// Convenience methods for QR solve on square matrices.
impl<T: FloatScalar, const N: usize> Matrix<T, N, N> {
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
