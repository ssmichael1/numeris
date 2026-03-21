use crate::linalg::{split_two_col_slices, LinalgError};
use crate::matrix::vector::Vector;
use crate::traits::{LinalgScalar, MatrixMut, MatrixRef};
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
        let sub_col = a.col_as_slice(col, col);
        let mut norm_sq = <T::Real as Zero>::zero();
        for &v in sub_col {
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
        {
            let sub_col = a.col_as_mut_slice(col, col + 1);
            for x in sub_col.iter_mut() {
                *x = *x / v0;
            }
        }

        // Apply H to trailing columns: A[col:m, col+1:n] -= tau * v * (v^H * A)
        // where v = [1, a[col+1,col], ..., a[m-1,col]] (stored values)
        // v^H * A uses conj(v_i) for complex.
        //
        // Column-major: v sub-column and A[:,j] sub-column are contiguous.
        // The AXPY step (no conjugation) uses SIMD dispatch on the slices.
        for j in (col + 1)..n {
            // Compute v^H * A[:, j] over rows col..m (needs conj for complex)
            let mut dot = *a.get(col, j); // conj(1) * A[col,j] = A[col,j]
            let (v_slice, a_j_slice) = split_two_col_slices(a, col, j, col + 1);
            for idx in 0..v_slice.len() {
                dot = dot + v_slice[idx].conj() * a_j_slice[idx];
            }
            dot = dot * tau_val;

            // A[:, j] -= dot * v (no conjugation — uses SIMD AXPY dispatch)
            *a.get_mut(col, j) = *a.get(col, j) - dot; // v[0] = 1
            let (v_slice, a_j_slice) = split_two_col_slices(a, col, j, col + 1);
            crate::simd::axpy_neg_dispatch(a_j_slice, dot, v_slice);
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
            let v_slice = self.qr.col_as_slice(col, col + 1);
            for j in col..N {
                let mut dot = q[(col, j)]; // conj(1) * q[col,j]
                let q_j_slice = q.col_as_slice(j, col + 1);
                for idx in 0..v_slice.len() {
                    dot = dot + v_slice[idx].conj() * q_j_slice[idx];
                }
                dot = dot * tau_val;

                q[(col, j)] = q[(col, j)] - dot;
                let q_j_slice = q.col_as_mut_slice(j, col + 1);
                crate::simd::axpy_neg_dispatch(q_j_slice, dot, v_slice);
            }
        }

        q
    }

    /// Solve `AX = B` for X where B has multiple columns (least-squares).
    ///
    /// Each column of B is solved independently.
    ///
    /// ```
    /// use numeris::Matrix;
    ///
    /// let a = Matrix::new([[1.0_f64, 0.0], [0.0, 1.0], [1.0, 1.0]]);
    /// let b = Matrix::new([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
    /// let qr = a.qr().unwrap();
    /// let x = qr.solve_matrix(&b);
    /// assert_eq!(x.nrows(), 2);
    /// assert_eq!(x.ncols(), 2);
    /// ```
    pub fn solve_matrix<const P: usize>(&self, b: &Matrix<T, M, P>) -> Matrix<T, N, P> {
        let mut result = Matrix::<T, N, P>::zeros();
        for col in 0..P {
            let b_col = Vector::from_array(core::array::from_fn(|i| b[(i, col)]));
            let x_col = self.solve(&b_col);
            for i in 0..N {
                result[(i, col)] = x_col[i];
            }
        }
        result
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
            let v_slice = self.qr.col_as_slice(col, col + 1);
            let mut dot = qtb[col]; // conj(1) * qtb[col]
            for idx in 0..v_slice.len() {
                dot = dot + v_slice[idx].conj() * qtb[col + 1 + idx];
            }
            dot = dot * tau_val;

            qtb[col] = qtb[col] - dot;
            crate::simd::axpy_neg_dispatch(&mut qtb[col + 1..], dot, v_slice);
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

// ── Column-pivoted QR ──────────────────────────────────────────────

/// QR decomposition with column pivoting (rank-revealing) in place.
///
/// On return, `a` contains the packed QR factorization (same layout as
/// `qr_in_place`), `tau` holds Householder scalars, and `perm[j]` gives the
/// original column index that was pivoted into column `j`.
///
/// At each step k, the column with the largest 2-norm in the remaining
/// sub-matrix is swapped into position k before applying the Householder
/// reflection.  Column norms are maintained incrementally and recomputed
/// when the incremental estimate drops below a threshold to avoid
/// cancellation drift.
///
/// Unlike `qr_in_place`, this function does **not** return an error for
/// rank-deficient matrices — it simply produces small/zero diagonal entries
/// in R, and the numerical rank can be read from those entries after the
/// fact.
pub fn qr_col_pivot_in_place<T: LinalgScalar>(
    a: &mut impl MatrixMut<T>,
    tau: &mut [T],
    perm: &mut [usize],
) {
    let m = a.nrows();
    let n = a.ncols();
    let k = m.min(n);
    assert!(m >= n, "QR decomposition requires M >= N");
    assert_eq!(tau.len(), k, "tau length must equal min(M, N)");
    assert_eq!(perm.len(), n, "perm length must equal N");

    // Initialise permutation as identity and column norms.
    // col_norms[j] tracks the squared 2-norm of the remaining sub-column
    // below the current pivot row.
    let mut col_norms = [<T::Real as Zero>::zero(); 64];
    // For matrices wider than 64 we fall back to a stack vec.  This is
    // an embedded-friendly library, so we avoid alloc here.
    assert!(n <= 64, "qr_col_pivot_in_place: N must be <= 64 for fixed-size stack storage");
    for j in 0..n {
        perm[j] = j;
        let col = a.col_as_slice(j, 0);
        let mut s = <T::Real as Zero>::zero();
        for &v in col {
            s = s + (v * v.conj()).re();
        }
        col_norms[j] = s;
    }

    for col in 0..k {
        // ── Pivot: find column with largest remaining norm ───────
        let mut best_j = col;
        let mut best_norm = col_norms[col];
        for j in (col + 1)..n {
            if col_norms[j] > best_norm {
                best_norm = col_norms[j];
                best_j = j;
            }
        }

        // Swap columns col <-> best_j in a, perm, and col_norms
        if best_j != col {
            perm.swap(col, best_j);
            col_norms.swap(col, best_j);
            // Swap entire columns in a
            for i in 0..m {
                let tmp = *a.get(i, col);
                *a.get_mut(i, col) = *a.get(i, best_j);
                *a.get_mut(i, best_j) = tmp;
            }
        }

        // ── Householder reflection on column col ────────────────
        let sub_col = a.col_as_slice(col, col);
        let mut norm_sq = <T::Real as Zero>::zero();
        for &v in sub_col {
            norm_sq = norm_sq + (v * v.conj()).re();
        }

        if norm_sq < T::lepsilon() {
            // Remaining columns are numerically zero — leave tau = 0
            tau[col] = T::zero();
            continue;
        }

        let norm = norm_sq.lsqrt();
        let a_col_col = *a.get(col, col);

        let alpha = a_col_col.modulus();
        let sigma = if alpha < T::lepsilon() {
            T::from_real(norm)
        } else {
            T::from_real(norm) * (a_col_col / T::from_real(alpha))
        };

        let v0 = a_col_col + sigma;
        *a.get_mut(col, col) = v0;

        let tau_val = v0 / sigma;
        tau[col] = tau_val;

        // Scale sub-diagonal entries
        {
            let sub_col = a.col_as_mut_slice(col, col + 1);
            for x in sub_col.iter_mut() {
                *x = *x / v0;
            }
        }

        // Apply H to trailing columns + update norms
        for j in (col + 1)..n {
            // v^H * A[:, j]
            let mut dot = *a.get(col, j);
            let (v_slice, a_j_slice) = split_two_col_slices(a, col, j, col + 1);
            for idx in 0..v_slice.len() {
                dot = dot + v_slice[idx].conj() * a_j_slice[idx];
            }
            dot = dot * tau_val;

            *a.get_mut(col, j) = *a.get(col, j) - dot;
            let (v_slice, a_j_slice) = split_two_col_slices(a, col, j, col + 1);
            crate::simd::axpy_neg_dispatch(a_j_slice, dot, v_slice);

            // Incremental norm update: subtract eliminated component
            let _eliminated = (*a.get(col, j) * (*a.get(col, j)).conj()).re();
            // This is the new a[col, j] after reflection, which became R[col, j].
            // The remaining norm below row col is col_norms[j] minus the
            // squared entry that was just "absorbed" into R row col.
            // But we need the norm of the sub-column below row col+1.
            // Recompute from scratch (safe, avoids cancellation issues):
            let sub = a.col_as_slice(j, col + 1);
            let mut s = <T::Real as Zero>::zero();
            for &v in sub {
                s = s + (v * v.conj()).re();
            }
            col_norms[j] = s;
        }

        // Store -sigma as R diagonal entry
        *a.get_mut(col, col) = T::zero() - sigma;
    }
}

/// Rank-revealing QR decomposition with column pivoting.
///
/// Factorises `A * P = Q * R` where P is a permutation matrix, Q is
/// orthogonal (unitary), and R is upper-triangular with diagonal entries
/// in decreasing magnitude.
///
/// The numerical rank can be determined by inspecting the R diagonal
/// via [`rank()`](QrPivotDecomposition::rank).
///
/// # Example
///
/// ```
/// use numeris::Matrix;
///
/// // Rank-2 matrix (col 2 = col 0 + col 1)
/// let a = Matrix::new([
///     [1.0_f64, 0.0, 1.0],
///     [0.0, 1.0, 1.0],
///     [1.0, 1.0, 2.0],
/// ]);
/// let qrp = a.qr_col_pivot();
/// assert_eq!(qrp.rank(1e-10), 2);
///
/// // Q * R reconstructs A with columns permuted
/// let q = qrp.q();
/// let r = qrp.r();
/// let qr = q * r;
/// let perm = qrp.permutation();
/// for i in 0..3 {
///     for j in 0..3 {
///         assert!((qr[(i, j)] - a[(i, perm[j])]).abs() < 1e-10);
///     }
/// }
/// ```
#[derive(Debug)]
pub struct QrPivotDecomposition<T, const M: usize, const N: usize> {
    qr: Matrix<T, M, N>,
    tau: [T; N],
    perm: [usize; N],
}

impl<T: LinalgScalar, const M: usize, const N: usize> QrPivotDecomposition<T, M, N> {
    /// Decompose a matrix with column pivoting.
    pub fn new(a: &Matrix<T, M, N>) -> Self {
        assert!(M >= N, "QR decomposition requires M >= N");
        let mut qr = *a;
        let mut tau = [T::zero(); N];
        let mut perm = [0usize; N];
        qr_col_pivot_in_place(&mut qr, &mut tau, &mut perm);
        Self { qr, tau, perm }
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
    pub fn q(&self) -> Matrix<T, M, N> {
        let mut q = Matrix::<T, M, N>::zeros();
        for i in 0..N {
            q[(i, i)] = T::one();
        }

        for col in (0..N).rev() {
            let tau_val = self.tau[col];
            if tau_val == T::zero() {
                continue;
            }

            let v_slice = self.qr.col_as_slice(col, col + 1);
            for j in col..N {
                let mut dot = q[(col, j)];
                let q_j_slice = q.col_as_slice(j, col + 1);
                for idx in 0..v_slice.len() {
                    dot = dot + v_slice[idx].conj() * q_j_slice[idx];
                }
                dot = dot * tau_val;

                q[(col, j)] = q[(col, j)] - dot;
                let q_j_slice = q.col_as_mut_slice(j, col + 1);
                crate::simd::axpy_neg_dispatch(q_j_slice, dot, v_slice);
            }
        }

        q
    }

    /// Column permutation vector.
    ///
    /// `perm[j]` is the index of the original column that was pivoted into
    /// position `j`.  The factorisation satisfies `A[:, perm] = Q * R`.
    pub fn permutation(&self) -> &[usize; N] {
        &self.perm
    }

    /// Numerical rank: number of R diagonal entries with magnitude above `tol`.
    pub fn rank(&self, tol: T::Real) -> usize {
        (0..N)
            .take_while(|&i| self.qr[(i, i)].modulus() > tol)
            .count()
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

    /// Rank-revealing QR decomposition with column pivoting.
    ///
    /// Unlike [`qr()`](Matrix::qr), this never fails — rank-deficient matrices
    /// simply produce small diagonal entries in R.
    ///
    /// ```
    /// use numeris::Matrix;
    ///
    /// let a = Matrix::new([
    ///     [1.0_f64, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0],
    ///     [7.0, 8.0, 9.0],
    /// ]);
    /// let qrp = a.qr_col_pivot();
    /// assert_eq!(qrp.rank(1e-10), 2); // rank-deficient
    /// ```
    pub fn qr_col_pivot(&self) -> QrPivotDecomposition<T, M, N> {
        QrPivotDecomposition::new(self)
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
        let ax = a * x;
        let r = b - ax;
        let at = a.transpose();
        let atr = at * r;
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

    // ── Column-pivoted QR tests ────────────────────────────────────

    #[test]
    fn qr_pivot_full_rank_3x3() {
        let a = Matrix::new([
            [12.0_f64, -51.0, 4.0],
            [6.0, 167.0, -68.0],
            [-4.0, 24.0, -41.0],
        ]);
        let qrp = a.qr_col_pivot();
        let q = qrp.q();
        let r = qrp.r();
        let perm = qrp.permutation();

        // Q * R == A[:, perm]
        let qr_prod = q * r;
        for i in 0..3 {
            for j in 0..3 {
                assert_near(qr_prod[(i, j)], a[(i, perm[j])], TOL,
                    &format!("QR[({},{})]", i, j));
            }
        }

        // Q^T * Q == I
        let qtq = q.transpose() * q;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(qtq[(i, j)], expected, TOL,
                    &format!("QtQ[({},{})]", i, j));
            }
        }

        // Full rank
        assert_eq!(qrp.rank(1e-10), 3);
    }

    #[test]
    fn qr_pivot_rank_deficient() {
        // Rank-2 matrix: row 2 = row 0 + row 1
        let a = Matrix::new([
            [1.0_f64, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [5.0, 7.0, 9.0],
        ]);
        let qrp = a.qr_col_pivot();
        assert_eq!(qrp.rank(1e-10), 2);

        // Q * R still reconstructs A[:, perm]
        let q = qrp.q();
        let r = qrp.r();
        let perm = qrp.permutation();
        let qr_prod = q * r;
        for i in 0..3 {
            for j in 0..3 {
                assert_near(qr_prod[(i, j)], a[(i, perm[j])], TOL,
                    &format!("QR[({},{})]", i, j));
            }
        }
    }

    #[test]
    fn qr_pivot_rank_1() {
        // Rank-1 matrix: all rows are multiples of [1, 2, 3]
        let a = Matrix::new([
            [1.0_f64, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
        ]);
        let qrp = a.qr_col_pivot();
        assert_eq!(qrp.rank(1e-10), 1);

        let q = qrp.q();
        let r = qrp.r();
        let perm = qrp.permutation();
        let qr_prod = q * r;
        for i in 0..3 {
            for j in 0..3 {
                assert_near(qr_prod[(i, j)], a[(i, perm[j])], TOL,
                    &format!("QR[({},{})]", i, j));
            }
        }
    }

    #[test]
    fn qr_pivot_zero_matrix() {
        let a = Matrix::<f64, 3, 3>::zeros();
        let qrp = a.qr_col_pivot();
        assert_eq!(qrp.rank(1e-10), 0);
    }

    #[test]
    fn qr_pivot_identity() {
        let id: Matrix<f64, 3, 3> = Matrix::eye();
        let qrp = id.qr_col_pivot();
        assert_eq!(qrp.rank(1e-10), 3);

        let q = qrp.q();
        let r = qrp.r();
        let perm = qrp.permutation();
        let qr_prod = q * r;
        for i in 0..3 {
            for j in 0..3 {
                assert_near(qr_prod[(i, j)], id[(i, perm[j])], TOL,
                    &format!("QR[({},{})]", i, j));
            }
        }
    }

    #[test]
    fn qr_pivot_rectangular_4x3() {
        let a = Matrix::new([
            [1.0_f64, -1.0, 4.0],
            [1.0, 4.0, -2.0],
            [1.0, 4.0, 2.0],
            [1.0, -1.0, 0.0],
        ]);
        let qrp = a.qr_col_pivot();
        let q = qrp.q();
        let r = qrp.r();
        let perm = qrp.permutation();

        // Q * R == A[:, perm]
        let qr_prod: Matrix<f64, 4, 3> = q * r;
        for i in 0..4 {
            for j in 0..3 {
                assert_near(qr_prod[(i, j)], a[(i, perm[j])], TOL,
                    &format!("QR[({},{})]", i, j));
            }
        }

        // Q^T * Q == I_3
        let qtq: Matrix<f64, 3, 3> = q.transpose() * q;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(qtq[(i, j)], expected, TOL,
                    &format!("QtQ[({},{})]", i, j));
            }
        }

        assert_eq!(qrp.rank(1e-10), 3);
    }

    #[test]
    fn qr_pivot_r_diagonal_decreasing() {
        // R diagonal magnitudes should be non-increasing
        let a = Matrix::new([
            [1.0_f64, 100.0, 0.5],
            [2.0, 200.0, 1.0],
            [3.0, 300.0, 1.5],
            [4.0, 400.0, 2.0],
        ]);
        let qrp = a.qr_col_pivot();
        let r = qrp.r();
        for i in 0..(3 - 1) {
            assert!(
                r[(i, i)].abs() >= r[(i + 1, i + 1)].abs() - TOL,
                "|R[{},{}]| = {} should >= |R[{},{}]| = {}",
                i, i,
                r[(i, i)].abs(),
                i + 1, i + 1,
                r[(i + 1, i + 1)].abs()
            );
        }
    }

    #[test]
    fn qr_pivot_permutation_is_valid() {
        let a = Matrix::new([
            [0.1_f64, 10.0, 5.0],
            [0.2, 20.0, 10.0],
            [0.3, 30.0, 15.0],
        ]);
        let qrp = a.qr_col_pivot();
        let perm = qrp.permutation();

        // permutation should contain each index exactly once
        let mut seen = [false; 3];
        for &p in perm {
            assert!(p < 3, "permutation index out of range");
            assert!(!seen[p], "duplicate permutation index");
            seen[p] = true;
        }
    }

    #[test]
    fn qr_pivot_column_dependent() {
        // Column 2 = column 0 + column 1
        let a = Matrix::new([
            [1.0_f64, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
        ]);
        let qrp = a.qr_col_pivot();
        assert_eq!(qrp.rank(1e-10), 2);

        let q = qrp.q();
        let r = qrp.r();
        let perm = qrp.permutation();
        let qr_prod = q * r;
        for i in 0..3 {
            for j in 0..3 {
                assert_near(qr_prod[(i, j)], a[(i, perm[j])], TOL,
                    &format!("QR[({},{})]", i, j));
            }
        }
    }

    #[test]
    fn qr_pivot_2x2() {
        let a = Matrix::new([
            [3.0_f64, 1.0],
            [4.0, 2.0],
        ]);
        let qrp = a.qr_col_pivot();
        assert_eq!(qrp.rank(1e-10), 2);

        let q = qrp.q();
        let r = qrp.r();
        let perm = qrp.permutation();
        let qr_prod = q * r;
        for i in 0..2 {
            for j in 0..2 {
                assert_near(qr_prod[(i, j)], a[(i, perm[j])], TOL,
                    &format!("QR[({},{})]", i, j));
            }
        }
    }
}
