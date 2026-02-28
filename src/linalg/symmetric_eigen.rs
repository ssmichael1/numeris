use crate::linalg::LinalgError;
use crate::traits::{LinalgScalar, MatrixMut, MatrixRef};
use crate::Matrix;
use num_traits::{Float, One, Zero};

/// Householder tridiagonalization: reduce a symmetric (Hermitian) matrix to
/// tridiagonal form via similarity transforms.
///
/// On return:
/// - `diag[0..n]` contains the diagonal of the tridiagonal matrix
/// - `off_diag[0..n-1]` contains the sub-diagonal (off_diag[i] = T_{i+1,i})
/// - `q` accumulates the orthogonal/unitary transform Q such that Q^H A Q = T
///
/// The input `a` is read but not modified.
pub fn tridiagonalize<T: LinalgScalar>(
    a: &impl MatrixRef<T>,
    diag: &mut [T::Real],
    off_diag: &mut [T::Real],
    q: &mut impl MatrixMut<T>,
) {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "tridiagonalize requires a square matrix");
    assert!(diag.len() >= n);
    assert!(off_diag.len() + 1 >= n);

    // Working copy of the matrix (flat row-major)
    let mut w = alloc_work::<T>(n);
    for i in 0..n {
        for j in 0..n {
            w[i * n + j] = *a.get(i, j);
        }
    }

    // Initialize Q = I
    for i in 0..n {
        for j in 0..n {
            *q.get_mut(i, j) = if i == j { T::one() } else { T::zero() };
        }
    }

    for k in 0..n.saturating_sub(2) {
        // Form Householder vector from w[k+1:n, k]
        let mut norm_sq = <T::Real as Zero>::zero();
        for i in (k + 1)..n {
            let v = w[i * n + k];
            norm_sq = norm_sq + (v * v.conj()).re();
        }

        if norm_sq <= T::lepsilon() * T::lepsilon() {
            off_diag[k] = <T::Real as Zero>::zero();
            continue;
        }

        let norm = norm_sq.lsqrt();
        let wk1k = w[(k + 1) * n + k];
        let alpha = wk1k.modulus();

        let sigma = if alpha < T::lepsilon() {
            T::from_real(norm)
        } else {
            T::from_real(norm) * (wk1k / T::from_real(alpha))
        };

        let v0 = wk1k + sigma;

        // ||v||^2
        let v0_sq = (v0 * v0.conj()).re();
        let mut v_norm_sq = v0_sq;
        for i in (k + 2)..n {
            let vi = w[i * n + k];
            v_norm_sq = v_norm_sq + (vi * vi.conj()).re();
        }

        let two = <T::Real as One>::one() + <T::Real as One>::one();
        let tau_real = two / v_norm_sq;
        let tau = T::from_real(tau_real);

        // Compute p = tau * A_sub * v, where A_sub = w[k+1:n, k+1:n]
        let sub_n = n - k - 1;
        let mut p = alloc_work_vec::<T>(sub_n);

        for i in 0..sub_n {
            let row = k + 1 + i;
            let mut dot = T::zero();
            for jj in 0..sub_n {
                let col = k + 1 + jj;
                let vj = if jj == 0 { v0 } else { w[(k + 1 + jj) * n + k] };
                dot = dot + w[row * n + col] * vj;
            }
            p[i] = tau * dot;
        }

        // Compute v^H * p
        let mut vhp = T::zero();
        vhp = vhp + v0.conj() * p[0];
        for i in 1..sub_n {
            let vi = w[(k + 1 + i) * n + k];
            vhp = vhp + vi.conj() * p[i];
        }

        // q_vec = p - (tau/2)(v^H p) v
        let half_tau_vhp = T::from_real(tau_real / two) * vhp;
        let mut q_vec = alloc_work_vec::<T>(sub_n);
        q_vec[0] = p[0] - half_tau_vhp * v0;
        for i in 1..sub_n {
            let vi = w[(k + 1 + i) * n + k];
            q_vec[i] = p[i] - half_tau_vhp * vi;
        }

        // Rank-2 update: A_sub -= v * q_vec^H + q_vec * v^H
        for i in 0..sub_n {
            let vi = if i == 0 { v0 } else { w[(k + 1 + i) * n + k] };
            for j in 0..sub_n {
                let row = k + 1 + i;
                let col = k + 1 + j;
                let vj = if j == 0 { v0 } else { w[(k + 1 + j) * n + k] };
                w[row * n + col] =
                    w[row * n + col] - vi * q_vec[j].conj() - q_vec[i] * vj.conj();
            }
        }

        // Off-diagonal entry T[k+1,k] = -sigma (result of Householder reflection).
        // sigma = sign(wk1k) * norm, so -sigma = -sign(wk1k) * norm.
        // Store the actual signed value (not absolute) for eigenvector consistency.
        let neg_sigma_re = (T::zero() - sigma).re();
        off_diag[k] = neg_sigma_re;

        // Accumulate Q: Q_new = Q * (I - tau * v * v^H)
        for row in 0..n {
            let mut s = T::zero();
            s = s + *q.get(row, k + 1) * v0;
            for m in 1..sub_n {
                let vm = w[(k + 1 + m) * n + k];
                s = s + *q.get(row, k + 1 + m) * vm;
            }
            s = tau * s;

            *q.get_mut(row, k + 1) = *q.get(row, k + 1) - s * v0.conj();
            for j in 1..sub_n {
                let vj = w[(k + 1 + j) * n + k];
                *q.get_mut(row, k + 1 + j) = *q.get(row, k + 1 + j) - s * vj.conj();
            }
        }
    }

    // Extract diagonal
    for i in 0..n {
        diag[i] = w[i * n + i].re();
    }
    // Last off-diagonal entry (for n >= 2): read from the rank-2-updated subblock.
    if n >= 2 {
        off_diag[n - 2] = w[(n - 1) * n + (n - 2)].re();
    }
}

/// Implicit QR iteration with Wilkinson shift on a symmetric tridiagonal matrix.
/// Accumulates Givens rotations into `q`.
///
/// - `diag[0..n]`: diagonal entries (overwritten with eigenvalues, sorted ascending)
/// - `off_diag[0..n-1]`: sub-diagonal entries (destroyed)
/// - `q`: eigenvector matrix to accumulate rotations into
/// - `max_iter`: maximum total iterations
pub fn tridiagonal_qr_with_vecs<T: LinalgScalar>(
    diag: &mut [T::Real],
    off_diag: &mut [T::Real],
    q: &mut impl MatrixMut<T>,
    max_iter: usize,
) -> Result<(), LinalgError> {
    let n = diag.len();
    if n <= 1 {
        return Ok(());
    }

    let eps = T::lepsilon();
    let mut iter = 0usize;
    let mut hi = n - 1;

    while hi > 0 {
        let mut lo = hi;
        while lo > 0 {
            let threshold = eps * (diag[lo - 1].abs() + diag[lo].abs());
            if off_diag[lo - 1].abs() <= threshold {
                off_diag[lo - 1] = <T::Real as Zero>::zero();
                break;
            }
            lo -= 1;
        }

        if lo == hi {
            hi -= 1;
            continue;
        }

        iter += 1;
        if iter > max_iter {
            return Err(LinalgError::ConvergenceFailure);
        }

        // Wilkinson shift
        let two = <T::Real as One>::one() + <T::Real as One>::one();
        let d = (diag[hi - 1] - diag[hi]) / two;
        let e = off_diag[hi - 1];
        let r = (d * d + e * e).sqrt();
        let shift = diag[hi]
            - e * e
                / (d + if d >= <T::Real as Zero>::zero() {
                    r
                } else {
                    <T::Real as Zero>::zero() - r
                });

        let mut x = diag[lo] - shift;
        let mut z = off_diag[lo];

        for k in lo..hi {
            // Givens rotation: G * [x; z] = [r; 0]
            // where G = [[c, s], [-s, c]].
            let (c, s) = givens(x, z);

            if k > lo {
                off_diag[k - 1] = c * x + s * z;
            }

            // Apply similarity transform T' = G * T * G^T.
            let d_k = diag[k];
            let d_k1 = diag[k + 1];
            let e_k = off_diag[k];

            diag[k] = c * c * d_k + two * c * s * e_k + s * s * d_k1;
            diag[k + 1] = s * s * d_k - two * c * s * e_k + c * c * d_k1;
            off_diag[k] = c * s * (d_k1 - d_k) + (c * c - s * s) * e_k;

            if k + 1 < hi {
                // Left multiply G creates bulge at (k, k+2):
                // T'[k, k+2] = s * off_diag[k+1]
                // T'[k+1, k+2] = c * off_diag[k+1]
                let e_next = off_diag[k + 1];
                x = off_diag[k];
                z = s * e_next;
                off_diag[k + 1] = c * e_next;
            }

            // Accumulate eigenvectors: Q_new = Q * G^T
            // G^T = [[c, -s], [s, c]]
            let n_rows = q.nrows();
            for i in 0..n_rows {
                let qik = *q.get(i, k);
                let qik1 = *q.get(i, k + 1);
                *q.get_mut(i, k) = T::from_real(c) * qik + T::from_real(s) * qik1;
                *q.get_mut(i, k + 1) = T::from_real(c) * qik1 - T::from_real(s) * qik;
            }
        }
    }

    // Sort eigenvalues ascending, permute eigenvector columns
    sort_eigen_with_vecs::<T>(diag, q);

    Ok(())
}

/// Implicit QR iteration without eigenvector accumulation (eigenvalues only).
pub fn tridiagonal_qr_no_vecs<T: LinalgScalar>(
    diag: &mut [T::Real],
    off_diag: &mut [T::Real],
    max_iter: usize,
) -> Result<(), LinalgError> {
    let n = diag.len();
    if n <= 1 {
        return Ok(());
    }

    let eps = T::lepsilon();
    let mut iter = 0usize;
    let mut hi = n - 1;

    while hi > 0 {
        let mut lo = hi;
        while lo > 0 {
            let threshold = eps * (diag[lo - 1].abs() + diag[lo].abs());
            if off_diag[lo - 1].abs() <= threshold {
                off_diag[lo - 1] = <T::Real as Zero>::zero();
                break;
            }
            lo -= 1;
        }

        if lo == hi {
            hi -= 1;
            continue;
        }

        iter += 1;
        if iter > max_iter {
            return Err(LinalgError::ConvergenceFailure);
        }

        let two = <T::Real as One>::one() + <T::Real as One>::one();
        let d = (diag[hi - 1] - diag[hi]) / two;
        let e = off_diag[hi - 1];
        let r = (d * d + e * e).sqrt();
        let shift = diag[hi]
            - e * e
                / (d + if d >= <T::Real as Zero>::zero() {
                    r
                } else {
                    <T::Real as Zero>::zero() - r
                });

        let mut x = diag[lo] - shift;
        let mut z = off_diag[lo];

        for k in lo..hi {
            let (c, s) = givens(x, z);

            if k > lo {
                off_diag[k - 1] = c * x + s * z;
            }

            let d_k = diag[k];
            let d_k1 = diag[k + 1];
            let e_k = off_diag[k];

            diag[k] = c * c * d_k + two * c * s * e_k + s * s * d_k1;
            diag[k + 1] = s * s * d_k - two * c * s * e_k + c * c * d_k1;
            off_diag[k] = c * s * (d_k1 - d_k) + (c * c - s * s) * e_k;

            if k + 1 < hi {
                let e_next = off_diag[k + 1];
                x = off_diag[k];
                z = s * e_next;
                off_diag[k + 1] = c * e_next;
            }
        }
    }

    // Sort eigenvalues ascending (no eigenvector columns to permute)
    sort_eigenvalues(diag);

    Ok(())
}

/// Givens rotation: compute (c, s) such that [c, s; -s, c] * [a; b] = [r; 0].
#[inline]
pub(crate) fn givens<R: Float + Zero>(a: R, b: R) -> (R, R) {
    if b == R::zero() {
        (R::one(), R::zero())
    } else if b.abs() > a.abs() {
        let t = a / b;
        let s = R::one() / (R::one() + t * t).sqrt();
        (s * t, s)
    } else {
        let t = b / a;
        let c = R::one() / (R::one() + t * t).sqrt();
        (c, c * t)
    }
}

/// Sort eigenvalues ascending and permute eigenvector columns.
fn sort_eigen_with_vecs<T: LinalgScalar>(
    diag: &mut [T::Real],
    q: &mut impl MatrixMut<T>,
) {
    let n = diag.len();
    for i in 0..n {
        let mut min_idx = i;
        for j in (i + 1)..n {
            if diag[j] < diag[min_idx] {
                min_idx = j;
            }
        }
        if min_idx != i {
            diag.swap(i, min_idx);
            let n_rows = q.nrows();
            for row in 0..n_rows {
                let tmp = *q.get(row, i);
                *q.get_mut(row, i) = *q.get(row, min_idx);
                *q.get_mut(row, min_idx) = tmp;
            }
        }
    }
}

/// Sort eigenvalues ascending (no eigenvectors).
fn sort_eigenvalues<R: Float>(diag: &mut [R]) {
    let n = diag.len();
    for i in 0..n {
        let mut min_idx = i;
        for j in (i + 1)..n {
            if diag[j] < diag[min_idx] {
                min_idx = j;
            }
        }
        if min_idx != i {
            diag.swap(i, min_idx);
        }
    }
}

// Workspace allocation helpers
#[cfg(feature = "alloc")]
fn alloc_work<T: LinalgScalar>(n: usize) -> alloc::vec::Vec<T> {
    alloc::vec![T::zero(); n * n]
}

#[cfg(feature = "alloc")]
fn alloc_work_vec<T: LinalgScalar>(n: usize) -> alloc::vec::Vec<T> {
    alloc::vec![T::zero(); n]
}

#[cfg(not(feature = "alloc"))]
fn alloc_work<T: LinalgScalar>(n: usize) -> [T; 36] {
    assert!(n <= 6, "tridiagonalize without alloc supports at most 6×6");
    [T::zero(); 36]
}

#[cfg(not(feature = "alloc"))]
fn alloc_work_vec<T: LinalgScalar>(n: usize) -> [T; 6] {
    assert!(n <= 6, "tridiagonalize without alloc supports at most 6×6");
    [T::zero(); 6]
}

/// Symmetric/Hermitian eigendecomposition of a fixed-size square matrix.
///
/// Computes all eigenvalues and eigenvectors of a real symmetric or
/// complex Hermitian matrix using Householder tridiagonalization followed
/// by implicit QR iteration with Wilkinson shifts.
///
/// Eigenvalues are sorted in ascending order. Eigenvectors are the columns
/// of the orthogonal/unitary matrix Q.
///
/// # Example
///
/// ```
/// use numeris::Matrix;
/// use numeris::linalg::SymmetricEigen;
///
/// let a = Matrix::new([
///     [2.0_f64, -1.0],
///     [-1.0, 2.0],
/// ]);
/// let eig = SymmetricEigen::new(&a).unwrap();
/// assert!((eig.eigenvalues()[0] - 1.0).abs() < 1e-10);
/// assert!((eig.eigenvalues()[1] - 3.0).abs() < 1e-10);
///
/// // Verify A * v ≈ λ * v for first eigenpair
/// let q = eig.eigenvectors();
/// for i in 0..2 {
///     let av = a[(i, 0)] * q[(0, 0)] + a[(i, 1)] * q[(1, 0)];
///     assert!((av - eig.eigenvalues()[0] * q[(i, 0)]).abs() < 1e-10);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SymmetricEigen<T: LinalgScalar, const N: usize> {
    eigenvalues: [T::Real; N],
    eigenvectors: Matrix<T, N, N>,
}

impl<T: LinalgScalar, const N: usize> SymmetricEigen<T, N> {
    /// Decompose a symmetric (Hermitian) matrix.
    ///
    /// Returns `Err(ConvergenceFailure)` if QR iteration does not converge.
    pub fn new(a: &Matrix<T, N, N>) -> Result<Self, LinalgError> {
        let mut diag = [<T::Real as Zero>::zero(); N];
        let mut off_diag = [<T::Real as Zero>::zero(); N];
        let mut q = Matrix::<T, N, N>::zeros();

        if N == 0 {
            return Ok(Self {
                eigenvalues: diag,
                eigenvectors: q,
            });
        }

        tridiagonalize(a, &mut diag, &mut off_diag, &mut q);
        tridiagonal_qr_with_vecs::<T>(
            &mut diag,
            &mut off_diag[..N.saturating_sub(1)],
            &mut q,
            30 * N,
        )?;

        Ok(Self {
            eigenvalues: diag,
            eigenvectors: q,
        })
    }

    /// Compute eigenvalues only (faster, no eigenvector accumulation).
    pub fn eigenvalues_only(a: &Matrix<T, N, N>) -> Result<[T::Real; N], LinalgError> {
        let mut diag = [<T::Real as Zero>::zero(); N];
        let mut off_diag = [<T::Real as Zero>::zero(); N];
        let mut q = Matrix::<T, N, N>::zeros();

        if N == 0 {
            return Ok(diag);
        }

        tridiagonalize(a, &mut diag, &mut off_diag, &mut q);
        tridiagonal_qr_no_vecs::<T>(&mut diag, &mut off_diag[..N.saturating_sub(1)], 30 * N)?;

        Ok(diag)
    }

    /// The eigenvalues, sorted ascending.
    #[inline]
    pub fn eigenvalues(&self) -> &[T::Real; N] {
        &self.eigenvalues
    }

    /// The eigenvector matrix Q (columns are eigenvectors).
    #[inline]
    pub fn eigenvectors(&self) -> &Matrix<T, N, N> {
        &self.eigenvectors
    }
}

/// Convenience methods for symmetric/Hermitian eigendecomposition.
impl<T: LinalgScalar, const N: usize> Matrix<T, N, N> {
    /// Symmetric/Hermitian eigendecomposition.
    ///
    /// Returns eigenvalues (ascending) and eigenvectors (as columns of Q).
    /// The caller is responsible for ensuring the matrix is symmetric/Hermitian.
    ///
    /// ```
    /// use numeris::Matrix;
    ///
    /// let a = Matrix::new([[5.0_f64, 2.0], [2.0, 2.0]]);
    /// let eig = a.eig_symmetric().unwrap();
    /// let vals = eig.eigenvalues();
    /// assert!((vals[0] - 1.0).abs() < 1e-10);
    /// assert!((vals[1] - 6.0).abs() < 1e-10);
    /// ```
    pub fn eig_symmetric(&self) -> Result<SymmetricEigen<T, N>, LinalgError> {
        SymmetricEigen::new(self)
    }

    /// Eigenvalues of a symmetric/Hermitian matrix (no eigenvectors).
    ///
    /// ```
    /// use numeris::Matrix;
    ///
    /// let a = Matrix::new([[3.0_f64, 1.0], [1.0, 3.0]]);
    /// let vals = a.eigenvalues_symmetric().unwrap();
    /// assert!((vals[0] - 2.0).abs() < 1e-10);
    /// assert!((vals[1] - 4.0).abs() < 1e-10);
    /// ```
    pub fn eigenvalues_symmetric(&self) -> Result<[T::Real; N], LinalgError> {
        SymmetricEigen::eigenvalues_only(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn assert_near(a: f64, b: f64, tol: f64, msg: &str) {
        assert!(
            (a - b).abs() < tol,
            "{}: {} vs {} (diff {})",
            msg,
            a,
            b,
            (a - b).abs()
        );
    }

    #[test]
    fn identity_eigenvalues() {
        let id: Matrix<f64, 3, 3> = Matrix::eye();
        let eig = id.eig_symmetric().unwrap();
        for i in 0..3 {
            assert_near(eig.eigenvalues()[i], 1.0, TOL, &format!("λ[{}]", i));
        }
        let q = eig.eigenvectors();
        let qtq = q.transpose() * *q;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(qtq[(i, j)], expected, TOL, &format!("QtQ[({},{})]", i, j));
            }
        }
    }

    #[test]
    fn diagonal_matrix() {
        let a = Matrix::new([
            [3.0_f64, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ]);
        let eig = a.eig_symmetric().unwrap();
        assert_near(eig.eigenvalues()[0], 1.0, TOL, "λ[0]");
        assert_near(eig.eigenvalues()[1], 2.0, TOL, "λ[1]");
        assert_near(eig.eigenvalues()[2], 3.0, TOL, "λ[2]");
    }

    #[test]
    fn known_2x2() {
        let a = Matrix::new([[2.0_f64, -1.0], [-1.0, 2.0]]);
        let eig = a.eig_symmetric().unwrap();
        assert_near(eig.eigenvalues()[0], 1.0, TOL, "λ[0]");
        assert_near(eig.eigenvalues()[1], 3.0, TOL, "λ[1]");
    }

    #[test]
    fn known_3x3_eigenvectors() {
        let a = Matrix::new([
            [2.0_f64, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ]);
        let eig = a.eig_symmetric().unwrap();
        let q = eig.eigenvectors();

        for col in 0..3 {
            let lambda = eig.eigenvalues()[col];
            for row in 0..3 {
                let mut av = 0.0;
                for k in 0..3 {
                    av += a[(row, k)] * q[(k, col)];
                }
                assert_near(
                    av,
                    lambda * q[(row, col)],
                    TOL,
                    &format!("Av=λv [({},{})]", row, col),
                );
            }
        }
    }

    #[test]
    fn reconstruction() {
        let a = Matrix::new([
            [4.0_f64, 1.0, -1.0],
            [1.0, 3.0, 2.0],
            [-1.0, 2.0, 5.0],
        ]);
        let eig = a.eig_symmetric().unwrap();
        let q = eig.eigenvectors();
        let vals = eig.eigenvalues();

        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += q[(i, k)] * vals[k] * q[(j, k)];
                }
                assert_near(sum, a[(i, j)], TOL, &format!("A[({},{})]", i, j));
            }
        }
    }

    #[test]
    fn orthogonality() {
        let a = Matrix::new([
            [4.0_f64, 1.0, -1.0],
            [1.0, 3.0, 2.0],
            [-1.0, 2.0, 5.0],
        ]);
        let eig = a.eig_symmetric().unwrap();
        let q = eig.eigenvectors();
        let qtq = q.transpose() * *q;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(qtq[(i, j)], expected, TOL, &format!("QtQ[({},{})]", i, j));
            }
        }
    }

    #[test]
    fn sorted_ascending() {
        let a = Matrix::new([
            [10.0_f64, 3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 7.0, 2.0],
            [0.0, 0.0, 2.0, 4.0],
        ]);
        let eig = a.eig_symmetric().unwrap();
        let vals = eig.eigenvalues();
        for i in 0..3 {
            assert!(
                vals[i] <= vals[i + 1] + TOL,
                "not ascending: λ[{}]={} > λ[{}]={}",
                i,
                vals[i],
                i + 1,
                vals[i + 1]
            );
        }
    }

    #[test]
    fn repeated_eigenvalues() {
        let a = Matrix::new([
            [2.0_f64, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]);
        let eig = a.eig_symmetric().unwrap();
        for i in 0..3 {
            assert_near(eig.eigenvalues()[i], 2.0, TOL, &format!("λ[{}]", i));
        }
    }

    #[test]
    fn negative_eigenvalues() {
        let a = Matrix::new([[1.0_f64, 3.0], [3.0, 1.0]]);
        let eig = a.eig_symmetric().unwrap();
        assert_near(eig.eigenvalues()[0], -2.0, TOL, "λ[0]");
        assert_near(eig.eigenvalues()[1], 4.0, TOL, "λ[1]");
    }

    #[test]
    fn eigenvalues_only() {
        let a = Matrix::new([[5.0_f64, 2.0], [2.0, 2.0]]);
        let vals = a.eigenvalues_symmetric().unwrap();
        assert_near(vals[0], 1.0, TOL, "λ[0]");
        assert_near(vals[1], 6.0, TOL, "λ[1]");
    }

    #[test]
    fn f32_support() {
        let a = Matrix::new([[2.0_f32, -1.0], [-1.0, 2.0]]);
        let eig = a.eig_symmetric().unwrap();
        assert!((eig.eigenvalues()[0] - 1.0).abs() < 1e-5);
        assert!((eig.eigenvalues()[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn size_1x1() {
        let a = Matrix::new([[7.0_f64]]);
        let eig = a.eig_symmetric().unwrap();
        assert_near(eig.eigenvalues()[0], 7.0, TOL, "λ[0]");
    }

    #[test]
    fn size_5x5() {
        let a = Matrix::new([
            [5.0_f64, 1.0, 0.5, 0.25, 0.125],
            [1.0, 4.0, 1.0, 0.5, 0.25],
            [0.5, 1.0, 3.0, 1.0, 0.5],
            [0.25, 0.5, 1.0, 2.0, 1.0],
            [0.125, 0.25, 0.5, 1.0, 1.0],
        ]);
        let eig = a.eig_symmetric().unwrap();
        let q = eig.eigenvectors();

        for col in 0..5 {
            let lambda = eig.eigenvalues()[col];
            for row in 0..5 {
                let mut av = 0.0;
                for k in 0..5 {
                    av += a[(row, k)] * q[(k, col)];
                }
                assert_near(
                    av,
                    lambda * q[(row, col)],
                    1e-9,
                    &format!("Av=λv [({},{})]", row, col),
                );
            }
        }

        let qtq = q.transpose() * *q;
        for i in 0..5 {
            for j in 0..5 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(qtq[(i, j)], expected, 1e-9, &format!("QtQ[({},{})]", i, j));
            }
        }

        let trace = a[(0, 0)] + a[(1, 1)] + a[(2, 2)] + a[(3, 3)] + a[(4, 4)];
        let eig_sum: f64 = eig.eigenvalues().iter().sum();
        assert_near(eig_sum, trace, TOL, "trace");
    }

    #[cfg(feature = "complex")]
    mod complex_tests {
        use super::*;
        use num_complex::Complex;

        #[test]
        fn hermitian_real_eigenvalues() {
            let a = Matrix::new([
                [Complex::new(3.0_f64, 0.0), Complex::new(1.0, -1.0)],
                [Complex::new(1.0, 1.0), Complex::new(2.0, 0.0)],
            ]);
            let eig = a.eig_symmetric().unwrap();

            let expected_0 = (5.0 - 5.0_f64.sqrt()) / 2.0;
            let expected_1 = (5.0 + 5.0_f64.sqrt()) / 2.0;
            assert_near(eig.eigenvalues()[0], expected_0, TOL, "λ[0]");
            assert_near(eig.eigenvalues()[1], expected_1, TOL, "λ[1]");

            let q = eig.eigenvectors();
            for i in 0..2 {
                for j in 0..2 {
                    let mut dot = Complex::new(0.0, 0.0);
                    for k in 0..2 {
                        dot = dot + q[(k, i)].conj() * q[(k, j)];
                    }
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert_near(dot.re, expected, TOL, &format!("QHQ[({},{})] re", i, j));
                    assert_near(dot.im, 0.0, TOL, &format!("QHQ[({},{})] im", i, j));
                }
            }
        }
    }
}
