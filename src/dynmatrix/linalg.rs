use alloc::vec;
use alloc::vec::Vec;

use crate::linalg::LinalgError;
use crate::linalg::lu::{lu_in_place, lu_solve};
use crate::linalg::cholesky::{cholesky_in_place, forward_substitute, back_substitute_lt};
use crate::linalg::qr::qr_in_place;
use crate::linalg::symmetric_eigen::{tridiagonalize, tridiagonal_qr_with_vecs, tridiagonal_qr_no_vecs};
use crate::linalg::hessenberg::hessenberg;
use crate::linalg::schur::francis_qr;
use crate::linalg::svd::{bidiagonalize, bidiagonal_qr};
use crate::traits::{FloatScalar, LinalgScalar};
use num_traits::Float;

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
/// let a = DynMatrix::from_rows(2, 2, &[2.0_f64, 1.0, 5.0, 3.0]);
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
/// let a = DynMatrix::from_rows(2, 2, &[4.0_f64, 2.0, 2.0, 3.0]);
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

    /// Determinant: det(A) = product(L\[i\]\[i\])^2.
    pub fn det(&self) -> T {
        let n = self.l.nrows();
        let mut prod = T::one();
        for i in 0..n {
            prod = prod * self.l[(i, i)];
        }
        prod * prod
    }

    /// Log-determinant: ln(det(A)) = 2 * sum(ln(L\[i\]\[i\])).
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
/// let a = DynMatrix::from_rows(3, 2, &[
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

// ── DynSymmetricEigen ──────────────────────────────────────────────

/// Symmetric/Hermitian eigendecomposition of a dynamically-sized square matrix.
///
/// Eigenvalues are sorted ascending. Eigenvectors are columns of Q.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
///
/// let a = DynMatrix::from_rows(2, 2, &[2.0_f64, -1.0, -1.0, 2.0]);
/// let eig = a.eig_symmetric().unwrap();
/// assert!((eig.eigenvalues()[0] - 1.0).abs() < 1e-10);
/// assert!((eig.eigenvalues()[1] - 3.0).abs() < 1e-10);
/// ```
#[derive(Debug)]
pub struct DynSymmetricEigen<T: LinalgScalar> {
    eigenvalues: Vec<T::Real>,
    eigenvectors: DynMatrix<T>,
}

impl<T: LinalgScalar> DynSymmetricEigen<T> {
    /// Decompose a symmetric (Hermitian) matrix.
    pub fn new(a: &DynMatrix<T>) -> Result<Self, LinalgError> {
        assert!(a.is_square(), "symmetric eigen requires a square matrix");
        let n = a.nrows();

        if n == 0 {
            return Ok(Self {
                eigenvalues: Vec::new(),
                eigenvectors: DynMatrix::zeros(0, 0, T::zero()),
            });
        }

        let mut diag = vec![<T::Real as num_traits::Zero>::zero(); n];
        let mut off_diag = vec![<T::Real as num_traits::Zero>::zero(); n];
        let mut q = DynMatrix::zeros(n, n, T::zero());

        tridiagonalize(a, &mut diag, &mut off_diag, &mut q);
        tridiagonal_qr_with_vecs::<T>(
            &mut diag,
            &mut off_diag[..n.saturating_sub(1)],
            &mut q,
            30 * n,
        )?;

        Ok(Self {
            eigenvalues: diag,
            eigenvectors: q,
        })
    }

    /// Compute eigenvalues only (no eigenvectors).
    pub fn eigenvalues_only(a: &DynMatrix<T>) -> Result<Vec<T::Real>, LinalgError> {
        assert!(a.is_square(), "symmetric eigen requires a square matrix");
        let n = a.nrows();

        if n == 0 {
            return Ok(Vec::new());
        }

        let mut diag = vec![<T::Real as num_traits::Zero>::zero(); n];
        let mut off_diag = vec![<T::Real as num_traits::Zero>::zero(); n];
        let mut q = DynMatrix::zeros(n, n, T::zero());

        tridiagonalize(a, &mut diag, &mut off_diag, &mut q);
        tridiagonal_qr_no_vecs::<T>(&mut diag, &mut off_diag[..n.saturating_sub(1)], 30 * n)?;

        Ok(diag)
    }

    /// The eigenvalues, sorted ascending.
    #[inline]
    pub fn eigenvalues(&self) -> &[T::Real] {
        &self.eigenvalues
    }

    /// The eigenvector matrix Q (columns are eigenvectors).
    #[inline]
    pub fn eigenvectors(&self) -> &DynMatrix<T> {
        &self.eigenvectors
    }
}

// ── DynSchur ──────────────────────────────────────────────────────

/// Real Schur decomposition of a dynamically-sized square matrix.
///
/// Computes orthogonal Q and quasi-upper-triangular S such that `A = Q S Q^T`.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
///
/// let a = DynMatrix::from_rows(2, 2, &[0.0_f64, -1.0, 1.0, 0.0]);
/// let schur = a.schur().unwrap();
/// let (re, im) = schur.eigenvalues();
/// assert!(re[0].abs() < 1e-10);
/// assert!((im[0].abs() - 1.0).abs() < 1e-10);
/// ```
#[derive(Debug)]
pub struct DynSchur<T: FloatScalar> {
    s: DynMatrix<T>,
    q: DynMatrix<T>,
}

impl<T: FloatScalar> DynSchur<T> {
    /// Compute the real Schur decomposition.
    pub fn new(a: &DynMatrix<T>) -> Result<Self, LinalgError> {
        assert!(a.is_square(), "Schur decomposition requires a square matrix");
        let n = a.nrows();
        let mut s = a.clone();
        let mut q = DynMatrix::zeros(n, n, T::zero());

        if n <= 1 {
            for i in 0..n {
                q[(i, i)] = T::one();
            }
            return Ok(Self { s, q });
        }

        hessenberg(&mut s, &mut q);
        francis_qr(&mut s, &mut q, 30 * n)?;

        Ok(Self { s, q })
    }

    /// The quasi-upper-triangular Schur form S.
    #[inline]
    pub fn schur_form(&self) -> &DynMatrix<T> {
        &self.s
    }

    /// The orthogonal Schur vectors Q.
    #[inline]
    pub fn schur_vectors(&self) -> &DynMatrix<T> {
        &self.q
    }

    /// Extract eigenvalues as (real_parts, imaginary_parts).
    pub fn eigenvalues(&self) -> (Vec<T>, Vec<T>) {
        let n = self.s.nrows();
        let mut re = vec![T::zero(); n];
        let mut im = vec![T::zero(); n];
        let eps = T::epsilon();

        let mut i = 0;
        while i < n {
            if i + 1 < n && self.s[(i + 1, i)].abs() > eps {
                let a = self.s[(i, i)];
                let b = self.s[(i, i + 1)];
                let c = self.s[(i + 1, i)];
                let d = self.s[(i + 1, i + 1)];

                let half = T::one() / (T::one() + T::one());
                let tr = (a + d) * half;
                let det = a * d - b * c;
                let disc = tr * tr - det;

                if disc >= T::zero() {
                    let sq = disc.sqrt();
                    re[i] = tr + sq;
                    re[i + 1] = tr - sq;
                } else {
                    let sq = (T::zero() - disc).sqrt();
                    re[i] = tr;
                    re[i + 1] = tr;
                    im[i] = sq;
                    im[i + 1] = T::zero() - sq;
                }
                i += 2;
            } else {
                re[i] = self.s[(i, i)];
                i += 1;
            }
        }

        (re, im)
    }
}

// ── DynSvd ─────────────────────────────────────────────────────────

/// Singular value decomposition of a dynamically-sized matrix.
///
/// Computes thin U (M×K), singular values σ (length K = min(M,N),
/// sorted descending), and thin V^T (K×N) such that `A = U · diag(σ) · V^T`.
///
/// Using thin factors saves memory for tall or wide matrices: U is M×K
/// instead of M×M, and V^T is K×N instead of N×N.
///
/// Handles both tall (M ≥ N) and wide (M < N) matrices. For wide
/// matrices, the transpose is decomposed internally and U/V are swapped.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
///
/// let a = DynMatrix::from_rows(3, 2, &[
///     1.0_f64, 0.0,
///     0.0, 1.0,
///     0.0, 0.0,
/// ]);
/// let svd = a.svd().unwrap();
/// assert_eq!(svd.u().nrows(), 3);  // M
/// assert_eq!(svd.u().ncols(), 2);  // min(M,N)
/// assert_eq!(svd.vt().nrows(), 2); // min(M,N)
/// assert_eq!(svd.vt().ncols(), 2); // N
/// assert!((svd.singular_values()[0] - 1.0).abs() < 1e-10);
/// assert!((svd.singular_values()[1] - 1.0).abs() < 1e-10);
/// ```
#[derive(Debug)]
pub struct DynSvd<T: LinalgScalar> {
    u: DynMatrix<T>,
    singular_values: Vec<T::Real>,
    vt: DynMatrix<T>,
}

impl<T: LinalgScalar> DynSvd<T> {
    /// Compute the SVD of a matrix.
    ///
    /// Works for any M×N matrix (both tall and wide).
    /// Returns thin factors: U is M×K, V^T is K×N where K = min(M,N).
    pub fn new(a: &DynMatrix<T>) -> Result<Self, LinalgError> {
        let m = a.nrows();
        let n = a.ncols();
        let k = m.min(n);

        if m == 0 || n == 0 {
            return Ok(Self {
                u: DynMatrix::zeros(m, k, T::zero()),
                singular_values: Vec::new(),
                vt: DynMatrix::zeros(k, n, T::zero()),
            });
        }

        // For wide matrices, transpose and swap U↔V at the end
        let transposed = m < n;
        let work_matrix = if transposed { a.transpose() } else { a.clone() };
        let (rows, cols) = (work_matrix.nrows(), work_matrix.ncols());

        let mut work = work_matrix;
        let mut u_mat = DynMatrix::zeros(rows, rows, T::zero());
        let mut v_mat = DynMatrix::zeros(cols, cols, T::zero());
        let mut diag = vec![<T::Real as num_traits::Zero>::zero(); cols];
        let mut off_diag = vec![<T::Real as num_traits::Zero>::zero(); cols];

        bidiagonalize(&mut work, &mut diag, &mut off_diag, &mut u_mat, &mut v_mat, true, true);
        bidiagonal_qr::<T>(
            &mut diag,
            &mut off_diag[..cols.saturating_sub(1)],
            &mut u_mat,
            &mut v_mat,
            true,
            true,
            30 * rows.max(cols),
        )?;

        // Extract thin factors.
        // Full SVD of work matrix (rows×cols, rows≥cols):
        //   U_full (rows×rows), σ (cols), V_full (cols×cols)
        // Thin: U_thin = first `cols` columns of U_full → rows×cols
        //        V^T_thin = first `cols` rows of V^H_full → cols×cols (same as full)
        //
        // For the original matrix:
        //   If not transposed: A's U_thin = M×K, A's V^T_thin = K×N
        //   If transposed: A = (A^T)^T, so A's U = V, A's V^T = U^T

        if transposed {
            // A^T = U_full · diag(σ) · V_full^T  →  A = V_full · diag(σ) · U_full^T
            // A's thin U (M×K): first K=M columns of V_full → cols×cols (cols=M, K=M → full V)
            // A's thin V^T (K×N): first K=M rows of U_full^T → M × rows, rows=N
            let mut u_thin = DynMatrix::zeros(m, k, T::zero());
            for i in 0..m {
                for j in 0..k {
                    u_thin[(i, j)] = v_mat[(i, j)];
                }
            }
            let mut vt_thin = DynMatrix::zeros(k, n, T::zero());
            for i in 0..k {
                for j in 0..n {
                    vt_thin[(i, j)] = u_mat[(j, i)].conj();
                }
            }
            Ok(Self {
                u: u_thin,
                singular_values: diag,
                vt: vt_thin,
            })
        } else {
            // U_thin = first K=N columns of U_full → M×N
            let mut u_thin = DynMatrix::zeros(m, k, T::zero());
            for i in 0..m {
                for j in 0..k {
                    u_thin[(i, j)] = u_mat[(i, j)];
                }
            }
            // V^T_thin = first K=N rows of V_full^H → N×N (same as full)
            let mut vt_thin = DynMatrix::zeros(k, n, T::zero());
            for i in 0..k {
                for j in 0..n {
                    vt_thin[(i, j)] = v_mat[(j, i)].conj();
                }
            }
            Ok(Self {
                u: u_thin,
                singular_values: diag,
                vt: vt_thin,
            })
        }
    }

    /// Compute only the singular values (faster, no U/V accumulation).
    pub fn singular_values_only(a: &DynMatrix<T>) -> Result<Vec<T::Real>, LinalgError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }

        let work_matrix = if m < n { a.transpose() } else { a.clone() };
        let (rows, cols) = (work_matrix.nrows(), work_matrix.ncols());

        let mut work = work_matrix;
        let mut u_mat = DynMatrix::zeros(rows, rows, T::zero());
        let mut v_mat = DynMatrix::zeros(cols, cols, T::zero());
        let mut diag = vec![<T::Real as num_traits::Zero>::zero(); cols];
        let mut off_diag = vec![<T::Real as num_traits::Zero>::zero(); cols];

        bidiagonalize(
            &mut work, &mut diag, &mut off_diag, &mut u_mat, &mut v_mat, false, false,
        );
        bidiagonal_qr::<T>(
            &mut diag,
            &mut off_diag[..cols.saturating_sub(1)],
            &mut u_mat,
            &mut v_mat,
            false,
            false,
            30 * rows.max(cols),
        )?;

        Ok(diag)
    }

    /// The singular values, sorted descending.
    #[inline]
    pub fn singular_values(&self) -> &[T::Real] {
        &self.singular_values
    }

    /// The left singular vectors U (M×K thin matrix, K = min(M,N)).
    /// Columns are the left singular vectors.
    #[inline]
    pub fn u(&self) -> &DynMatrix<T> {
        &self.u
    }

    /// The right singular vectors V^T (K×N thin matrix, K = min(M,N)).
    /// Rows are the right singular vectors.
    #[inline]
    pub fn vt(&self) -> &DynMatrix<T> {
        &self.vt
    }

    /// Numerical rank: number of singular values above `tol`.
    pub fn rank(&self, tol: T::Real) -> usize {
        self.singular_values.iter().filter(|&&s| s > tol).count()
    }

    /// Condition number: σ_max / σ_min.
    ///
    /// Returns infinity if the smallest singular value is zero.
    pub fn condition_number(&self) -> T::Real {
        if self.singular_values.is_empty() {
            return <T::Real as num_traits::One>::one();
        }
        let s_max = self.singular_values[0];
        let s_min = *self.singular_values.last().unwrap();
        if s_min == <T::Real as num_traits::Zero>::zero() {
            T::Real::infinity()
        } else {
            s_max / s_min
        }
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
    /// let a = DynMatrix::from_rows(2, 2, &[2.0_f64, 1.0, 5.0, 3.0]);
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
    /// let a = DynMatrix::from_rows(2, 2, &[4.0_f64, 7.0, 2.0, 6.0]);
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

    /// Singular value decomposition.
    ///
    /// Works for any M×N matrix (tall or wide).
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let a = DynMatrix::from_rows(3, 2, &[
    ///     1.0_f64, 0.0,
    ///     0.0, 1.0,
    ///     0.0, 0.0,
    /// ]);
    /// let svd = a.svd().unwrap();
    /// assert!((svd.singular_values()[0] - 1.0).abs() < 1e-10);
    /// ```
    pub fn svd(&self) -> Result<DynSvd<T>, LinalgError> {
        DynSvd::new(self)
    }

    /// Singular values only (no U/V computation).
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let a = DynMatrix::from_rows(2, 2, &[3.0_f64, 0.0, 0.0, 4.0]);
    /// let sv = a.singular_values_only().unwrap();
    /// assert!((sv[0] - 4.0).abs() < 1e-10);
    /// assert!((sv[1] - 3.0).abs() < 1e-10);
    /// ```
    pub fn singular_values_only(&self) -> Result<Vec<T::Real>, LinalgError> {
        DynSvd::singular_values_only(self)
    }

    /// Symmetric/Hermitian eigendecomposition.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let a = DynMatrix::from_rows(2, 2, &[5.0_f64, 2.0, 2.0, 2.0]);
    /// let eig = a.eig_symmetric().unwrap();
    /// assert!((eig.eigenvalues()[0] - 1.0).abs() < 1e-10);
    /// assert!((eig.eigenvalues()[1] - 6.0).abs() < 1e-10);
    /// ```
    pub fn eig_symmetric(&self) -> Result<DynSymmetricEigen<T>, LinalgError> {
        DynSymmetricEigen::new(self)
    }

    /// Eigenvalues of a symmetric/Hermitian matrix (no eigenvectors).
    pub fn eigenvalues_symmetric(&self) -> Result<Vec<T::Real>, LinalgError> {
        DynSymmetricEigen::eigenvalues_only(self)
    }
}

/// Convenience methods for Schur decomposition (real floats only).
impl<T: FloatScalar> DynMatrix<T> {
    /// Real Schur decomposition: `A = Q S Q^T`.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let a = DynMatrix::from_rows(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]);
    /// let schur = a.schur().unwrap();
    /// let (re, im) = schur.eigenvalues();
    /// let trace = a[(0, 0)] + a[(1, 1)];
    /// assert!((re[0] + re[1] - trace).abs() < 1e-10);
    /// ```
    pub fn schur(&self) -> Result<DynSchur<T>, LinalgError> {
        DynSchur::new(self)
    }

    /// General eigenvalues as (real_parts, imaginary_parts).
    pub fn eigenvalues(&self) -> Result<(Vec<T>, Vec<T>), LinalgError> {
        Ok(self.schur()?.eigenvalues())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Symmetric eigen tests ──

    #[test]
    fn symmetric_eigen_2x2() {
        let a = DynMatrix::from_rows(2, 2, &[2.0_f64, -1.0, -1.0, 2.0]);
        let eig = a.eig_symmetric().unwrap();
        assert!((eig.eigenvalues()[0] - 1.0).abs() < 1e-10);
        assert!((eig.eigenvalues()[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn symmetric_eigen_3x3_reconstruction() {
        let a = DynMatrix::from_rows(
            3, 3,
            &[4.0_f64, 1.0, -1.0, 1.0, 3.0, 2.0, -1.0, 2.0, 5.0],
        );
        let eig = a.eig_symmetric().unwrap();
        let q = eig.eigenvectors();
        let vals = eig.eigenvalues();

        // Q * diag(λ) * Q^T ≈ A
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += q[(i, k)] * vals[k] * q[(j, k)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-10,
                    "A[({},{})]",
                    i, j
                );
            }
        }

        // Q^T Q ≈ I
        let qt = q.transpose();
        let qtq = &qt * q;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[(i, j)] - expected).abs() < 1e-10,
                    "QtQ[({},{})]",
                    i, j
                );
            }
        }
    }

    #[test]
    fn symmetric_eigenvalues_only() {
        let a = DynMatrix::from_rows(2, 2, &[5.0_f64, 2.0, 2.0, 2.0]);
        let vals = a.eigenvalues_symmetric().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 6.0).abs() < 1e-10);
    }

    // ── Schur tests ──

    #[test]
    fn schur_2x2_complex_pair() {
        // Rotation: eigenvalues ±i
        let a = DynMatrix::from_rows(2, 2, &[0.0_f64, -1.0, 1.0, 0.0]);
        let schur = a.schur().unwrap();
        let (re, im) = schur.eigenvalues();
        assert!(re[0].abs() < 1e-10);
        assert!((im[0].abs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn schur_3x3_similarity() {
        let a = DynMatrix::from_rows(
            3, 3,
            &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0],
        );
        let schur = a.schur().unwrap();
        let s = schur.schur_form();
        let q = schur.schur_vectors();

        // Q^T A Q ≈ S
        let qt = q.transpose();
        let qtaq = &(&qt * &a) * q;
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (qtaq[(i, j)] - s[(i, j)]).abs() < 1e-10,
                    "Q^TAQ[({},{})]",
                    i, j
                );
            }
        }

        // trace preserved
        let (re, _) = schur.eigenvalues();
        let trace = a[(0, 0)] + a[(1, 1)] + a[(2, 2)];
        let eig_sum = re[0] + re[1] + re[2];
        assert!((eig_sum - trace).abs() < 1e-10, "trace");
    }

    #[test]
    fn schur_eigenvalues() {
        let a = DynMatrix::from_rows(2, 2, &[2.0_f64, -1.0, 1.0, 0.0]);
        let (re, im) = a.eigenvalues().unwrap();
        assert!((re[0] - 1.0).abs() < 1e-10);
        assert!((re[1] - 1.0).abs() < 1e-10);
        assert!(im[0].abs() < 1e-10);
    }

    // ── LU tests ──

    #[test]
    fn lu_solve_2x2() {
        let a = DynMatrix::from_rows(2, 2, &[3.0_f64, 2.0, 1.0, 4.0]);
        let b = DynVector::from_slice(&[7.0, 9.0]);
        let x = a.solve(&b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn lu_solve_3x3() {
        let a = DynMatrix::from_rows(
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
        let a = DynMatrix::from_rows(2, 2, &[4.0_f64, 7.0, 2.0, 6.0]);
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
        let a = DynMatrix::from_rows(2, 2, &[3.0_f64, 8.0, 4.0, 6.0]);
        let lu = a.lu().unwrap();
        assert!((lu.det() - (-14.0)).abs() < 1e-12);
    }

    #[test]
    fn lu_singular() {
        let a = DynMatrix::from_rows(2, 2, &[1.0_f64, 2.0, 2.0, 4.0]);
        assert_eq!(a.lu().unwrap_err(), LinalgError::Singular);
    }

    #[test]
    fn cholesky_solve() {
        let a = DynMatrix::from_rows(2, 2, &[4.0_f64, 2.0, 2.0, 3.0]);
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
        let a = DynMatrix::from_rows(2, 2, &[4.0_f64, 2.0, 2.0, 3.0]);
        let chol = a.cholesky().unwrap();
        let det_chol = chol.det();
        let det_lu = a.lu().unwrap().det();
        assert!((det_chol - det_lu).abs() < 1e-12);
    }

    #[test]
    fn cholesky_inverse() {
        let a = DynMatrix::from_rows(
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
        let a = DynMatrix::from_rows(2, 2, &[1.0_f64, 5.0, 5.0, 1.0]);
        assert_eq!(a.cholesky().unwrap_err(), LinalgError::NotPositiveDefinite);
    }

    #[test]
    fn qr_solve_square() {
        let a = DynMatrix::from_rows(
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
        let a = DynMatrix::from_rows(3, 2, &[1.0_f64, 0.0, 1.0, 1.0, 1.0, 2.0]);
        let b = DynVector::from_slice(&[1.0, 2.0, 4.0]);
        let qr = a.qr().unwrap();
        let x = qr.solve(&b);
        assert!((x[0] - 5.0 / 6.0).abs() < 1e-10);
        assert!((x[1] - 3.0 / 2.0).abs() < 1e-10);
    }

    #[test]
    fn qr_q_orthogonal() {
        let a = DynMatrix::from_rows(
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
        let a = DynMatrix::from_rows(
            3,
            3,
            &[6.0_f64, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0],
        );
        let det_qr = a.qr().unwrap().det();
        let det_lu = a.lu().unwrap().det();
        assert!((det_qr.abs() - det_lu.abs()).abs() < 1e-10);
    }

    // ── SVD tests ──

    #[test]
    fn svd_3x3_reconstruction() {
        let a = DynMatrix::from_rows(
            3, 3,
            &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0],
        );
        let svd = a.svd().unwrap();
        let u = svd.u();
        let vt = svd.vt();
        let sv = svd.singular_values();

        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += u[(i, k)] * sv[k] * vt[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-9,
                    "UΣV^T[({},{})] = {}, expected {}",
                    i, j, sum, a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn svd_tall_4x2() {
        let a = DynMatrix::from_rows(
            4, 2,
            &[1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        );
        let svd = a.svd().unwrap();
        let u = svd.u();
        let vt = svd.vt();
        let sv = svd.singular_values();

        assert_eq!(sv.len(), 2);
        assert_eq!(u.nrows(), 4);  // M
        assert_eq!(u.ncols(), 2);  // K = min(M,N)
        assert_eq!(vt.nrows(), 2); // K
        assert_eq!(vt.ncols(), 2); // N

        for i in 0..4 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += u[(i, k)] * sv[k] * vt[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-9,
                    "tall UΣV^T[({},{})]",
                    i, j
                );
            }
        }
    }

    #[test]
    fn svd_wide_2x4() {
        let a = DynMatrix::from_rows(
            2, 4,
            &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let svd = a.svd().unwrap();
        let u = svd.u();
        let vt = svd.vt();
        let sv = svd.singular_values();

        assert_eq!(sv.len(), 2);
        assert_eq!(u.nrows(), 2);  // M
        assert_eq!(u.ncols(), 2);  // K = min(M,N)
        assert_eq!(vt.nrows(), 2); // K
        assert_eq!(vt.ncols(), 4); // N

        for i in 0..2 {
            for j in 0..4 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += u[(i, k)] * sv[k] * vt[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-9,
                    "wide UΣV^T[({},{})]",
                    i, j
                );
            }
        }
    }

    #[test]
    fn svd_10x5() {
        // Larger matrix test
        let data: Vec<f64> = (0..50).map(|i| (i as f64 + 1.0) * 0.1).collect();
        // Make it full rank by adding identity-ish perturbation
        let mut a = DynMatrix::from_vec(10, 5, data);
        for i in 0..5 {
            a[(i, i)] = a[(i, i)] + 10.0;
        }

        let svd = a.svd().unwrap();
        let u = svd.u();
        let vt = svd.vt();
        let sv = svd.singular_values();

        assert_eq!(sv.len(), 5);

        // Check reconstruction
        for i in 0..10 {
            for j in 0..5 {
                let mut sum = 0.0;
                for k in 0..5 {
                    sum += u[(i, k)] * sv[k] * vt[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-8,
                    "10x5 UΣV^T[({},{})]",
                    i, j
                );
            }
        }

        // Singular values sorted descending
        for i in 0..4 {
            assert!(sv[i] >= sv[i + 1] - 1e-10);
        }
    }

    #[test]
    fn svd_singular_values_only() {
        let a = DynMatrix::from_rows(2, 2, &[3.0_f64, 0.0, 0.0, 4.0]);
        let sv = a.singular_values_only().unwrap();
        assert!((sv[0] - 4.0).abs() < 1e-10);
        assert!((sv[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn svd_rank_and_condition() {
        let a = DynMatrix::from_rows(3, 3, &[
            1.0_f64, 2.0, 3.0,
            2.0, 4.0, 6.0,
            3.0, 6.0, 9.0,
        ]);
        let svd = a.svd().unwrap();
        assert_eq!(svd.rank(1e-9), 1);
        assert!(svd.condition_number() > 1e10);
    }

    #[test]
    fn solve_verify_residual() {
        let a = DynMatrix::from_rows(
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
