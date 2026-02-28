use crate::linalg::LinalgError;
use crate::linalg::symmetric_eigen::givens;
use crate::traits::{LinalgScalar, MatrixMut};
use crate::Matrix;
use num_traits::{Float, One, Zero};

// ── Householder bidiagonalization ───────────────────────────────────

/// Householder bidiagonalization: reduce an M×N matrix (M ≥ N) to upper
/// bidiagonal form via orthogonal/unitary transforms.
///
/// On return:
/// - `diag[0..n]` contains the bidiagonal diagonal
/// - `off_diag[0..n-1]` contains the bidiagonal superdiagonal
/// - `u` (M×M) accumulates the left orthogonal/unitary transform
/// - `v` (N×N) accumulates the right orthogonal/unitary transform
///
/// The result satisfies `A = U · B · V^H` where B = bidiag(diag, off_diag).
pub(crate) fn bidiagonalize<T: LinalgScalar>(
    a: &mut impl MatrixMut<T>,
    diag: &mut [T::Real],
    off_diag: &mut [T::Real],
    u: &mut impl MatrixMut<T>,
    v: &mut impl MatrixMut<T>,
    compute_u: bool,
    compute_v: bool,
) {
    let m = a.nrows();
    let n = a.ncols();
    assert!(m >= n, "bidiagonalize requires M >= N");
    assert!(diag.len() >= n);
    assert!(off_diag.len() + 1 >= n);

    // Initialize U = I_m, V = I_n
    if compute_u {
        for i in 0..m {
            for j in 0..m {
                *u.get_mut(i, j) = if i == j { T::one() } else { T::zero() };
            }
        }
    }
    if compute_v {
        for i in 0..n {
            for j in 0..n {
                *v.get_mut(i, j) = if i == j { T::one() } else { T::zero() };
            }
        }
    }

    for k in 0..n {
        // ── Left Householder: zero out a[k+1:m, k] ──
        let mut norm_sq = <T::Real as Zero>::zero();
        for i in k..m {
            let val = *a.get(i, k);
            norm_sq = norm_sq + (val * val.conj()).re();
        }

        if norm_sq > T::lepsilon() * T::lepsilon() {
            let norm = norm_sq.lsqrt();
            let akk = *a.get(k, k);
            let alpha = akk.modulus();

            let sigma = if alpha < T::lepsilon() {
                T::from_real(norm)
            } else {
                T::from_real(norm) * (akk / T::from_real(alpha))
            };

            let v0 = akk + sigma;
            *a.get_mut(k, k) = v0;

            // Scale sub-diagonal entries
            for i in (k + 1)..m {
                let val = *a.get(i, k) / v0;
                *a.get_mut(i, k) = val;
            }

            let tau = v0 / sigma;

            // Apply to trailing columns: A[k:m, k+1:n] -= tau * v * (v^H * A)
            for j in (k + 1)..n {
                let mut dot = *a.get(k, j); // v[0] = 1 (implicit)
                for i in (k + 1)..m {
                    dot = dot + (*a.get(i, k)).conj() * *a.get(i, j);
                }
                dot = dot * tau;

                *a.get_mut(k, j) = *a.get(k, j) - dot;
                for i in (k + 1)..m {
                    let vi = *a.get(i, k);
                    *a.get_mut(i, j) = *a.get(i, j) - dot * vi;
                }
            }

            // Accumulate U: U = U * H_L = U * (I - tau * v * v^H)
            if compute_u {
                for row in 0..m {
                    let mut dot = *u.get(row, k);
                    for i in (k + 1)..m {
                        dot = dot + *u.get(row, i) * *a.get(i, k);
                    }
                    dot = dot * tau;

                    *u.get_mut(row, k) = *u.get(row, k) - dot;
                    for i in (k + 1)..m {
                        let vi_conj = (*a.get(i, k)).conj();
                        *u.get_mut(row, i) = *u.get(row, i) - dot * vi_conj;
                    }
                }
            }

            diag[k] = (T::zero() - sigma).re();
        } else {
            diag[k] = (*a.get(k, k)).re();
        }

        // ── Right Householder: zero out a[k, k+2:n] ──
        if k + 2 <= n.saturating_sub(1) {
            let mut norm_sq = <T::Real as Zero>::zero();
            for j in (k + 1)..n {
                let val = *a.get(k, j);
                norm_sq = norm_sq + (val * val.conj()).re();
            }

            if norm_sq > T::lepsilon() * T::lepsilon() {
                let norm = norm_sq.lsqrt();
                let ak_k1 = *a.get(k, k + 1);
                let alpha = ak_k1.modulus();

                let sigma = if alpha < T::lepsilon() {
                    T::from_real(norm)
                } else {
                    T::from_real(norm) * (ak_k1 / T::from_real(alpha))
                };

                let v0 = ak_k1 + sigma;
                *a.get_mut(k, k + 1) = v0;

                for j in (k + 2)..n {
                    let val = *a.get(k, j) / v0;
                    *a.get_mut(k, j) = val;
                }

                let tau = v0 / sigma;

                // Apply from the right to rows k+1..m
                for i in (k + 1)..m {
                    let mut dot = *a.get(i, k + 1);
                    for j in (k + 2)..n {
                        dot = dot + *a.get(i, j) * *a.get(k, j);
                    }
                    dot = dot * tau;

                    *a.get_mut(i, k + 1) = *a.get(i, k + 1) - dot;
                    for j in (k + 2)..n {
                        let vj_conj = (*a.get(k, j)).conj();
                        *a.get_mut(i, j) = *a.get(i, j) - dot * vj_conj;
                    }
                }

                // Accumulate V: V = V * H_R
                if compute_v {
                    for row in 0..n {
                        let mut dot = *v.get(row, k + 1);
                        for j in (k + 2)..n {
                            dot = dot + *v.get(row, j) * *a.get(k, j);
                        }
                        dot = dot * tau;

                        *v.get_mut(row, k + 1) = *v.get(row, k + 1) - dot;
                        for j in (k + 2)..n {
                            let vj_conj = (*a.get(k, j)).conj();
                            *v.get_mut(row, j) = *v.get(row, j) - dot * vj_conj;
                        }
                    }
                }

                off_diag[k] = (T::zero() - sigma).re();
            } else {
                off_diag[k] = (*a.get(k, k + 1)).re();
            }
        } else if k + 1 < n {
            off_diag[k] = (*a.get(k, k + 1)).re();
        }
    }
}

// ── Golub-Kahan bidiagonal QR ───────────────────────────────────────

/// Golub-Kahan implicit-shift QR iteration on a bidiagonal matrix.
///
/// On entry:
/// - `diag[0..n]`: bidiagonal diagonal entries
/// - `off_diag[0..n-1]`: bidiagonal superdiagonal entries
/// - `u`, `v`: orthogonal/unitary matrices to accumulate rotations into
/// - `compute_u`, `compute_v`: whether to actually accumulate
///
/// On return:
/// - `diag` contains non-negative singular values sorted descending
/// - `off_diag` is zeroed
pub(crate) fn bidiagonal_qr<T: LinalgScalar>(
    diag: &mut [T::Real],
    off_diag: &mut [T::Real],
    u: &mut impl MatrixMut<T>,
    v: &mut impl MatrixMut<T>,
    compute_u: bool,
    compute_v: bool,
    max_iter: usize,
) -> Result<(), LinalgError> {
    let n = diag.len();
    if n <= 1 {
        if n == 1 && diag[0] < <T::Real as Zero>::zero() {
            diag[0] = <T::Real as Zero>::zero() - diag[0];
            if compute_u {
                let m = u.nrows();
                for i in 0..m {
                    let val = *u.get(i, 0);
                    *u.get_mut(i, 0) = T::zero() - val;
                }
            }
        }
        return Ok(());
    }

    let eps = T::lepsilon();
    let mut iter = 0usize;
    let mut hi = n - 1;

    while hi > 0 {
        // Deflation: check if trailing off_diag is negligible
        {
            let threshold = eps * (diag[hi - 1].abs() + diag[hi].abs());
            if off_diag[hi - 1].abs() <= threshold {
                off_diag[hi - 1] = <T::Real as Zero>::zero();
                hi -= 1;
                continue;
            }
        }

        // Find lo: start of unreduced block
        let mut lo = hi - 1;
        while lo > 0 {
            let threshold = eps * (diag[lo - 1].abs() + diag[lo].abs());
            if off_diag[lo - 1].abs() <= threshold {
                off_diag[lo - 1] = <T::Real as Zero>::zero();
                break;
            }
            lo -= 1;
        }

        iter += 1;
        if iter > max_iter {
            return Err(LinalgError::ConvergenceFailure);
        }

        // Check for zero diagonal entries in the unreduced block.
        // When d[idx] ≈ 0, the Wilkinson shift formula can break down.
        // Chase the corresponding off-diagonal entry to zero using
        // left Givens rotations, which decouples the problem.
        {
            let zero = <T::Real as Zero>::zero();
            let mut found_zero = false;
            for idx in lo..hi {
                if diag[idx].abs() <= eps {
                    diag[idx] = zero;
                    // Chase off_diag[idx] off the bottom using left rotations.
                    // Each rotation mixes rows (j, idx) to zero the fill-in
                    // at position (idx, j), then creates new fill-in at (idx, j+1).
                    let mut z = off_diag[idx];
                    off_diag[idx] = zero;
                    for j in (idx + 1)..=hi {
                        let (c, s) = givens(diag[j], z);
                        diag[j] = c * diag[j] + s * z;
                        if j < hi {
                            z = zero - s * off_diag[j];
                            off_diag[j] = c * off_diag[j];
                        }
                        if compute_u {
                            let mu = u.nrows();
                            for row in 0..mu {
                                let uj = *u.get(row, j);
                                let ui = *u.get(row, idx);
                                *u.get_mut(row, j) =
                                    T::from_real(c) * uj + T::from_real(s) * ui;
                                *u.get_mut(row, idx) =
                                    T::from_real(c) * ui - T::from_real(s) * uj;
                            }
                        }
                    }
                    found_zero = true;
                    break;
                }
            }
            if found_zero {
                continue;
            }
        }

        // Wilkinson shift from trailing 2×2 of B^T B
        let d_hi = diag[hi];
        let d_hi1 = diag[hi - 1];
        let e_hi1 = off_diag[hi - 1];
        let e_hi2 = if hi >= 2 && hi - 2 >= lo {
            off_diag[hi - 2]
        } else {
            <T::Real as Zero>::zero()
        };

        let t11 = d_hi1 * d_hi1 + e_hi2 * e_hi2;
        let t12 = d_hi1 * e_hi1;
        let t22 = d_hi * d_hi + e_hi1 * e_hi1;

        let two = <T::Real as One>::one() + <T::Real as One>::one();
        let d = (t11 - t22) / two;
        let sign_d = if d >= <T::Real as Zero>::zero() {
            <T::Real as One>::one()
        } else {
            <T::Real as Zero>::zero() - <T::Real as One>::one()
        };
        let mu = t22 - t12 * t12 / (d + sign_d * (d * d + t12 * t12).sqrt());

        // Implicit QR chase
        let mut x = diag[lo] * diag[lo] - mu;
        let mut z = diag[lo] * off_diag[lo];

        for k in lo..hi {
            // Right Givens rotation: zero z
            let (c, s) = givens(x, z);

            if k > lo {
                off_diag[k - 1] = c * x + s * z;
            }

            // Right rotation on columns k, k+1 of B:
            // B[k, k] = c*dk + s*ek
            // B[k, k+1] = c*ek - s*dk
            // B[k+1, k] = s*dk1  (fill-in / bulge)
            // B[k+1, k+1] = c*dk1
            let dk = diag[k];
            let ek = off_diag[k];
            let dk1 = diag[k + 1];

            diag[k] = c * dk + s * ek;
            off_diag[k] = c * ek - s * dk;
            let bulge = s * dk1;
            diag[k + 1] = c * dk1;

            if compute_v {
                let nv = v.nrows();
                for row in 0..nv {
                    let vk = *v.get(row, k);
                    let vk1 = *v.get(row, k + 1);
                    *v.get_mut(row, k) = T::from_real(c) * vk + T::from_real(s) * vk1;
                    *v.get_mut(row, k + 1) = T::from_real(c) * vk1 - T::from_real(s) * vk;
                }
            }

            // Left Givens rotation: zero the bulge at B[k+1, k]
            let (c2, s2) = givens(diag[k], bulge);

            // Left rotation on rows k, k+1:
            // B[k, k] = c2*d[k] + s2*bulge
            // B[k, k+1] = c2*e[k] + s2*d[k+1]
            // B[k+1, k+1] = c2*d[k+1] - s2*e[k]
            // B[k, k+2] = s2*e[k+1]  (new fill-in, drives next chase step)
            // B[k+1, k+2] = c2*e[k+1]
            diag[k] = c2 * diag[k] + s2 * bulge;
            let old_ek = off_diag[k];
            let old_dk1 = diag[k + 1];
            off_diag[k] = c2 * old_ek + s2 * old_dk1;
            diag[k + 1] = c2 * old_dk1 - s2 * old_ek;

            if k + 1 < hi {
                let old_ek1 = off_diag[k + 1];
                // The fill-in at B[k, k+2] drives the next right rotation
                x = off_diag[k]; // B[k, k+1] — the entry to keep
                z = s2 * old_ek1; // B[k, k+2] — the fill-in to eliminate
                off_diag[k + 1] = c2 * old_ek1;
            }

            if compute_u {
                let mu = u.nrows();
                for row in 0..mu {
                    let uk = *u.get(row, k);
                    let uk1 = *u.get(row, k + 1);
                    *u.get_mut(row, k) = T::from_real(c2) * uk + T::from_real(s2) * uk1;
                    *u.get_mut(row, k + 1) = T::from_real(c2) * uk1 - T::from_real(s2) * uk;
                }
            }
        }
    }

    // Make all singular values non-negative
    for i in 0..n {
        if diag[i] < <T::Real as Zero>::zero() {
            diag[i] = <T::Real as Zero>::zero() - diag[i];
            if compute_u {
                let m = u.nrows();
                for row in 0..m {
                    let val = *u.get(row, i);
                    *u.get_mut(row, i) = T::zero() - val;
                }
            }
        }
    }

    // Sort singular values descending, permute U and V columns
    for i in 0..n {
        let mut max_idx = i;
        for j in (i + 1)..n {
            if diag[j] > diag[max_idx] {
                max_idx = j;
            }
        }
        if max_idx != i {
            diag.swap(i, max_idx);

            if compute_u {
                let m = u.nrows();
                for row in 0..m {
                    let tmp = *u.get(row, i);
                    *u.get_mut(row, i) = *u.get(row, max_idx);
                    *u.get_mut(row, max_idx) = tmp;
                }
            }

            if compute_v {
                let nv = v.nrows();
                for row in 0..nv {
                    let tmp = *v.get(row, i);
                    *v.get_mut(row, i) = *v.get(row, max_idx);
                    *v.get_mut(row, max_idx) = tmp;
                }
            }
        }
    }

    Ok(())
}

// ── SvdDecomposition wrapper ────────────────────────────────────────

/// Singular value decomposition of a fixed-size matrix (M ≥ N).
///
/// Computes orthogonal U (M×M), singular values σ (length N, sorted
/// descending), and orthogonal V^T (N×N) such that `A = U · diag(σ) · V^T`.
///
/// # Example
///
/// ```
/// use numeris::Matrix;
/// use numeris::linalg::SvdDecomposition;
///
/// let a = Matrix::new([
///     [3.0_f64, 2.0, 2.0],
///     [2.0, 3.0, -2.0],
/// ]);
/// // For M < N, use .transpose().svd() and swap U↔V
/// let at = a.transpose();
/// let svd = at.svd().unwrap();
/// let sigma = svd.singular_values();
/// assert!(sigma[0] >= sigma[1]); // sorted descending
/// ```
#[derive(Debug, Clone)]
pub struct SvdDecomposition<T: LinalgScalar, const M: usize, const N: usize> {
    u: Matrix<T, M, M>,
    singular_values: [T::Real; N],
    vt: Matrix<T, N, N>,
}

impl<T: LinalgScalar, const M: usize, const N: usize> SvdDecomposition<T, M, N> {
    /// Compute the SVD of a matrix.
    ///
    /// Requires `M >= N`. For wide matrices (M < N), transpose first.
    ///
    /// Returns `Err(ConvergenceFailure)` if the iterative bidiagonal QR
    /// does not converge within the iteration budget.
    ///
    /// ```
    /// use numeris::Matrix;
    /// use numeris::linalg::SvdDecomposition;
    ///
    /// let a = Matrix::new([
    ///     [1.0_f64, 0.0],
    ///     [0.0, 2.0],
    ///     [0.0, 0.0],
    /// ]);
    /// let svd = SvdDecomposition::new(&a).unwrap();
    /// assert!((svd.singular_values()[0] - 2.0).abs() < 1e-10);
    /// assert!((svd.singular_values()[1] - 1.0).abs() < 1e-10);
    /// ```
    pub fn new(a: &Matrix<T, M, N>) -> Result<Self, LinalgError> {
        assert!(M >= N, "SVD requires M >= N; transpose first for wide matrices");

        if N == 0 {
            return Ok(Self {
                u: Matrix::<T, M, M>::eye(),
                singular_values: [<T::Real as Zero>::zero(); N],
                vt: Matrix::<T, N, N>::zeros(),
            });
        }

        let mut work = *a;
        let mut u = Matrix::<T, M, M>::zeros();
        let mut v = Matrix::<T, N, N>::zeros();
        let mut diag = [<T::Real as Zero>::zero(); N];
        let mut off_diag = [<T::Real as Zero>::zero(); N]; // only first N-1 used

        bidiagonalize(&mut work, &mut diag, &mut off_diag, &mut u, &mut v, true, true);
        bidiagonal_qr::<T>(
            &mut diag,
            &mut off_diag[..N.saturating_sub(1)],
            &mut u,
            &mut v,
            true,
            true,
            30 * M.max(N),
        )?;

        // V^T = V^H (conjugate transpose)
        let mut vt = Matrix::<T, N, N>::zeros();
        for i in 0..N {
            for j in 0..N {
                vt[(i, j)] = v[(j, i)].conj();
            }
        }

        Ok(Self {
            u,
            singular_values: diag,
            vt,
        })
    }

    /// Compute only the singular values (faster, no U/V accumulation).
    pub fn singular_values_only(a: &Matrix<T, M, N>) -> Result<[T::Real; N], LinalgError> {
        assert!(M >= N, "SVD requires M >= N; transpose first for wide matrices");

        if N == 0 {
            return Ok([<T::Real as Zero>::zero(); N]);
        }

        let mut work = *a;
        let mut u = Matrix::<T, M, M>::zeros();
        let mut v = Matrix::<T, N, N>::zeros();
        let mut diag = [<T::Real as Zero>::zero(); N];
        let mut off_diag = [<T::Real as Zero>::zero(); N];

        bidiagonalize(&mut work, &mut diag, &mut off_diag, &mut u, &mut v, false, false);
        bidiagonal_qr::<T>(
            &mut diag,
            &mut off_diag[..N.saturating_sub(1)],
            &mut u,
            &mut v,
            false,
            false,
            30 * M.max(N),
        )?;

        Ok(diag)
    }

    /// The singular values, sorted descending.
    #[inline]
    pub fn singular_values(&self) -> &[T::Real; N] {
        &self.singular_values
    }

    /// The left singular vectors U (M×M orthogonal/unitary matrix).
    #[inline]
    pub fn u(&self) -> &Matrix<T, M, M> {
        &self.u
    }

    /// The right singular vectors V^T (N×N orthogonal/unitary matrix).
    /// Rows of V^T are the right singular vectors.
    #[inline]
    pub fn vt(&self) -> &Matrix<T, N, N> {
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
        if N == 0 {
            return <T::Real as One>::one();
        }
        let s_max = self.singular_values[0];
        let s_min = self.singular_values[N - 1];
        if s_min == <T::Real as Zero>::zero() {
            T::Real::infinity()
        } else {
            s_max / s_min
        }
    }
}

/// Convenience methods for SVD on rectangular matrices (M ≥ N).
impl<T: LinalgScalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Singular value decomposition.
    ///
    /// Returns `U` (M×M), singular values (length N, descending), and `V^T` (N×N).
    /// Requires `M >= N`.
    ///
    /// ```
    /// use numeris::Matrix;
    ///
    /// let a = Matrix::new([
    ///     [1.0_f64, 0.0],
    ///     [0.0, 1.0],
    ///     [0.0, 0.0],
    /// ]);
    /// let svd = a.svd().unwrap();
    /// assert!((svd.singular_values()[0] - 1.0).abs() < 1e-10);
    /// assert!((svd.singular_values()[1] - 1.0).abs() < 1e-10);
    /// ```
    pub fn svd(&self) -> Result<SvdDecomposition<T, M, N>, LinalgError> {
        SvdDecomposition::new(self)
    }

    /// Singular values only (no U/V computation).
    ///
    /// Requires `M >= N`.
    ///
    /// ```
    /// use numeris::Matrix;
    ///
    /// let a = Matrix::new([[3.0_f64, 0.0], [0.0, 4.0]]);
    /// let sv = a.singular_values_only().unwrap();
    /// assert!((sv[0] - 4.0).abs() < 1e-10);
    /// assert!((sv[1] - 3.0).abs() < 1e-10);
    /// ```
    pub fn singular_values_only(&self) -> Result<[T::Real; N], LinalgError> {
        SvdDecomposition::singular_values_only(self)
    }
}

// ── Tests ───────────────────────────────────────────────────────────

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
    fn identity_2x2() {
        let a: Matrix<f64, 2, 2> = Matrix::eye();
        let svd = a.svd().unwrap();

        for i in 0..2 {
            assert_near(svd.singular_values()[i], 1.0, TOL, &format!("σ[{}]", i));
        }

        let u = svd.u();
        let utu = u.transpose() * *u;
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(utu[(i, j)], expected, TOL, &format!("U^TU[({},{})]", i, j));
            }
        }

        let vt = svd.vt();
        let vtv = *vt * vt.transpose();
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(vtv[(i, j)], expected, TOL, &format!("V^TV[({},{})]", i, j));
            }
        }
    }

    #[test]
    fn identity_3x3() {
        let a: Matrix<f64, 3, 3> = Matrix::eye();
        let svd = a.svd().unwrap();
        for i in 0..3 {
            assert_near(svd.singular_values()[i], 1.0, TOL, &format!("σ[{}]", i));
        }
    }

    #[test]
    fn diagonal_matrix() {
        let a = Matrix::new([
            [5.0_f64, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let svd = a.svd().unwrap();
        assert_near(svd.singular_values()[0], 5.0, TOL, "σ[0]");
        assert_near(svd.singular_values()[1], 3.0, TOL, "σ[1]");
        assert_near(svd.singular_values()[2], 1.0, TOL, "σ[2]");
    }

    #[test]
    fn diagonal_with_negative() {
        let a = Matrix::new([
            [-3.0_f64, 0.0],
            [0.0, 2.0],
        ]);
        let svd = a.svd().unwrap();
        assert_near(svd.singular_values()[0], 3.0, TOL, "σ[0]");
        assert_near(svd.singular_values()[1], 2.0, TOL, "σ[1]");
    }

    #[test]
    fn known_2x2() {
        let a = Matrix::new([
            [3.0_f64, 2.0],
            [2.0, 3.0],
        ]);
        let svd = a.svd().unwrap();
        // A^T A = [[13, 12], [12, 13]], eigenvalues 25 and 1
        assert_near(svd.singular_values()[0], 5.0, TOL, "σ[0]");
        assert_near(svd.singular_values()[1], 1.0, TOL, "σ[1]");
    }

    #[test]
    fn reconstruction_3x3() {
        let a = Matrix::new([
            [1.0_f64, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 0.0],
        ]);
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
                assert_near(sum, a[(i, j)], 1e-9, &format!("UΣV^T[({},{})]", i, j));
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
        let svd = a.svd().unwrap();

        let u = svd.u();
        let utu = u.transpose() * *u;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(utu[(i, j)], expected, 1e-9, &format!("U^TU[({},{})]", i, j));
            }
        }

        let vt = svd.vt();
        let vtv = *vt * vt.transpose();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(vtv[(i, j)], expected, 1e-9, &format!("VV^T[({},{})]", i, j));
            }
        }
    }

    #[test]
    fn sorted_descending() {
        let a = Matrix::new([
            [10.0_f64, 3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 7.0, 2.0],
            [0.0, 0.0, 2.0, 4.0],
        ]);
        let svd = a.svd().unwrap();
        let sv = svd.singular_values();
        for i in 0..3 {
            assert!(
                sv[i] >= sv[i + 1] - TOL,
                "not descending: σ[{}]={} < σ[{}]={}",
                i, sv[i], i + 1, sv[i + 1]
            );
        }
    }

    #[test]
    fn rank_deficient() {
        let a = Matrix::new([
            [1.0_f64, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
        ]);
        let svd = a.svd().unwrap();
        let sv = svd.singular_values();

        assert!(sv[0] > 1.0, "σ[0] should be large");
        assert!(sv[1].abs() < 1e-9, "σ[1] should be ≈ 0");
        assert!(sv[2].abs() < 1e-9, "σ[2] should be ≈ 0");
        assert_eq!(svd.rank(1e-9), 1);
    }

    #[test]
    fn rectangular_tall() {
        let a = Matrix::new([
            [1.0_f64, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]);
        let svd = a.svd().unwrap();
        let u = svd.u();
        let vt = svd.vt();
        let sv = svd.singular_values();

        for i in 0..4 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += u[(i, k)] * sv[k] * vt[(k, j)];
                }
                assert_near(sum, a[(i, j)], 1e-9, &format!("UΣV^T[({},{})]", i, j));
            }
        }
    }

    #[test]
    fn singular_values_only_path() {
        let a = Matrix::new([
            [3.0_f64, 0.0],
            [0.0, 4.0],
        ]);
        let sv = a.singular_values_only().unwrap();
        assert_near(sv[0], 4.0, TOL, "σ[0]");
        assert_near(sv[1], 3.0, TOL, "σ[1]");
    }

    #[test]
    fn rank_and_condition() {
        let a = Matrix::new([
            [2.0_f64, 0.0],
            [0.0, 0.5],
        ]);
        let svd = a.svd().unwrap();
        assert_eq!(svd.rank(1e-10), 2);
        assert_near(svd.condition_number(), 4.0, TOL, "cond");
    }

    #[test]
    fn f32_support() {
        let a = Matrix::new([
            [3.0_f32, 1.0],
            [1.0, 3.0],
        ]);
        let svd = a.svd().unwrap();
        assert!((svd.singular_values()[0] - 4.0).abs() < 1e-5);
        assert!((svd.singular_values()[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn size_1x1() {
        let a = Matrix::new([[7.0_f64]]);
        let svd = a.svd().unwrap();
        assert_near(svd.singular_values()[0], 7.0, TOL, "σ[0]");
    }

    #[test]
    fn size_1x1_negative() {
        let a = Matrix::new([[-5.0_f64]]);
        let svd = a.svd().unwrap();
        assert_near(svd.singular_values()[0], 5.0, TOL, "σ[0]");
    }

    #[test]
    fn reconstruction_5x3() {
        let a = Matrix::new([
            [1.0_f64, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 0.0],
            [10.0, 11.0, 1.0],
            [13.0, 14.0, 2.0],
        ]);
        let svd = a.svd().unwrap();
        let u = svd.u();
        let vt = svd.vt();
        let sv = svd.singular_values();

        for i in 0..5 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += u[(i, k)] * sv[k] * vt[(k, j)];
                }
                assert_near(sum, a[(i, j)], 1e-8, &format!("UΣV^T[({},{})]", i, j));
            }
        }
    }

    // Complex SVD is deferred — requires phase absorption in bidiagonalization
    // to produce a real bidiagonal form from complex Householder reflections.
}
