use crate::linalg::hessenberg::hessenberg;
use crate::linalg::LinalgError;
use crate::traits::{FloatScalar, MatrixMut};
use crate::Matrix;
use num_traits::{Float, One, Zero};

/// Helper: get element with dereference for calling Float methods.
#[inline]
fn g<T: Copy>(m: &impl crate::traits::MatrixRef<T>, i: usize, j: usize) -> T {
    *m.get(i, j)
}

/// Francis double-shift QR iteration on an upper Hessenberg matrix.
///
/// Transforms `h` to real Schur form (quasi-upper-triangular) in place.
/// Accumulates the orthogonal similarity transform into `q`.
///
/// Real eigenvalues appear as 1×1 diagonal blocks; complex conjugate pairs
/// as 2×2 blocks `[[a, b], [c, d]]` with `c*b < 0`.
pub fn francis_qr<T: FloatScalar>(
    h: &mut impl MatrixMut<T>,
    q: &mut impl MatrixMut<T>,
    max_iter: usize,
) -> Result<(), LinalgError> {
    let n = h.nrows();
    if n <= 1 {
        return Ok(());
    }

    let eps = T::epsilon();
    let mut iter = 0usize;
    let mut p = n; // active submatrix is h[0..p, 0..p]

    while p > 2 {
        // Deflation: check if h[p-1, p-2] is negligible
        let tol = eps * (g(h, p - 2, p - 2).abs() + g(h, p - 1, p - 1).abs());
        if g(h, p - 1, p - 2).abs() <= tol {
            *h.get_mut(p - 1, p - 2) = T::zero();
            p -= 1;
            continue;
        }

        // Check if h[p-2, p-3] is negligible (2×2 block at bottom)
        if p >= 3 {
            let tol2 = eps * (g(h, p - 3, p - 3).abs() + g(h, p - 2, p - 2).abs());
            if g(h, p - 2, p - 3).abs() <= tol2 {
                *h.get_mut(p - 2, p - 3) = T::zero();
                p -= 2;
                continue;
            }
        }

        iter += 1;
        if iter > max_iter {
            return Err(LinalgError::ConvergenceFailure);
        }

        // Find the start of the active unreduced block
        let mut q_start = p - 1;
        while q_start > 0 {
            let tol_q = eps * (g(h, q_start - 1, q_start - 1).abs() + g(h, q_start, q_start).abs());
            if g(h, q_start, q_start - 1).abs() <= tol_q {
                *h.get_mut(q_start, q_start - 1) = T::zero();
                break;
            }
            q_start -= 1;
        }

        // Exceptional shift every 10 iterations
        let (s, t) = if iter % 10 == 0 {
            let w = g(h, p - 1, p - 2).abs() + g(h, p - 2, p - 3).abs();
            (w + w, w * w)
        } else {
            // Francis double shift from bottom-right 2×2 block
            let a11 = g(h, p - 2, p - 2);
            let a12 = g(h, p - 2, p - 1);
            let a21 = g(h, p - 1, p - 2);
            let a22 = g(h, p - 1, p - 1);
            (a11 + a22, a11 * a22 - a12 * a21)
        };

        // Implicit double shift: first column of (H^2 - s*H + t*I)
        let h00 = g(h, q_start, q_start);
        let h10 = g(h, q_start + 1, q_start);
        let h01 = g(h, q_start, q_start + 1);
        let h11 = g(h, q_start + 1, q_start + 1);

        let mut x = h00 * h00 + h01 * h10 - s * h00 + t;
        let mut y = h10 * (h00 + h11 - s);
        let mut z = if q_start + 2 < p {
            h10 * g(h, q_start + 2, q_start + 1)
        } else {
            T::zero()
        };

        // Chase the bulge down the diagonal
        for k in q_start..(p - 1) {
            let (v0, v1, v2, tau) = if k + 2 < p {
                householder3(x, y, z)
            } else {
                let (v0h, v1h, tau_h) = householder2(x, y);
                (v0h, v1h, T::zero(), tau_h)
            };
            let use3 = k + 2 < p;

            let col_start = if k > q_start { k - 1 } else { k };

            // Apply from the left
            for j in col_start..n {
                let mut dot = v0 * g(h, k, j) + v1 * g(h, k + 1, j);
                if use3 {
                    dot = dot + v2 * g(h, k + 2, j);
                }
                dot = tau * dot;
                *h.get_mut(k, j) = g(h, k, j) - dot * v0;
                *h.get_mut(k + 1, j) = g(h, k + 1, j) - dot * v1;
                if use3 {
                    *h.get_mut(k + 2, j) = g(h, k + 2, j) - dot * v2;
                }
            }

            // Apply from the right
            let row_end = if use3 { (k + 4).min(p) } else { p }.min(n);
            for i in 0..row_end {
                let mut dot = v0 * g(h, i, k) + v1 * g(h, i, k + 1);
                if use3 {
                    dot = dot + v2 * g(h, i, k + 2);
                }
                dot = tau * dot;
                *h.get_mut(i, k) = g(h, i, k) - dot * v0;
                *h.get_mut(i, k + 1) = g(h, i, k + 1) - dot * v1;
                if use3 {
                    *h.get_mut(i, k + 2) = g(h, i, k + 2) - dot * v2;
                }
            }

            // Accumulate into Q
            for i in 0..n {
                let mut dot = v0 * g(q, i, k) + v1 * g(q, i, k + 1);
                if use3 {
                    dot = dot + v2 * g(q, i, k + 2);
                }
                dot = tau * dot;
                *q.get_mut(i, k) = g(q, i, k) - dot * v0;
                *q.get_mut(i, k + 1) = g(q, i, k + 1) - dot * v1;
                if use3 {
                    *q.get_mut(i, k + 2) = g(q, i, k + 2) - dot * v2;
                }
            }

            // Prepare next bulge: pick up the fill-in entries from column k.
            if k + 2 < p - 1 {
                // Next step will be a 3-element reflector
                x = g(h, k + 1, k);
                y = g(h, k + 2, k);
                z = g(h, k + 3, k);
            } else if k + 1 < p - 1 {
                // Next step will be a 2-element reflector
                x = g(h, k + 1, k);
                y = g(h, k + 2, k);
                z = T::zero();
            }
        }

        // Clean up sub-sub-diagonal entries
        for i in 0..n {
            for j in 0..i.saturating_sub(1) {
                if g(h, i, j).abs() < eps * (g(h, i, i).abs() + g(h, j, j).abs()) {
                    *h.get_mut(i, j) = T::zero();
                }
            }
        }
    }

    // Handle remaining 2×2 block
    if p == 2 {
        let tol = eps * (g(h, 0, 0).abs() + g(h, 1, 1).abs());
        if g(h, 1, 0).abs() <= tol {
            *h.get_mut(1, 0) = T::zero();
        }
    }

    Ok(())
}

/// 3-element Householder: returns (v0, v1, v2, tau) with v0 = 1.
#[inline]
fn householder3<T: Float + Zero + One>(x: T, y: T, z: T) -> (T, T, T, T) {
    let norm = (x * x + y * y + z * z).sqrt();
    if norm <= T::epsilon() {
        return (T::one(), T::zero(), T::zero(), T::zero());
    }
    let sign = if x >= T::zero() { T::one() } else { T::zero() - T::one() };
    let u0 = x + sign * norm;
    let v1 = y / u0;
    let v2 = z / u0;
    let tau = (T::one() + T::one()) / (T::one() + v1 * v1 + v2 * v2);
    (T::one(), v1, v2, tau)
}

/// 2-element Householder: returns (v0, v1, tau) with v0 = 1.
#[inline]
fn householder2<T: Float + Zero + One>(x: T, y: T) -> (T, T, T) {
    let norm = (x * x + y * y).sqrt();
    if norm <= T::epsilon() {
        return (T::one(), T::zero(), T::zero());
    }
    let sign = if x >= T::zero() { T::one() } else { T::zero() - T::one() };
    let u0 = x + sign * norm;
    let v1 = y / u0;
    let tau = (T::one() + T::one()) / (T::one() + v1 * v1);
    (T::one(), v1, tau)
}

/// Real Schur decomposition of a fixed-size square matrix.
///
/// For a real matrix A, computes orthogonal Q and quasi-upper-triangular S
/// such that `A = Q S Q^T`. The diagonal of S consists of 1×1 blocks (real
/// eigenvalues) and 2×2 blocks (complex conjugate pairs).
///
/// # Example
///
/// ```
/// use numeris::Matrix;
/// use numeris::linalg::SchurDecomposition;
///
/// let a = Matrix::new([
///     [1.0_f64, 2.0],
///     [3.0, 4.0],
/// ]);
/// let schur = SchurDecomposition::new(&a).unwrap();
/// let (re, im) = schur.eigenvalues();
///
/// // For this matrix, eigenvalues are real: (5 ± √33) / 2
/// let expected = [
///     (5.0 - 33.0_f64.sqrt()) / 2.0,
///     (5.0 + 33.0_f64.sqrt()) / 2.0,
/// ];
/// assert!((re[0] - expected[0]).abs() < 1e-10 || (re[0] - expected[1]).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct SchurDecomposition<T: FloatScalar, const N: usize> {
    s: Matrix<T, N, N>,
    q: Matrix<T, N, N>,
}

impl<T: FloatScalar, const N: usize> SchurDecomposition<T, N> {
    /// Compute the real Schur decomposition of a square matrix.
    pub fn new(a: &Matrix<T, N, N>) -> Result<Self, LinalgError> {
        let mut s = *a;
        let mut q = Matrix::<T, N, N>::zeros();

        if N <= 1 {
            for i in 0..N {
                q[(i, i)] = T::one();
            }
            return Ok(Self { s, q });
        }

        hessenberg(&mut s, &mut q);
        francis_qr(&mut s, &mut q, 30 * N)?;

        Ok(Self { s, q })
    }

    /// The quasi-upper-triangular Schur form S.
    #[inline]
    pub fn schur_form(&self) -> &Matrix<T, N, N> {
        &self.s
    }

    /// The orthogonal Schur vectors Q.
    #[inline]
    pub fn schur_vectors(&self) -> &Matrix<T, N, N> {
        &self.q
    }

    /// Extract eigenvalues as (real_parts, imaginary_parts).
    ///
    /// 1×1 diagonal blocks give real eigenvalues (imaginary part = 0).
    /// 2×2 diagonal blocks give a conjugate pair.
    pub fn eigenvalues(&self) -> ([T; N], [T; N]) {
        let mut re = [T::zero(); N];
        let mut im = [T::zero(); N];
        let eps = T::epsilon();

        let mut i = 0;
        while i < N {
            if i + 1 < N && self.s[(i + 1, i)].abs() > eps {
                // 2×2 block
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

/// Convenience methods for Schur decomposition and general eigenvalues.
impl<T: FloatScalar, const N: usize> Matrix<T, N, N> {
    /// Real Schur decomposition: `A = Q S Q^T`.
    ///
    /// ```
    /// use numeris::Matrix;
    ///
    /// let a = Matrix::new([
    ///     [0.0_f64, -1.0],
    ///     [1.0, 0.0],
    /// ]);
    /// let schur = a.schur().unwrap();
    /// let (re, im) = schur.eigenvalues();
    /// // 90° rotation: eigenvalues ±i
    /// assert!(re[0].abs() < 1e-10);
    /// assert!((im[0].abs() - 1.0).abs() < 1e-10);
    /// ```
    pub fn schur(&self) -> Result<SchurDecomposition<T, N>, LinalgError> {
        SchurDecomposition::new(self)
    }

    /// General eigenvalues as (real_parts, imaginary_parts).
    ///
    /// Uses Schur decomposition internally. For symmetric matrices,
    /// prefer `eigenvalues_symmetric()` which is faster.
    ///
    /// ```
    /// use numeris::Matrix;
    ///
    /// let a = Matrix::new([
    ///     [2.0_f64, -1.0],
    ///     [1.0, 0.0],
    /// ]);
    /// let (re, im) = a.eigenvalues().unwrap();
    /// assert!((re[0] - 1.0).abs() < 1e-10);
    /// assert!((re[1] - 1.0).abs() < 1e-10);
    /// assert!(im[0].abs() < 1e-10);
    /// ```
    pub fn eigenvalues(&self) -> Result<([T; N], [T; N]), LinalgError> {
        Ok(self.schur()?.eigenvalues())
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

    fn verify_schur_3x3(a: &Matrix<f64, 3, 3>, schur: &SchurDecomposition<f64, 3>) {
        let s = schur.schur_form();
        let q = schur.schur_vectors();

        // Q^T A Q = S
        let qtaq = q.transpose() * *a * *q;
        for i in 0..3 {
            for j in 0..3 {
                assert_near(qtaq[(i, j)], s[(i, j)], TOL, &format!("Q^TAQ[({},{})]", i, j));
            }
        }

        // Q^T Q = I
        let qtq = q.transpose() * *q;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(qtq[(i, j)], expected, TOL, &format!("QtQ[({},{})]", i, j));
            }
        }

        // S is quasi-upper-triangular: below sub-diagonal is zero
        for i in 2usize..3 {
            for j in 0..i.saturating_sub(1) {
                assert_near(s[(i, j)], 0.0, TOL, &format!("S[({},{})] should be 0", i, j));
            }
        }
    }

    #[test]
    fn schur_all_real_eigenvalues() {
        let a = Matrix::new([
            [1.0_f64, 2.0, 3.0],
            [0.0, 4.0, 5.0],
            [0.0, 0.0, 6.0],
        ]);
        let schur = a.schur().unwrap();
        verify_schur_3x3(&a, &schur);

        let (re, im) = schur.eigenvalues();
        let mut sorted = [re[0], re[1], re[2]];
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_near(sorted[0], 1.0, TOL, "λ[0]");
        assert_near(sorted[1], 4.0, TOL, "λ[1]");
        assert_near(sorted[2], 6.0, TOL, "λ[2]");
        for i in 0..3 {
            assert_near(im[i], 0.0, TOL, &format!("im[{}]", i));
        }
    }

    #[test]
    fn schur_general_3x3() {
        let a = Matrix::new([
            [1.0_f64, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 0.0],
        ]);
        let schur = a.schur().unwrap();
        verify_schur_3x3(&a, &schur);

        let (re, _) = schur.eigenvalues();
        let trace = a[(0, 0)] + a[(1, 1)] + a[(2, 2)];
        let eig_sum = re[0] + re[1] + re[2];
        assert_near(eig_sum, trace, TOL, "trace");
    }

    #[test]
    fn schur_complex_conjugate_pair() {
        let theta = core::f64::consts::FRAC_PI_4;
        let c = theta.cos();
        let s = theta.sin();
        let a = Matrix::new([[c, -s], [s, c]]);
        let schur = a.schur().unwrap();

        let (re, im) = schur.eigenvalues();
        assert_near(re[0], c, TOL, "re[0]");
        assert_near(re[1], c, TOL, "re[1]");
        assert_near(im[0].abs(), s, TOL, "|im[0]|");
        assert_near(im[1].abs(), s, TOL, "|im[1]|");
        assert!(im[0] * im[1] < 0.0, "conjugate pair should have opposite signs");
    }

    #[test]
    fn schur_trace_det() {
        let a = Matrix::new([
            [2.0_f64, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [1.0, 0.0, 1.0],
        ]);
        let schur = a.schur().unwrap();
        verify_schur_3x3(&a, &schur);

        let (re, _) = schur.eigenvalues();
        let trace = a[(0, 0)] + a[(1, 1)] + a[(2, 2)];
        let eig_sum = re[0] + re[1] + re[2];
        assert_near(eig_sum, trace, TOL, "trace");

        let det_a = a.det();
        let s = schur.schur_form();
        let mut det_s = 1.0;
        let mut i = 0usize;
        while i < 3 {
            if i + 1 < 3 && s[(i + 1, i)].abs() > TOL {
                det_s *= s[(i, i)] * s[(i + 1, i + 1)] - s[(i, i + 1)] * s[(i + 1, i)];
                i += 2;
            } else {
                det_s *= s[(i, i)];
                i += 1;
            }
        }
        assert_near(det_s, det_a, TOL, "det");
    }

    #[test]
    fn schur_companion_matrix() {
        // p(x) = x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        let a = Matrix::new([
            [0.0_f64, 0.0, 6.0],
            [1.0, 0.0, -11.0],
            [0.0, 1.0, 6.0],
        ]);
        let schur = a.schur().unwrap();
        let (re, im) = schur.eigenvalues();

        let mut sorted = [re[0], re[1], re[2]];
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_near(sorted[0], 1.0, TOL, "root 1");
        assert_near(sorted[1], 2.0, TOL, "root 2");
        assert_near(sorted[2], 3.0, TOL, "root 3");
        for i in 0..3 {
            assert_near(im[i], 0.0, TOL, &format!("im[{}]", i));
        }
    }

    #[test]
    fn schur_identity() {
        let id: Matrix<f64, 3, 3> = Matrix::eye();
        let schur = id.schur().unwrap();
        let (re, im) = schur.eigenvalues();
        for i in 0..3 {
            assert_near(re[i], 1.0, TOL, &format!("re[{}]", i));
            assert_near(im[i], 0.0, TOL, &format!("im[{}]", i));
        }
    }

    #[test]
    fn eigenvalues_convenience() {
        let a = Matrix::new([[2.0_f64, -1.0], [1.0, 0.0]]);
        let (re, im) = a.eigenvalues().unwrap();
        assert_near(re[0], 1.0, TOL, "re[0]");
        assert_near(re[1], 1.0, TOL, "re[1]");
        assert_near(im[0], 0.0, TOL, "im[0]");
        assert_near(im[1], 0.0, TOL, "im[1]");
    }

    #[test]
    fn schur_4x4() {
        let a = Matrix::new([
            [4.0_f64, 1.0, -2.0, 2.0],
            [1.0, 2.0, 0.0, 1.0],
            [-2.0, 0.0, 3.0, -2.0],
            [2.0, 1.0, -2.0, 1.0],
        ]);
        let schur = a.schur().unwrap();
        let s = schur.schur_form();
        let q = schur.schur_vectors();

        let qtaq = q.transpose() * a * *q;
        for i in 0..4 {
            for j in 0..4 {
                assert_near(qtaq[(i, j)], s[(i, j)], TOL, &format!("Q^TAQ[({},{})]", i, j));
            }
        }

        let qtq = q.transpose() * *q;
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(qtq[(i, j)], expected, TOL, &format!("QtQ[({},{})]", i, j));
            }
        }

        let (re, _) = schur.eigenvalues();
        let trace = a[(0, 0)] + a[(1, 1)] + a[(2, 2)] + a[(3, 3)];
        let eig_sum = re[0] + re[1] + re[2] + re[3];
        assert_near(eig_sum, trace, TOL, "trace");
    }

    #[test]
    fn f32_support() {
        let a = Matrix::new([[1.0_f32, 2.0], [3.0, 4.0]]);
        let schur = a.schur().unwrap();
        let (re, im) = schur.eigenvalues();
        let trace = a[(0, 0)] + a[(1, 1)];
        let eig_sum = re[0] + re[1];
        assert!((eig_sum - trace).abs() < 1e-5);
        assert!(im[0].abs() < 1e-5);
        assert!(im[1].abs() < 1e-5);
    }

    #[test]
    fn schur_1x1() {
        let a = Matrix::new([[42.0_f64]]);
        let schur = a.schur().unwrap();
        let (re, im) = schur.eigenvalues();
        assert_near(re[0], 42.0, TOL, "re[0]");
        assert_near(im[0], 0.0, TOL, "im[0]");
    }

    #[test]
    fn schur_2x2_real() {
        let a = Matrix::new([[5.0_f64, 3.0], [0.0, 2.0]]);
        let schur = a.schur().unwrap();
        let (re, im) = schur.eigenvalues();
        let mut sorted = [re[0], re[1]];
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_near(sorted[0], 2.0, TOL, "λ[0]");
        assert_near(sorted[1], 5.0, TOL, "λ[1]");
        assert_near(im[0], 0.0, TOL, "im[0]");
        assert_near(im[1], 0.0, TOL, "im[1]");
    }
}
