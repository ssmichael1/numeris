use crate::linalg::split_two_col_slices;
use crate::traits::{LinalgScalar, MatrixMut};
use num_traits::Zero;

/// Reduce a square matrix to upper Hessenberg form via Householder similarity
/// transforms: `Q^H A Q = H`.
///
/// On return:
/// - `a` is overwritten with the upper Hessenberg matrix H
/// - `q` accumulates the orthogonal/unitary transform Q
///
/// The result satisfies `A = Q H Q^H` (or `Q^T H Q^T` for real matrices).
pub fn hessenberg<T: LinalgScalar>(
    a: &mut impl MatrixMut<T>,
    q: &mut impl MatrixMut<T>,
) {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "hessenberg requires a square matrix");
    assert_eq!(q.nrows(), n);
    assert_eq!(q.ncols(), n);

    // Initialize Q = I
    for i in 0..n {
        for j in 0..n {
            *q.get_mut(i, j) = if i == j { T::one() } else { T::zero() };
        }
    }

    for k in 0..n.saturating_sub(2) {
        // Form Householder vector from a[k+1:n, k] (contiguous column data)
        let sub_col = a.col_as_slice(k, k + 1);
        let mut norm_sq = T::Real::zero();
        for &v in sub_col {
            norm_sq = norm_sq + (v * v.conj()).re();
        }

        if norm_sq <= T::lepsilon() * T::lepsilon() {
            continue;
        }

        let norm = norm_sq.lsqrt();
        let ak1k = *a.get(k + 1, k);
        let alpha = ak1k.modulus();

        let sigma = if alpha < T::lepsilon() {
            T::from_real(norm)
        } else {
            T::from_real(norm) * (ak1k / T::from_real(alpha))
        };

        let v0 = ak1k + sigma;

        // Store normalized Householder vector in a[k+2:n, k] (v[0] = 1 implicit)
        {
            let sub_col = a.col_as_mut_slice(k, k + 2);
            for x in sub_col.iter_mut() {
                *x = *x / v0;
            }
        }

        // tau = v0 / sigma for the real case, or more generally:
        let tau = v0 / sigma;

        // Apply from the left: A[k+1:n, k+1:n] = (I - tau v v^H) A[k+1:n, k+1:n]
        // v = [1, a[k+2,k], ..., a[n-1,k]]
        // Column-major: v sub-column and A[:,j] sub-column are contiguous.
        // Note: column k is NOT included because the reflector was computed from it;
        // the result for column k is set explicitly below (a[k+1,k] = -sigma).
        for j in (k + 1)..n {
            let mut dot = *a.get(k + 1, j); // v[0]=1, conj(1)=1
            let (v_slice, a_j_slice) = split_two_col_slices(a, k, j, k + 2);
            for idx in 0..v_slice.len() {
                dot = dot + v_slice[idx].conj() * a_j_slice[idx];
            }
            dot = dot * tau;

            *a.get_mut(k + 1, j) = *a.get(k + 1, j) - dot;
            let (v_slice, a_j_slice) = split_two_col_slices(a, k, j, k + 2);
            crate::simd::axpy_neg_dispatch(a_j_slice, dot, v_slice);
        }

        // Apply from the right: A[0:n, k+1:n] = A[0:n, k+1:n] (I - tau v v^H)^H
        // For real: (I - tau v v^T)^T = (I - tau v v^T) since tau is real.
        // For complex: (I - tau v v^H)^H = I - conj(tau) v^* v^T ... but similarity
        // transforms use the same reflector on both sides.
        // Actually: right-multiply by H = I - tau v v^H means:
        // A[:, j] -= tau * v[j-k-1] * (A * conj(v)) ... let's do it column-by-column.
        // For each row: row -= conj(tau) * (row · v) * v^H
        // Wait: A * H = A * (I - tau v v^H) = A - tau * (A v) v^H
        // So we compute w = A[0:n, k+1:n] * v, then A[0:n, k+1:n] -= tau * w * v^H.

        for i in 0..n {
            let mut dot = *a.get(i, k + 1); // v[0] = 1
            for jj in (k + 2)..n {
                dot = dot + *a.get(i, jj) * *a.get(jj, k); // v stored in a[jj, k]
            }
            dot = dot * tau;

            *a.get_mut(i, k + 1) = *a.get(i, k + 1) - dot; // conj(v[0]) = conj(1) = 1
            for jj in (k + 2)..n {
                let vj_conj = (*a.get(jj, k)).conj();
                *a.get_mut(i, jj) = *a.get(i, jj) - dot * vj_conj;
            }
        }

        // Accumulate Q: Q = Q * H = Q * (I - tau v v^H)
        // Same right-multiply pattern as above.
        for i in 0..n {
            let mut dot = *q.get(i, k + 1);
            for jj in (k + 2)..n {
                dot = dot + *q.get(i, jj) * *a.get(jj, k);
            }
            dot = dot * tau;

            *q.get_mut(i, k + 1) = *q.get(i, k + 1) - dot;
            for jj in (k + 2)..n {
                let vj_conj = (*a.get(jj, k)).conj();
                *q.get_mut(i, jj) = *q.get(i, jj) - dot * vj_conj;
            }
        }

        // Zero out the sub-sub-diagonal entries and set a[k+1,k] = -sigma
        *a.get_mut(k + 1, k) = T::zero() - sigma;
        for i in (k + 2)..n {
            *a.get_mut(i, k) = T::zero();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;

    const TOL: f64 = 1e-10;

    #[test]
    fn hessenberg_3x3() {
        let orig = Matrix::new([
            [4.0_f64, 1.0, -2.0],
            [1.0, 2.0, 0.0],
            [-2.0, 0.0, 3.0],
        ]);
        let mut a = orig;
        let mut q = Matrix::<f64, 3, 3>::zeros();
        hessenberg(&mut a, &mut q);

        // Verify H is upper Hessenberg (below sub-diagonal is zero)
        for i in 2..3 {
            for j in 0..i - 1 {
                assert!(
                    a[(i, j)].abs() < TOL,
                    "H[({},{})] = {} should be zero",
                    i,
                    j,
                    a[(i, j)]
                );
            }
        }

        // Verify Q^T * orig * Q = H
        let qt = q.transpose();
        let qtaq = qt * orig * q;
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (qtaq[(i, j)] - a[(i, j)]).abs() < TOL,
                    "Q^TAQ[({},{})] = {}, H = {}",
                    i,
                    j,
                    qtaq[(i, j)],
                    a[(i, j)]
                );
            }
        }

        // Verify Q is orthogonal
        let qtq = qt * q;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[(i, j)] - expected).abs() < TOL,
                    "QtQ[({},{})] = {}, expected {}",
                    i,
                    j,
                    qtq[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn hessenberg_4x4() {
        let orig = Matrix::new([
            [1.0_f64, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        let mut a = orig;
        let mut q = Matrix::<f64, 4, 4>::zeros();
        hessenberg(&mut a, &mut q);

        // Verify upper Hessenberg structure
        for i in 0usize..4 {
            for j in 0..i.saturating_sub(1) {
                assert!(
                    a[(i, j)].abs() < TOL,
                    "H[({},{})] = {} should be zero",
                    i,
                    j,
                    a[(i, j)]
                );
            }
        }

        // Verify Q^T A Q = H
        let qtaq = q.transpose() * orig * q;
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (qtaq[(i, j)] - a[(i, j)]).abs() < TOL,
                    "Q^TAQ[({},{})] diff = {}",
                    i,
                    j,
                    (qtaq[(i, j)] - a[(i, j)]).abs()
                );
            }
        }
    }

    #[test]
    fn hessenberg_already_hessenberg() {
        // Upper triangular is already Hessenberg
        let orig = Matrix::new([
            [1.0_f64, 2.0, 3.0],
            [0.0, 4.0, 5.0],
            [0.0, 0.0, 6.0],
        ]);
        let mut a = orig;
        let mut q = Matrix::<f64, 3, 3>::zeros();
        hessenberg(&mut a, &mut q);

        // H should be unchanged (or similar, up to sign)
        let qtaq = q.transpose() * orig * q;
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (qtaq[(i, j)] - a[(i, j)]).abs() < TOL,
                    "Q^TAQ[({},{})]",
                    i,
                    j
                );
            }
        }
    }
}
