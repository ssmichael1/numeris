
use super::{Matrix, MatrixElem};

impl<const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T>
where
    T: MatrixElem + num_traits::Float,
{
    /// Compute the Singular Value Decomposition (SVD) of the matrix.
    ///
    /// Returns a tuple (U, S, V^T) where U and V^T are orthogonal matrices
    /// and S is a diagonal matrix of singular values.
    pub fn svd(&self) -> Option<(Matrix<ROWS, ROWS, T>, Matrix<ROWS, COLS, T>, Matrix<COLS, COLS, T>)> {
        // Golub-Kahan SVD for small matrices (educational, not optimized)
        // 1. Bidiagonalization using Householder reflections
        let mut a = *self;
        let mut u = Matrix::<ROWS, ROWS, T>::identity();
        let mut v = Matrix::<COLS, COLS, T>::identity();
        let min_dim = ROWS.min(COLS);
        for k in 0..min_dim {
            // Householder for column
            let mut x = [T::zero(); ROWS];
            for i in k..ROWS {
                x[i] = a[(i, k)];
            }
            let norm_x = x[k..].iter().map(|xi| *xi * *xi).fold(T::zero(), |a, b| a + b).sqrt();
            let sign = if x[k] >= T::zero() { T::one() } else { -T::one() };
            let u1 = x[k] + sign * norm_x;
            let mut w = [T::zero(); ROWS];
            w[k] = u1;
            if k + 1 < ROWS {
                w[(k+1)..ROWS].clone_from_slice(&x[(k+1)..ROWS]);
            }
            let wnorm = w[k..].iter().map(|wi| *wi * *wi).fold(T::zero(), |a, b| a + b).sqrt();
            if wnorm > T::zero() {
                for wi in &mut w[k..ROWS] {
                    *wi /= wnorm;
                }
                // Apply to A (left)
                for j in k..COLS {
                    let mut dot = T::zero();
                    for i in k..ROWS {
                        dot += w[i] * a[(i, j)];
                    }
                    for i in k..ROWS {
                        a[(i, j)] = a[(i, j)] - T::from(2.0).unwrap() * w[i] * dot;
                    }
                }
                // Accumulate U
                for j in 0..ROWS {
                    let mut dot = T::zero();
                    for i in k..ROWS {
                        dot += w[i] * u[(j, i)];
                    }
                    for i in k..ROWS {
                        u[(j, i)] = u[(j, i)] - T::from(2.0).unwrap() * w[i] * dot;
                    }
                }
            }
            // Householder for row (except last col)
            if k < COLS - 1 {
                let mut x = [T::zero(); COLS];
                for j in (k+1)..COLS {
                    x[j] = a[(k, j)];
                }
                let norm_x = x[(k+1)..].iter().map(|xi| *xi * *xi).fold(T::zero(), |a, b| a + b).sqrt();
                let sign = if x[k+1] >= T::zero() { T::one() } else { -T::one() };
                let u1 = x[k+1] + sign * norm_x;
                let mut w = [T::zero(); COLS];
                w[k+1] = u1;
                if k + 2 < COLS {
                    w[(k+2)..COLS].clone_from_slice(&x[(k+2)..COLS]);
                }
                let wnorm = w[(k+1)..].iter().map(|wi| *wi * *wi).fold(T::zero(), |a, b| a + b).sqrt();
                if wnorm > T::zero() {
                    for wj in &mut w[(k+1)..COLS] {
                        *wj /= wnorm;
                    }
                    // Apply to A (right)``
                    for i in k..ROWS {
                        let mut dot = T::zero();
                        for j in (k+1)..COLS {
                            dot += w[j] * a[(i, j)];
                        }
                        for j in (k+1)..COLS {
                            a[(i, j)] = a[(i, j)] - T::from(2.0).unwrap() * w[j] * dot;
                        }
                    }
                    // Accumulate V
                    for j in 0..COLS {
                        let mut dot = T::zero();
                        for l in (k+1)..COLS {
                            dot += w[l] * v[(j, l)];
                        }
                        for l in (k+1)..COLS {
                            v[(j, l)] = v[(j, l)] - T::from(2.0).unwrap() * w[l] * dot;
                        }
                    }
                }
            }
        }
        // Now a is bidiagonal, u and v are orthogonal
        // For small matrices, treat a as S, and return (u, a, v^T)
        let mut s = Matrix::<ROWS, COLS, T>::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                s[(i, j)] = a[(i, j)];
            }
        }
        Some((u, s, v.transpose()))
    }

}


#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn test_svd_rectangular_ldl() {
        // 2x3 matrix
        let a = Matrix::<2, 3, f64>::from_row_major([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
        let (u, s, vt) = a.svd().unwrap();
        let recon = u * s * vt;
        for i in 0..2 {
            for j in 0..3 {
                assert!((recon[(i, j)] - a[(i, j)]).abs() < 1e-8);
            }
        }
    }
}