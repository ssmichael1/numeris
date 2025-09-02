use super::{Matrix, MatrixElem};

/// Implementation for square matrices
impl<const N: usize, T> Matrix<N, N, T>
where
    T: MatrixElem,
{
    /// Identity matrix
    pub fn identity() -> Self {
        let mut data = [[T::zero(); N]; N];
        for (i, row) in data.iter_mut().enumerate() {
            row[i] = T::one();
        }
        Self { data }
    }

    /// Compute determinant for larger-size matrices
    /// The minor size (N-1) must be explicitly passed in as generic parameter
    ///
    /// Note: convenience function "determinant" does this automatically
    /// for matrices up to 9x9
    ///
    fn determinant_largemat<const RM1: usize>(&self) -> T {
        let mut det = T::zero();
        for col in 0..N {
            let minor = self.minor::<RM1, RM1>(0, col);
            det += self.data[0][col] * minor.determinant();
        }
        det
    }

    // Explicitly compute inverse for larger-size matrices
    fn inv_largemat<const RM1: usize>(&self) -> Option<Self> {
        let det = self.determinant();
        if det == T::zero() {
            return None;
        }
        let mut cofactors = [[T::zero(); N]; N];
        for (r, row) in cofactors.iter_mut().enumerate() {
            for (c, elem) in row.iter_mut().enumerate() {
                let minor = self.minor::<RM1, RM1>(r, c);
                let sign = match (r + c) % 2 {
                    0 => T::one(),
                    _ => -T::one(),
                };
                *elem = sign * minor.determinant();
            }
        }
        // Transpose the cofactor matrix to get the adjugate
        let mut adjugate = [[T::zero(); N]; N];
        for (r, row) in cofactors.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                adjugate[c][r] = val;
            }
        }
        // Divide by determinant to get the inverse
        let mut inverse = Self::zeros();

        for r in 0..N {
            for c in 0..N {
                inverse[(r, c)] = adjugate[r][c] / det;
            }
        }
        Some(inverse)
    }

    /// Matrix inverse for matrices up to 9x9
    ///
    /// This function computes the inverse of a square matrix using
    /// the adjugate method. It is only implemented for matrices
    /// up to 9x9 in size.
    ///
    /// Could be larger if rust allowed math on const generics
    /// (maybe someday...)
    ///
    /// # Returns:
    ///
    /// The inverse of the matrix, if it exists, else None
    ///
    pub fn inv(&self) -> Option<Self> {
        if N == 1 {
            let mut data = [[T::zero(); N]; N];
            data[0][0] = T::one() / self.data[0][0];
            return Some(Self { data });
        } else if N == 2 {
            let det = self.determinant();
            if det == T::zero() {
                return None;
            }
            let mut data = [[T::zero(); N]; N];
            data[0][0] = self.data[1][1] / det;
            data[0][1] = -self.data[0][1] / det;
            data[1][0] = -self.data[1][0] / det;
            data[1][1] = self.data[0][0] / det;
            return Some(Self { data });
        } else if N == 3 {
            return self.inv_largemat::<2>();
        } else if N == 4 {
            return self.inv_largemat::<3>();
        } else if N == 5 {
            return self.inv_largemat::<4>();
        } else if N == 6 {
            return self.inv_largemat::<5>();
        } else if N == 7 {
            return self.inv_largemat::<6>();
        } else if N == 8 {
            return self.inv_largemat::<7>();
        } else if N == 9 {
            return self.inv_largemat::<8>();
        }
        None
    }

    /// Compute the determinant of the matrix
    ///
    /// Note: due to inability of rust to handle math with generics,
    /// the matrix minor sizes must be explicitly stated, so this
    /// function only supports up to 10x10 matrices
    ///
    /// Larger matrices will result in a panic
    /// Should not be issue since matrix sizes are known at compile time
    ///
    ///
    /// Returns:
    ///
    /// The determinant of the matrix
    ///
    pub fn determinant(&self) -> T {
        if N == 1 {
            self.data[0][0]
        } else if N == 2 {
            self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        } else if N == 3 {
            let a = self.data[0][0];
            let b = self.data[0][1];
            let c = self.data[0][2];
            let d = self.data[1][0];
            let e = self.data[1][1];
            let f = self.data[1][2];
            let g = self.data[2][0];
            let h = self.data[2][1];
            let i = self.data[2][2];
            a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        } else if N == 4 {
            self.determinant_largemat::<3>()
        } else if N == 5 {
            self.determinant_largemat::<4>()
        } else if N == 6 {
            self.determinant_largemat::<5>()
        } else if N == 7 {
            self.determinant_largemat::<6>()
        } else if N == 8 {
            self.determinant_largemat::<7>()
        } else if N == 9 {
            self.determinant_largemat::<8>()
        } else {
            panic!("Determinant not implemented for matrices larger than 9x9 (need better generic handling in rust)")
        }
    }

}

impl<const N: usize, T> Matrix<N, N, T>
where T: MatrixElem + num_traits::Float
{
    /// QR factorization using Gram-Schmidt process
    ///
    /// Returns (Q, R) where Q is orthogonal and R is upper triangular
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::matrix::Matrix;
    /// let a = Matrix::<3, 3, f64>::from_row_major([
    ///     [12.0, -51.0, 4.0],
    ///     [6.0, 167.0, -68.0],
    ///     [-4.0, 24.0, -41.0],
    /// ]);
    /// let (q, r) = a.qr();
    /// let a2 = q * r;
    /// for i in 0..3 {
    ///     for j in 0..3 {
    ///         assert!((a2[(i, j)] - a[(i, j)]).abs() < 1e-8);
    ///     }
    /// }
    /// ```
    pub fn qr(&self) -> (Self, Self) {
        let mut q = Self::zeros();
        let mut r = Self::zeros();
        let a = *self;
        // Gram-Schmidt
        for j in 0..N {
            // v_j = a_j
            let mut v = [T::zero(); N];
            for i in 0..N {
                v[i] = a[(i, j)];
            }
            for k in 0..j {
                let mut r_kj = T::zero();
                for i in 0..N {
                    r_kj += q[(i, k)] * a[(i, j)];
                }
                r[(k, j)] = r_kj;
                for i in 0..N {
                    v[i] = v[i] - r_kj * q[(i, k)];
                }
            }
            let norm = v.iter().map(|x| *x * *x).fold(T::zero(), |a, b| a + b).sqrt();
            r[(j, j)] = norm;
            for i in 0..N {
                q[(i, j)] = v[i] / norm;
            }
        }
        (q, r)
    }
    /// LDL^T decomposition
    ///
    /// # Returns
    ///
    /// (L, D) where L is lower-triangular with unit diagonal, D is diagonal
    /// Returns None if the matrix is not symmetric or not positive definite
    ///
    /// # Examples
    ///
    /// ```
    /// use numeris::prelude::*;
    /// use num_traits::Float;
    /// let m = Matrix::from_row_major([[4.0, 2.0], [2.0, 3.0]]);
    /// let (l, d) = m.ldl().unwrap();
    /// let m2 = &l * &d * &l.transpose();
    /// for i in 0..2 {
    ///     for j in 0..2 {
    ///         assert!((m2[(i, j)] - m[(i, j)]).abs() < 1e-10);
    ///     }
    /// }
    /// ```
    pub fn ldl(&self) -> Option<(Self, Self)> {
        let mut l = Self::identity();
        let mut d = Self::zeros();
        for j in 0..N {
            let mut sum = T::zero();
            for k in 0..j {
                sum += l[(j, k)] * l[(j, k)] * d[(k, k)];
            }
            d[(j, j)] = self[(j, j)] - sum;
            if d[(j, j)].is_zero() {
                return None;
            }
            for i in (j + 1)..N {
                let mut sum = T::zero();
                for k in 0..j {
                    sum += l[(i, k)] * l[(j, k)] * d[(k, k)];
                }
                l[(i, j)] = (self[(i, j)] - sum) / d[(j, j)];
            }
        }
        Some((l, d))
    }

    /// Cholesky decomposition
    /// 
    /// # Returns
    /// 
    /// Lower-triangular matrix of decomposition, if successful
    /// else None (Matrix must be positive definite)
    /// 
    /// # Notes
    /// 
    /// Uses the Cholesky-Crout algorithm to compute matrix
    /// column by column
    /// See: <https://en.wikipedia.org/wiki/Cholesky_decomposition>
    /// 
    /// # Examples
    ///
    /// ```
    /// use numeris::prelude::*;
    /// use num_traits::Float;
    /// let mat = Matrix::from_row_major([[4.0, 2.0], [2.0, 3.0]]);
    /// let l = mat.cholesky().unwrap();
    /// println!("l = {}", l);
    /// let c2 = l * l.transpose();
    /// for i in 0..2 {
    ///     for j in 0..2 {
    ///         assert!((c2[(i, j)] - mat[(i, j)]).abs() < 1e-10);
    ///     }
    /// }
    /// ```
    pub fn cholesky(&self) -> Option<Self> {
        if N == 1 {
            let mut data = [[T::zero(); N]; N];
            data[0][0] = self.data[0][0].sqrt();
            Some(Self::new(data))
        } else {
            let mut c = Self::zeros();
            for j in 0..N {
                let mut sum = T::zero();
                for k in 0..j {
                    sum += c[(j, k)] * c[(j, k)];
                }
                c[(j, j)] = (self[(j, j)] - sum).sqrt();
                for i in (j + 1)..N {
                    let mut sum = T::zero();
                    for k in 0..j {
                        sum += c[(i, k)] * c[(j, k)];
                    }
                    if c[(j, j)].is_zero() {
                        return None;
                    }
                    c[(i, j)] = (self[(i, j)] - sum) / c[(j, j)];
                }
            }
            Some(c)
        }
    }

}

#[cfg(test)]
mod tests {
    use num_traits::Float;

    use super::*;

    #[test]
    fn test_determinant_2x2() {
        let mat = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(mat.determinant(), -2.0);
    }

    #[test]
    fn test_determinant_3x3() {
        let mat = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        assert_eq!(mat.determinant(), 0.0);
    }

    #[test]
    fn test_cholesky() {
        let mat = Matrix::from_row_major([[4.0, 2.0], [2.0, 3.0]]);
        let l = mat.cholesky().unwrap();

        let c2 = l * l.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert!((c2[(i, j)] - mat[(i, j)]).abs() < 1e-10);
            }
        }
        }

    #[test]
    fn test_ldl() {
        let mat = Matrix::from_row_major([[4.0, 2.0], [2.0, 3.0]]);
        let (l, d) = mat.ldl().unwrap();
        // Reconstruct: L D L^T
        let c2 = l * d * l.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert!((c2[(i, j)] - mat[(i, j)]).abs() < 1e-10);
            }
        }
    }


    #[test]
    fn test_inverse_2x2() {
        let mat = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let inv = mat.inv();
        assert!(inv.is_some());
        let inv = inv.unwrap();
        assert_eq!(inv.data, [[-2.0, 1.0], [1.5, -0.5]]);
    }

    #[test]
    fn test_inverse_3x3() {
        let mat = Matrix::new([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]);
        let inv = mat.inv();
        assert!(inv.is_some());
        let matinv = Matrix::new([[-24.0, 18.0, 5.0], [20.0, -15.0, -4.0], [-5.0, 4.0, 1.0]]);
        let inv = inv.unwrap();
        assert_eq!(inv, matinv);
    }

    #[test]
    fn test_qr() {
        let a = Matrix::from_row_major([[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]]);
        let (q, r) = a.qr();
        // Check that Q*R ≈ A
        let a2 = q * r;
        for i in 0..3 {
            for j in 0..3 {
                assert!((a2[(i, j)] - a[(i, j)]).abs() < 1e-8);
            }
        }
        // Check that Q^T Q ≈ I
        let qtq = q.transpose() * q;
        let id = Matrix::<3, 3, f64>::identity();
        for i in 0..3 {
            for j in 0..3 {
                assert!((qtq[(i, j)] - id[(i, j)]).abs() < 1e-8);
            }
        }
    }
}
