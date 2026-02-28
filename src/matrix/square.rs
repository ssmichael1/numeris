use crate::traits::{LinalgScalar, Scalar};
use crate::matrix::vector::Vector;
use crate::Matrix;

impl<T: Scalar, const N: usize> Matrix<T, N, N> {
    /// Sum of diagonal elements.
    pub fn trace(&self) -> T {
        let mut sum = T::zero();
        for i in 0..N {
            sum = sum + self[(i, i)];
        }
        sum
    }

    /// Extract the diagonal as a vector.
    pub fn diag(&self) -> Vector<T, N> {
        let mut v = Vector::zeros();
        for i in 0..N {
            v[i] = self[(i, i)];
        }
        v
    }

    /// Create a diagonal matrix from a vector.
    pub fn from_diag(v: &Vector<T, N>) -> Self {
        let mut m = Self::zeros();
        for i in 0..N {
            m[(i, i)] = v[i];
        }
        m
    }

    /// Integer matrix power via repeated squaring.
    ///
    /// `pow(0)` returns the identity matrix.
    pub fn pow(&self, mut n: u32) -> Self {
        let mut result = Self::eye();
        let mut base = *self;
        while n > 0 {
            if n & 1 == 1 {
                result = result * base;
            }
            base = base * base;
            n >>= 1;
        }
        result
    }

    /// Check if the matrix is symmetric (A == A^T).
    pub fn is_symmetric(&self) -> bool {
        for i in 0..N {
            for j in (i + 1)..N {
                if self[(i, j)] != self[(j, i)] {
                    return false;
                }
            }
        }
        true
    }
}

impl<T: LinalgScalar, const N: usize> Matrix<T, N, N> {
    /// Determinant via Gaussian elimination with partial pivoting.
    pub fn det(&self) -> T {
        let mut a = *self;
        let mut sign = T::one();

        for col in 0..N {
            // Partial pivoting: find row with largest modulus
            let mut max_row = col;
            let mut max_val = a[(col, col)].modulus();
            for row in (col + 1)..N {
                let val = a[(row, col)].modulus();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }

            if max_val < T::lepsilon() {
                return T::zero();
            }

            // Swap rows if needed
            if max_row != col {
                for j in 0..N {
                    let tmp = a[(col, j)];
                    a[(col, j)] = a[(max_row, j)];
                    a[(max_row, j)] = tmp;
                }
                sign = T::zero() - sign;
            }

            // Eliminate below pivot
            let pivot = a[(col, col)];
            for row in (col + 1)..N {
                let factor = a[(row, col)] / pivot;
                for j in (col + 1)..N {
                    let val = a[(col, j)];
                    a[(row, j)] = a[(row, j)] - factor * val;
                }
                a[(row, col)] = T::zero();
            }
        }

        // Product of diagonal
        let mut det = sign;
        for i in 0..N {
            det = det * a[(i, i)];
        }
        det
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(m.trace(), 5.0);

        let id: Matrix<f64, 3, 3> = Matrix::eye();
        assert_eq!(id.trace(), 3.0);
    }

    #[test]
    fn trace_integer() {
        let m = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        assert_eq!(m.trace(), 15);
    }

    #[test]
    fn diag_and_from_diag() {
        let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let d = m.diag();
        assert_eq!(d[0], 1.0);
        assert_eq!(d[1], 5.0);
        assert_eq!(d[2], 9.0);

        let m2 = Matrix::from_diag(&d);
        assert_eq!(m2[(0, 0)], 1.0);
        assert_eq!(m2[(1, 1)], 5.0);
        assert_eq!(m2[(2, 2)], 9.0);
        assert_eq!(m2[(0, 1)], 0.0);
    }

    #[test]
    fn pow() {
        let m = Matrix::new([[1.0, 1.0], [0.0, 1.0]]);

        let m0 = m.pow(0);
        assert_eq!(m0, Matrix::eye());

        let m1 = m.pow(1);
        assert_eq!(m1, m);

        let m3 = m.pow(3);
        assert_eq!(m3[(0, 0)], 1.0);
        assert_eq!(m3[(0, 1)], 3.0);
        assert_eq!(m3[(1, 0)], 0.0);
        assert_eq!(m3[(1, 1)], 1.0);
    }

    #[test]
    fn is_symmetric() {
        let sym = Matrix::new([[1.0, 2.0, 3.0], [2.0, 5.0, 6.0], [3.0, 6.0, 9.0]]);
        assert!(sym.is_symmetric());

        let asym = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        assert!(!asym.is_symmetric());

        let id: Matrix<f64, 3, 3> = Matrix::eye();
        assert!(id.is_symmetric());
    }

    #[test]
    fn det_2x2() {
        let m = Matrix::new([[3.0_f64, 8.0], [4.0, 6.0]]);
        let d = m.det();
        assert!((d - (-14.0)).abs() < 1e-12);
    }

    #[test]
    fn det_3x3() {
        let m = Matrix::new([[6.0_f64, 1.0, 1.0], [4.0, -2.0, 5.0], [2.0, 8.0, 7.0]]);
        let d = m.det();
        assert!((d - (-306.0)).abs() < 1e-10);
    }

    #[test]
    fn det_identity() {
        let id: Matrix<f64, 4, 4> = Matrix::eye();
        assert!((id.det() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn det_singular() {
        let m = Matrix::new([[1.0_f64, 2.0], [2.0, 4.0]]);
        assert!(m.det().abs() < 1e-12);
    }
}
