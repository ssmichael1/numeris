use crate::matrix::vector::Vector;
use crate::traits::{FloatScalar, Scalar};
use crate::Matrix;

// ── Vector norms ────────────────────────────────────────────────────

impl<T: Scalar, const N: usize> Vector<T, N> {
    /// Squared L2 norm (dot product with self). No sqrt, works with integers.
    pub fn norm_squared(&self) -> T {
        self.dot(self)
    }
}

impl<T: FloatScalar, const N: usize> Vector<T, N> {
    /// L2 (Euclidean) norm.
    pub fn norm(&self) -> T {
        self.norm_squared().sqrt()
    }

    /// L1 norm (sum of absolute values).
    pub fn norm_l1(&self) -> T {
        let mut sum = T::zero();
        for i in 0..N {
            sum = sum + self[i].abs();
        }
        sum
    }

    /// Return a unit vector in the same direction.
    ///
    /// Panics if the norm is zero.
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        *self * (T::one() / n)
    }
}

// ── Matrix norms ────────────────────────────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Squared Frobenius norm (sum of all elements squared). No sqrt.
    pub fn frobenius_norm_squared(&self) -> T {
        let mut sum = T::zero();
        for i in 0..M {
            for j in 0..N {
                sum = sum + self[(i, j)] * self[(i, j)];
            }
        }
        sum
    }
}

impl<T: FloatScalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Frobenius norm (square root of sum of squared elements).
    pub fn frobenius_norm(&self) -> T {
        self.frobenius_norm_squared().sqrt()
    }

    /// Infinity norm (maximum absolute row sum).
    pub fn norm_inf(&self) -> T {
        let mut max = T::zero();
        for i in 0..M {
            let mut row_sum = T::zero();
            for j in 0..N {
                row_sum = row_sum + self[(i, j)].abs();
            }
            if row_sum > max {
                max = row_sum;
            }
        }
        max
    }

    /// One norm (maximum absolute column sum).
    pub fn norm_one(&self) -> T {
        let mut max = T::zero();
        for j in 0..N {
            let mut col_sum = T::zero();
            for i in 0..M {
                col_sum = col_sum + self[(i, j)].abs();
            }
            if col_sum > max {
                max = col_sum;
            }
        }
        max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Vector norm tests ───────────────────────────────────────

    #[test]
    fn vector_norm_squared() {
        let v = Vector::from_array([3.0, 4.0]);
        assert_eq!(v.norm_squared(), 25.0);
    }

    #[test]
    fn vector_norm_squared_integer() {
        let v = Vector::from_array([3, 4]);
        assert_eq!(v.norm_squared(), 25);
    }

    #[test]
    fn vector_norm() {
        let v = Vector::from_array([3.0_f64, 4.0]);
        assert!((v.norm() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn vector_norm_l1() {
        let v = Vector::from_array([1.0_f64, -2.0, 3.0]);
        assert!((v.norm_l1() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn vector_normalize() {
        let v = Vector::from_array([3.0_f64, 4.0]);
        let u = v.normalize();
        assert!((u.norm() - 1.0).abs() < 1e-12);
        assert!((u[0] - 0.6).abs() < 1e-12);
        assert!((u[1] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn vector_normalize_3d() {
        let v = Vector::from_array([1.0_f64, 1.0, 1.0]);
        let u = v.normalize();
        assert!((u.norm() - 1.0).abs() < 1e-12);
    }

    // ── Matrix norm tests ───────────────────────────────────────

    #[test]
    fn frobenius_norm() {
        let m = Matrix::new([[1.0_f64, 2.0], [3.0, 4.0]]);
        // sqrt(1 + 4 + 9 + 16) = sqrt(30)
        assert!((m.frobenius_norm() - 30.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn frobenius_norm_squared_integer() {
        let m = Matrix::new([[1, 2], [3, 4]]);
        assert_eq!(m.frobenius_norm_squared(), 30);
    }

    #[test]
    fn norm_inf() {
        let m = Matrix::new([[1.0_f64, -2.0], [3.0, 4.0]]);
        // row sums: |1|+|-2| = 3, |3|+|4| = 7
        assert!((m.norm_inf() - 7.0).abs() < 1e-12);
    }

    #[test]
    fn norm_one() {
        let m = Matrix::new([[1.0_f64, -2.0], [3.0, 4.0]]);
        // col sums: |1|+|3| = 4, |-2|+|4| = 6
        assert!((m.norm_one() - 6.0).abs() < 1e-12);
    }
}
