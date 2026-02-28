use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::matrix::vector::Vector;
use crate::traits::Scalar;
use crate::Matrix;

// ── Element-wise addition ───────────────────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Add for Matrix<T, M, N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut out = self;
        for i in 0..M {
            for j in 0..N {
                out[(i, j)] = self[(i, j)] + rhs[(i, j)];
            }
        }
        out
    }
}

impl<T: Scalar, const M: usize, const N: usize> AddAssign for Matrix<T, M, N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..M {
            for j in 0..N {
                self[(i, j)] = self[(i, j)] + rhs[(i, j)];
            }
        }
    }
}

// ── Element-wise subtraction ────────────────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Sub for Matrix<T, M, N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let mut out = self;
        for i in 0..M {
            for j in 0..N {
                out[(i, j)] = self[(i, j)] - rhs[(i, j)];
            }
        }
        out
    }
}

impl<T: Scalar, const M: usize, const N: usize> SubAssign for Matrix<T, M, N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..M {
            for j in 0..N {
                self[(i, j)] = self[(i, j)] - rhs[(i, j)];
            }
        }
    }
}

// ── Negation ────────────────────────────────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Neg for Matrix<T, M, N> {
    type Output = Self;

    fn neg(self) -> Self {
        let mut out = Self::zeros();
        for i in 0..M {
            for j in 0..N {
                out[(i, j)] = T::zero() - self[(i, j)];
            }
        }
        out
    }
}

impl<T: Scalar, const M: usize, const N: usize> Neg for &Matrix<T, M, N> {
    type Output = Matrix<T, M, N>;

    fn neg(self) -> Matrix<T, M, N> {
        (*self).neg()
    }
}

impl<T: Scalar, const M: usize, const N: usize> AddAssign<&Matrix<T, M, N>> for Matrix<T, M, N> {
    fn add_assign(&mut self, rhs: &Matrix<T, M, N>) {
        self.add_assign(*rhs);
    }
}

impl<T: Scalar, const M: usize, const N: usize> SubAssign<&Matrix<T, M, N>> for Matrix<T, M, N> {
    fn sub_assign(&mut self, rhs: &Matrix<T, M, N>) {
        self.sub_assign(*rhs);
    }
}

// ── Matrix multiplication: (M×N) * (N×P) → (M×P) ──────────────────

impl<T: Scalar, const M: usize, const N: usize, const P: usize> Mul<Matrix<T, N, P>>
    for Matrix<T, M, N>
{
    type Output = Matrix<T, M, P>;

    fn mul(self, rhs: Matrix<T, N, P>) -> Matrix<T, M, P> {
        let mut out = Matrix::<T, M, P>::zeros();
        for i in 0..M {
            for j in 0..P {
                let mut sum = T::zero();
                for k in 0..N {
                    sum = sum + self[(i, k)] * rhs[(k, j)];
                }
                out[(i, j)] = sum;
            }
        }
        out
    }
}

// ── Scalar multiplication: matrix * scalar ──────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Mul<T> for Matrix<T, M, N> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        let mut out = self;
        for i in 0..M {
            for j in 0..N {
                out[(i, j)] = self[(i, j)] * rhs;
            }
        }
        out
    }
}

impl<T: Scalar, const M: usize, const N: usize> MulAssign<T> for Matrix<T, M, N> {
    fn mul_assign(&mut self, rhs: T) {
        for i in 0..M {
            for j in 0..N {
                self[(i, j)] = self[(i, j)] * rhs;
            }
        }
    }
}

// ── Reference variants for same-shape binary ops ────────────────────
// Matrix is Copy, so &Matrix ops just deref and delegate.

macro_rules! forward_ref_binop {
    ($Op:ident, $method:ident) => {
        impl<T: Scalar, const M: usize, const N: usize> $Op<Matrix<T, M, N>>
            for &Matrix<T, M, N>
        {
            type Output = Matrix<T, M, N>;
            fn $method(self, rhs: Matrix<T, M, N>) -> Matrix<T, M, N> {
                (*self).$method(rhs)
            }
        }

        impl<T: Scalar, const M: usize, const N: usize> $Op<&Matrix<T, M, N>>
            for Matrix<T, M, N>
        {
            type Output = Matrix<T, M, N>;
            fn $method(self, rhs: &Matrix<T, M, N>) -> Matrix<T, M, N> {
                self.$method(*rhs)
            }
        }

        impl<T: Scalar, const M: usize, const N: usize> $Op<&Matrix<T, M, N>>
            for &Matrix<T, M, N>
        {
            type Output = Matrix<T, M, N>;
            fn $method(self, rhs: &Matrix<T, M, N>) -> Matrix<T, M, N> {
                (*self).$method(*rhs)
            }
        }
    };
}

forward_ref_binop!(Add, add);
forward_ref_binop!(Sub, sub);

// ── Reference variants for matrix multiplication ────────────────────

impl<T: Scalar, const M: usize, const N: usize, const P: usize> Mul<Matrix<T, N, P>>
    for &Matrix<T, M, N>
{
    type Output = Matrix<T, M, P>;
    fn mul(self, rhs: Matrix<T, N, P>) -> Matrix<T, M, P> {
        (*self).mul(rhs)
    }
}

impl<T: Scalar, const M: usize, const N: usize, const P: usize> Mul<&Matrix<T, N, P>>
    for Matrix<T, M, N>
{
    type Output = Matrix<T, M, P>;
    fn mul(self, rhs: &Matrix<T, N, P>) -> Matrix<T, M, P> {
        self.mul(*rhs)
    }
}

impl<T: Scalar, const M: usize, const N: usize, const P: usize> Mul<&Matrix<T, N, P>>
    for &Matrix<T, M, N>
{
    type Output = Matrix<T, M, P>;
    fn mul(self, rhs: &Matrix<T, N, P>) -> Matrix<T, M, P> {
        (*self).mul(*rhs)
    }
}

// ── Reference variant for scalar multiplication ─────────────────────

impl<T: Scalar, const M: usize, const N: usize> Mul<T> for &Matrix<T, M, N> {
    type Output = Matrix<T, M, N>;
    fn mul(self, rhs: T) -> Matrix<T, M, N> {
        (*self).mul(rhs)
    }
}

// ── scalar * matrix (concrete impls to avoid orphan rules) ──────────

macro_rules! impl_scalar_mul {
    ($($t:ty),*) => {
        $(
            impl<const M: usize, const N: usize> Mul<Matrix<$t, M, N>> for $t {
                type Output = Matrix<$t, M, N>;

                fn mul(self, rhs: Matrix<$t, M, N>) -> Matrix<$t, M, N> {
                    rhs * self
                }
            }

            impl<const M: usize, const N: usize> Mul<&Matrix<$t, M, N>> for $t {
                type Output = Matrix<$t, M, N>;

                fn mul(self, rhs: &Matrix<$t, M, N>) -> Matrix<$t, M, N> {
                    *rhs * self
                }
            }
        )*
    };
}

impl_scalar_mul!(f32, f64, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

// ── Matrix-vector product ────────────────────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Matrix-vector product: A * v → result.
    ///
    /// Takes and returns row vectors for convenience, avoiding
    /// explicit transpose. Equivalent to `(A * v^T)^T`.
    pub fn vecmul(&self, v: &Vector<T, N>) -> Vector<T, M> {
        let mut out = Vector::<T, M>::zeros();
        for i in 0..M {
            let mut sum = T::zero();
            for j in 0..N {
                sum = sum + self[(i, j)] * v[j];
            }
            out[i] = sum;
        }
        out
    }
}

// ── Element-wise multiplication (Hadamard product) ──────────────────

impl<T: Scalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Element-wise (Hadamard) product: `c[i][j] = a[i][j] * b[i][j]`.
    pub fn element_mul(&self, rhs: &Self) -> Self {
        let mut out = *self;
        for i in 0..M {
            for j in 0..N {
                out[(i, j)] = self[(i, j)] * rhs[(i, j)];
            }
        }
        out
    }
}

// ── Transpose ───────────────────────────────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Transpose: (M×N) → (N×M).
    pub fn transpose(&self) -> Matrix<T, N, M> {
        let mut out = Matrix::<T, N, M>::zeros();
        for i in 0..M {
            for j in 0..N {
                out[(j, i)] = self[(i, j)];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_sub() {
        let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);

        let c = a + b;
        assert_eq!(c[(0, 0)], 6.0);
        assert_eq!(c[(1, 1)], 12.0);

        let d = b - a;
        assert_eq!(d[(0, 0)], 4.0);
        assert_eq!(d[(1, 1)], 4.0);
    }

    #[test]
    fn add_assign_sub_assign() {
        let mut a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);

        a += b;
        assert_eq!(a[(0, 0)], 6.0);

        a -= b;
        assert_eq!(a[(0, 0)], 1.0);
    }

    #[test]
    fn negation() {
        let a = Matrix::new([[1.0, -2.0], [3.0, -4.0]]);
        let b = -a;
        assert_eq!(b[(0, 0)], -1.0);
        assert_eq!(b[(0, 1)], 2.0);
    }

    #[test]
    fn matrix_multiply() {
        let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);

        let c = a * b;
        assert_eq!(c[(0, 0)], 19.0); // 1*5 + 2*7
        assert_eq!(c[(0, 1)], 22.0); // 1*6 + 2*8
        assert_eq!(c[(1, 0)], 43.0); // 3*5 + 4*7
        assert_eq!(c[(1, 1)], 50.0); // 3*6 + 4*8
    }

    #[test]
    fn matrix_multiply_non_square() {
        // (2×3) * (3×2) → (2×2)
        let a = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let b = Matrix::new([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);

        let c = a * b;
        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 2);
        assert_eq!(c[(0, 0)], 58.0); // 1*7 + 2*9 + 3*11
        assert_eq!(c[(0, 1)], 64.0); // 1*8 + 2*10 + 3*12
    }

    #[test]
    fn scalar_multiply() {
        let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);

        let b = a * 3.0;
        assert_eq!(b[(0, 0)], 3.0);
        assert_eq!(b[(1, 1)], 12.0);

        let c = 3.0 * a;
        assert_eq!(c, b);
    }

    #[test]
    fn mul_assign_scalar() {
        let mut a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        a *= 2.0;
        assert_eq!(a[(0, 0)], 2.0);
        assert_eq!(a[(1, 1)], 8.0);
    }

    #[test]
    fn transpose() {
        let a = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let t = a.transpose();

        assert_eq!(t.nrows(), 3);
        assert_eq!(t.ncols(), 2);
        assert_eq!(t[(0, 0)], 1.0);
        assert_eq!(t[(1, 0)], 2.0);
        assert_eq!(t[(2, 1)], 6.0);
    }

    #[test]
    fn ref_add_sub() {
        let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);

        assert_eq!(&a + b, a + b);
        assert_eq!(a + &b, a + b);
        assert_eq!(&a + &b, a + b);

        assert_eq!(&b - a, b - a);
        assert_eq!(b - &a, b - a);
        assert_eq!(&b - &a, b - a);
    }

    #[test]
    fn ref_matrix_multiply() {
        let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);
        let expected = a * b;

        assert_eq!(&a * b, expected);
        assert_eq!(a * &b, expected);
        assert_eq!(&a * &b, expected);
    }

    #[test]
    fn ref_scalar_multiply() {
        let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let expected = a * 3.0;

        assert_eq!(&a * 3.0, expected);
        assert_eq!(3.0 * &a, expected);
    }

    #[test]
    fn ref_neg() {
        let a = Matrix::new([[1.0, -2.0], [3.0, -4.0]]);
        assert_eq!(-&a, -a);
    }

    #[test]
    fn ref_assign_ops() {
        let mut a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);

        a += &b;
        assert_eq!(a[(0, 0)], 6.0);

        a -= &b;
        assert_eq!(a[(0, 0)], 1.0);
    }

    #[test]
    fn identity_multiply() {
        let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let id: Matrix<f64, 2, 2> = Matrix::eye();
        assert_eq!(a * id, a);
        assert_eq!(id * a, a);
    }

    #[test]
    fn vecmul_square() {
        let a = Matrix::new([[2.0, 1.0], [5.0, 3.0]]);
        let v = Vector::from_array([1.0, 2.0]);
        let result = a.vecmul(&v);
        assert_eq!(result[0], 4.0);  // 2*1 + 1*2
        assert_eq!(result[1], 11.0); // 5*1 + 3*2
    }

    #[test]
    fn vecmul_non_square() {
        // (2×3) * vec(3) → vec(2)
        let a = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let v = Vector::from_array([7.0, 8.0, 9.0]);
        let result = a.vecmul(&v);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 50.0);  // 1*7 + 2*8 + 3*9
        assert_eq!(result[1], 122.0); // 4*7 + 5*8 + 6*9
    }

    #[test]
    fn vecmul_identity() {
        let id: Matrix<f64, 3, 3> = Matrix::eye();
        let v = Vector::from_array([1.0, 2.0, 3.0]);
        assert_eq!(id.vecmul(&v), v);
    }

    #[test]
    fn element_mul() {
        let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);
        let c = a.element_mul(&b);
        assert_eq!(c[(0, 0)], 5.0);
        assert_eq!(c[(0, 1)], 12.0);
        assert_eq!(c[(1, 0)], 21.0);
        assert_eq!(c[(1, 1)], 32.0);
    }
}
