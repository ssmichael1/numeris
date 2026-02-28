use alloc::vec;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::traits::Scalar;

use super::DynMatrix;

// ── Element-wise addition ───────────────────────────────────────────

impl<T: Scalar> Add for DynMatrix<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        assert_eq!(
            (self.nrows, self.ncols),
            (rhs.nrows, rhs.ncols),
            "dimension mismatch: {}x{} + {}x{}",
            self.nrows, self.ncols, rhs.nrows, rhs.ncols,
        );
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        DynMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

impl<T: Scalar> Add<&DynMatrix<T>> for DynMatrix<T> {
    type Output = DynMatrix<T>;
    fn add(self, rhs: &DynMatrix<T>) -> DynMatrix<T> {
        assert_eq!(
            (self.nrows, self.ncols),
            (rhs.nrows, rhs.ncols),
            "dimension mismatch: {}x{} + {}x{}",
            self.nrows, self.ncols, rhs.nrows, rhs.ncols,
        );
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        DynMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

impl<T: Scalar> Add<DynMatrix<T>> for &DynMatrix<T> {
    type Output = DynMatrix<T>;
    fn add(self, rhs: DynMatrix<T>) -> DynMatrix<T> {
        rhs + self
    }
}

impl<T: Scalar> Add<&DynMatrix<T>> for &DynMatrix<T> {
    type Output = DynMatrix<T>;
    fn add(self, rhs: &DynMatrix<T>) -> DynMatrix<T> {
        assert_eq!(
            (self.nrows, self.ncols),
            (rhs.nrows, rhs.ncols),
            "dimension mismatch: {}x{} + {}x{}",
            self.nrows, self.ncols, rhs.nrows, rhs.ncols,
        );
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        DynMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

impl<T: Scalar> AddAssign for DynMatrix<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}

impl<T: Scalar> AddAssign<&DynMatrix<T>> for DynMatrix<T> {
    fn add_assign(&mut self, rhs: &DynMatrix<T>) {
        assert_eq!(
            (self.nrows, self.ncols),
            (rhs.nrows, rhs.ncols),
            "dimension mismatch: {}x{} += {}x{}",
            self.nrows, self.ncols, rhs.nrows, rhs.ncols,
        );
        for (a, &b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a = *a + b;
        }
    }
}

// ── Element-wise subtraction ────────────────────────────────────────

impl<T: Scalar> Sub for DynMatrix<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        assert_eq!(
            (self.nrows, self.ncols),
            (rhs.nrows, rhs.ncols),
            "dimension mismatch: {}x{} - {}x{}",
            self.nrows, self.ncols, rhs.nrows, rhs.ncols,
        );
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        DynMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

impl<T: Scalar> Sub<&DynMatrix<T>> for DynMatrix<T> {
    type Output = DynMatrix<T>;
    fn sub(self, rhs: &DynMatrix<T>) -> DynMatrix<T> {
        assert_eq!(
            (self.nrows, self.ncols),
            (rhs.nrows, rhs.ncols),
            "dimension mismatch",
        );
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        DynMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

impl<T: Scalar> Sub<DynMatrix<T>> for &DynMatrix<T> {
    type Output = DynMatrix<T>;
    fn sub(self, rhs: DynMatrix<T>) -> DynMatrix<T> {
        assert_eq!(
            (self.nrows, self.ncols),
            (rhs.nrows, rhs.ncols),
            "dimension mismatch",
        );
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        DynMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

impl<T: Scalar> Sub<&DynMatrix<T>> for &DynMatrix<T> {
    type Output = DynMatrix<T>;
    fn sub(self, rhs: &DynMatrix<T>) -> DynMatrix<T> {
        assert_eq!(
            (self.nrows, self.ncols),
            (rhs.nrows, rhs.ncols),
            "dimension mismatch",
        );
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        DynMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

impl<T: Scalar> SubAssign for DynMatrix<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs);
    }
}

impl<T: Scalar> SubAssign<&DynMatrix<T>> for DynMatrix<T> {
    fn sub_assign(&mut self, rhs: &DynMatrix<T>) {
        assert_eq!(
            (self.nrows, self.ncols),
            (rhs.nrows, rhs.ncols),
            "dimension mismatch",
        );
        for (a, &b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a = *a - b;
        }
    }
}

// ── Negation ────────────────────────────────────────────────────────

impl<T: Scalar> Neg for DynMatrix<T> {
    type Output = Self;

    fn neg(self) -> Self {
        let data = self.data.iter().map(|&x| T::zero() - x).collect();
        DynMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

impl<T: Scalar> Neg for &DynMatrix<T> {
    type Output = DynMatrix<T>;

    fn neg(self) -> DynMatrix<T> {
        let data = self.data.iter().map(|&x| T::zero() - x).collect();
        DynMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

// ── Matrix multiplication: (M×N) * (N×P) → (M×P) ──────────────────

impl<T: Scalar> Mul for DynMatrix<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        &self * &rhs
    }
}

impl<T: Scalar> Mul<&DynMatrix<T>> for DynMatrix<T> {
    type Output = DynMatrix<T>;
    fn mul(self, rhs: &DynMatrix<T>) -> DynMatrix<T> {
        &self * rhs
    }
}

impl<T: Scalar> Mul<DynMatrix<T>> for &DynMatrix<T> {
    type Output = DynMatrix<T>;
    fn mul(self, rhs: DynMatrix<T>) -> DynMatrix<T> {
        self * &rhs
    }
}

impl<T: Scalar> Mul<&DynMatrix<T>> for &DynMatrix<T> {
    type Output = DynMatrix<T>;

    fn mul(self, rhs: &DynMatrix<T>) -> DynMatrix<T> {
        assert_eq!(
            self.ncols, rhs.nrows,
            "dimension mismatch: {}x{} * {}x{}",
            self.nrows, self.ncols, rhs.nrows, rhs.ncols,
        );
        let m = self.nrows;
        let n = self.ncols;
        let p = rhs.ncols;
        let mut data = vec![T::zero(); m * p];
        for i in 0..m {
            for k in 0..n {
                let a_ik = self.data[i * n + k];
                for j in 0..p {
                    data[i * p + j] = data[i * p + j] + a_ik * rhs.data[k * p + j];
                }
            }
        }
        DynMatrix {
            data,
            nrows: m,
            ncols: p,
        }
    }
}

// ── Scalar multiplication: matrix * scalar ──────────────────────────

impl<T: Scalar> Mul<T> for DynMatrix<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        let data = self.data.iter().map(|&x| x * rhs).collect();
        DynMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

impl<T: Scalar> Mul<T> for &DynMatrix<T> {
    type Output = DynMatrix<T>;

    fn mul(self, rhs: T) -> DynMatrix<T> {
        let data = self.data.iter().map(|&x| x * rhs).collect();
        DynMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

impl<T: Scalar> MulAssign<T> for DynMatrix<T> {
    fn mul_assign(&mut self, rhs: T) {
        for x in self.data.iter_mut() {
            *x = *x * rhs;
        }
    }
}

// ── scalar * matrix (concrete impls) ────────────────────────────────

macro_rules! impl_scalar_mul_dyn {
    ($($t:ty),*) => {
        $(
            impl Mul<DynMatrix<$t>> for $t {
                type Output = DynMatrix<$t>;
                fn mul(self, rhs: DynMatrix<$t>) -> DynMatrix<$t> {
                    rhs * self
                }
            }

            impl Mul<&DynMatrix<$t>> for $t {
                type Output = DynMatrix<$t>;
                fn mul(self, rhs: &DynMatrix<$t>) -> DynMatrix<$t> {
                    rhs * self
                }
            }
        )*
    };
}

impl_scalar_mul_dyn!(f32, f64, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

// ── Scalar division: matrix / scalar ─────────────────────────────────

impl<T: Scalar> Div<T> for DynMatrix<T> {
    type Output = Self;

    fn div(self, rhs: T) -> Self {
        let data = self.data.iter().map(|&x| x / rhs).collect();
        DynMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

impl<T: Scalar> Div<T> for &DynMatrix<T> {
    type Output = DynMatrix<T>;

    fn div(self, rhs: T) -> DynMatrix<T> {
        let data = self.data.iter().map(|&x| x / rhs).collect();
        DynMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

impl<T: Scalar> DivAssign<T> for DynMatrix<T> {
    fn div_assign(&mut self, rhs: T) {
        for x in self.data.iter_mut() {
            *x = *x / rhs;
        }
    }
}

// ── Element-wise multiplication (Hadamard product) ──────────────────

impl<T: Scalar> DynMatrix<T> {
    /// Element-wise (Hadamard) product: `c[i][j] = a[i][j] * b[i][j]`.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let a = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    /// let b = DynMatrix::from_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
    /// let c = a.element_mul(&b);
    /// assert_eq!(c[(0, 0)], 5.0);
    /// assert_eq!(c[(1, 1)], 32.0);
    /// ```
    pub fn element_mul(&self, rhs: &Self) -> Self {
        assert_eq!(
            (self.nrows, self.ncols),
            (rhs.nrows, rhs.ncols),
            "dimension mismatch",
        );
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        DynMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    /// Element-wise division: `c[i][j] = a[i][j] / b[i][j]`.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let a = DynMatrix::from_slice(2, 2, &[10.0, 12.0, 21.0, 32.0]);
    /// let b = DynMatrix::from_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
    /// let c = a.element_div(&b);
    /// assert_eq!(c[(0, 0)], 2.0);
    /// assert_eq!(c[(1, 1)], 4.0);
    /// ```
    pub fn element_div(&self, rhs: &Self) -> Self {
        assert_eq!(
            (self.nrows, self.ncols),
            (rhs.nrows, rhs.ncols),
            "dimension mismatch",
        );
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&a, &b)| a / b)
            .collect();
        DynMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    /// Transpose: (M×N) → (N×M).
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let a = DynMatrix::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let t = a.transpose();
    /// assert_eq!(t.nrows(), 3);
    /// assert_eq!(t.ncols(), 2);
    /// assert_eq!(t[(1, 0)], 2.0);
    /// ```
    pub fn transpose(&self) -> Self
    where
        T: Copy,
    {
        let m = self.nrows;
        let n = self.ncols;
        DynMatrix::from_fn(n, m, |i, j| self.data[j * n + i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_sub() {
        let a = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = DynMatrix::from_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);

        let c = &a + &b;
        assert_eq!(c[(0, 0)], 6.0);
        assert_eq!(c[(1, 1)], 12.0);

        let d = &b - &a;
        assert_eq!(d[(0, 0)], 4.0);
        assert_eq!(d[(1, 1)], 4.0);
    }

    #[test]
    fn add_assign() {
        let mut a = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = DynMatrix::from_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        a += &b;
        assert_eq!(a[(0, 0)], 6.0);
        a -= &b;
        assert_eq!(a[(0, 0)], 1.0);
    }

    #[test]
    fn neg() {
        let a = DynMatrix::from_slice(2, 2, &[1.0, -2.0, 3.0, -4.0]);
        let b = -a;
        assert_eq!(b[(0, 0)], -1.0);
        assert_eq!(b[(0, 1)], 2.0);
    }

    #[test]
    fn matrix_multiply() {
        let a = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = DynMatrix::from_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        let c = &a * &b;
        assert_eq!(c[(0, 0)], 19.0);
        assert_eq!(c[(0, 1)], 22.0);
        assert_eq!(c[(1, 0)], 43.0);
        assert_eq!(c[(1, 1)], 50.0);
    }

    #[test]
    fn matrix_multiply_non_square() {
        let a = DynMatrix::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = DynMatrix::from_slice(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = &a * &b;
        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 2);
        assert_eq!(c[(0, 0)], 58.0);
        assert_eq!(c[(0, 1)], 64.0);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn multiply_dim_mismatch() {
        let a = DynMatrix::from_slice(2, 3, &[0.0; 6]);
        let b = DynMatrix::from_slice(2, 2, &[0.0; 4]);
        let _ = &a * &b;
    }

    #[test]
    fn scalar_multiply() {
        let a = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = &a * 3.0;
        assert_eq!(b[(0, 0)], 3.0);
        assert_eq!(b[(1, 1)], 12.0);

        let c = 3.0 * &a;
        assert_eq!(c, b);
    }

    #[test]
    fn scalar_divide() {
        let a = DynMatrix::from_slice(2, 2, &[2.0, 4.0, 6.0, 8.0]);
        let b = &a / 2.0;
        assert_eq!(b[(0, 0)], 1.0);
        assert_eq!(b[(1, 1)], 4.0);
    }

    #[test]
    fn mul_div_assign() {
        let mut a = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        a *= 2.0;
        assert_eq!(a[(0, 0)], 2.0);
        a /= 2.0;
        assert_eq!(a[(0, 0)], 1.0);
    }

    #[test]
    fn element_mul() {
        let a = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = DynMatrix::from_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        let c = a.element_mul(&b);
        assert_eq!(c[(0, 0)], 5.0);
        assert_eq!(c[(1, 1)], 32.0);
    }

    #[test]
    fn element_div() {
        let a = DynMatrix::from_slice(2, 2, &[10.0, 12.0, 21.0, 32.0]);
        let b = DynMatrix::from_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        let c = a.element_div(&b);
        assert_eq!(c[(0, 0)], 2.0);
        assert_eq!(c[(1, 1)], 4.0);
    }

    #[test]
    fn transpose() {
        let a = DynMatrix::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        assert_eq!(t.nrows(), 3);
        assert_eq!(t.ncols(), 2);
        assert_eq!(t[(0, 0)], 1.0);
        assert_eq!(t[(1, 0)], 2.0);
        assert_eq!(t[(2, 1)], 6.0);
    }

    #[test]
    fn ref_variants() {
        let a = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = DynMatrix::from_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);

        // All ref combinations should produce the same result
        let sum1 = &a + &b;
        let sum2 = a.clone() + &b;
        let sum3 = &a + b.clone();
        let sum4 = a.clone() + b.clone();
        assert_eq!(sum1, sum2);
        assert_eq!(sum1, sum3);
        assert_eq!(sum1, sum4);
    }

    #[test]
    fn identity_multiply() {
        let a = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let id = DynMatrix::eye(2, 0.0_f64);
        assert_eq!(&a * &id, a);
        assert_eq!(&id * &a, a);
    }
}
