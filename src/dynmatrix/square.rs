use crate::traits::{LinalgScalar, Scalar};

use super::vector::DynVector;
use super::DynMatrix;

impl<T: Scalar> DynMatrix<T> {
    /// Sum of diagonal elements.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::from_rows(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(m.trace(), 5.0);
    /// ```
    pub fn trace(&self) -> T {
        let n = self.nrows.min(self.ncols);
        let mut sum = T::zero();
        for i in 0..n {
            sum = sum + self[(i, i)];
        }
        sum
    }

    /// Extract the diagonal as a `DynVector`.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::from_rows(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    /// let d = m.diag();
    /// assert_eq!(d[0], 1.0);
    /// assert_eq!(d[1], 4.0);
    /// ```
    pub fn diag(&self) -> DynVector<T> {
        let n = self.nrows.min(self.ncols);
        let mut data = alloc::vec::Vec::with_capacity(n);
        for i in 0..n {
            data.push(self[(i, i)]);
        }
        DynVector::from_vec(data)
    }

    /// Create a square diagonal matrix from a vector.
    ///
    /// ```
    /// use numeris::{DynMatrix, DynVector};
    /// let v = DynVector::from_slice(&[2.0, 3.0]);
    /// let m = DynMatrix::from_diag(&v);
    /// assert_eq!(m[(0, 0)], 2.0);
    /// assert_eq!(m[(1, 1)], 3.0);
    /// assert_eq!(m[(0, 1)], 0.0);
    /// ```
    pub fn from_diag(v: &DynVector<T>) -> Self {
        let n = v.len();
        let mut m = Self::zeros(n, n, T::zero());
        for i in 0..n {
            m[(i, i)] = v[i];
        }
        m
    }

    /// Integer matrix power via repeated squaring.
    ///
    /// `pow(0)` returns the identity matrix. Panics if not square.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::from_rows(2, 2, &[1.0, 1.0, 0.0, 1.0]);
    /// let m3 = m.pow(3);
    /// assert_eq!(m3[(0, 1)], 3.0);
    /// ```
    pub fn pow(&self, mut n: u32) -> Self {
        assert!(self.is_square(), "pow requires a square matrix");
        let sz = self.nrows;
        let mut result = Self::eye(sz, T::zero());
        let mut base = self.clone();
        while n > 0 {
            if n & 1 == 1 {
                result = &result * &base;
            }
            base = &base * &base;
            n >>= 1;
        }
        result
    }

    /// Check if the matrix is symmetric (`A == A^T`).
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let sym = DynMatrix::from_rows(2, 2, &[1.0, 2.0, 2.0, 3.0]);
    /// assert!(sym.is_symmetric());
    /// ```
    pub fn is_symmetric(&self) -> bool {
        if !self.is_square() {
            return false;
        }
        let n = self.nrows;
        for i in 0..n {
            for j in (i + 1)..n {
                if self[(i, j)] != self[(j, i)] {
                    return false;
                }
            }
        }
        true
    }
}

impl<T: LinalgScalar> DynMatrix<T> {
    /// Determinant via Gaussian elimination with partial pivoting.
    ///
    /// Panics if the matrix is not square.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::from_rows(2, 2, &[3.0_f64, 8.0, 4.0, 6.0]);
    /// assert!((m.det() - (-14.0)).abs() < 1e-12);
    /// ```
    pub fn det(&self) -> T {
        assert!(self.is_square(), "determinant requires a square matrix");
        let n = self.nrows;
        let mut a = self.clone();
        let mut sign = T::one();

        for col in 0..n {
            let mut max_row = col;
            let mut max_val = a[(col, col)].modulus();
            for row in (col + 1)..n {
                let val = a[(row, col)].modulus();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }

            if max_val < T::lepsilon() {
                return T::zero();
            }

            if max_row != col {
                a.swap_rows(col, max_row);
                sign = T::zero() - sign;
            }

            let pivot = a[(col, col)];
            for row in (col + 1)..n {
                let factor = a[(row, col)] / pivot;
                for j in (col + 1)..n {
                    let val = a[(col, j)];
                    a[(row, j)] = a[(row, j)] - factor * val;
                }
                a[(row, col)] = T::zero();
            }
        }

        let mut det = sign;
        for i in 0..n {
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
        let m = DynMatrix::from_rows(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m.trace(), 5.0);

        let id = DynMatrix::eye(3, 0.0_f64);
        assert_eq!(id.trace(), 3.0);
    }

    #[test]
    fn diag_and_from_diag() {
        let m = DynMatrix::from_fn(3, 3, |i, j| (i * 3 + j + 1) as f64);
        let d = m.diag();
        assert_eq!(d[0], 1.0);
        assert_eq!(d[1], 5.0);
        assert_eq!(d[2], 9.0);

        let m2 = DynMatrix::from_diag(&d);
        assert_eq!(m2[(0, 0)], 1.0);
        assert_eq!(m2[(1, 1)], 5.0);
        assert_eq!(m2[(2, 2)], 9.0);
        assert_eq!(m2[(0, 1)], 0.0);
    }

    #[test]
    fn pow() {
        let m = DynMatrix::from_rows(2, 2, &[1.0, 1.0, 0.0, 1.0]);

        let m0 = m.pow(0);
        assert_eq!(m0, DynMatrix::eye(2, 0.0_f64));

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
        let sym = DynMatrix::from_rows(2, 2, &[1.0, 2.0, 2.0, 3.0]);
        assert!(sym.is_symmetric());

        let asym = DynMatrix::from_rows(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!(!asym.is_symmetric());

        let rect = DynMatrix::from_rows(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(!rect.is_symmetric());
    }

    #[test]
    fn det_2x2() {
        let m = DynMatrix::from_rows(2, 2, &[3.0_f64, 8.0, 4.0, 6.0]);
        assert!((m.det() - (-14.0)).abs() < 1e-12);
    }

    #[test]
    fn det_3x3() {
        let m = DynMatrix::from_rows(
            3,
            3,
            &[6.0_f64, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0],
        );
        assert!((m.det() - (-306.0)).abs() < 1e-10);
    }

    #[test]
    fn det_identity() {
        let id = DynMatrix::eye(4, 0.0_f64);
        assert!((id.det() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn det_singular() {
        let m = DynMatrix::from_rows(2, 2, &[1.0_f64, 2.0, 2.0, 4.0]);
        assert!(m.det().abs() < 1e-12);
    }
}
