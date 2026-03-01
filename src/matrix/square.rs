use crate::traits::{LinalgScalar, Scalar};
use crate::matrix::vector::Vector;
use crate::Matrix;

impl<T: Scalar, const N: usize> Matrix<T, N, N> {
    /// Sum of diagonal elements.
    ///
    /// ```
    /// use numeris::Matrix;
    /// let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(m.trace(), 5.0);
    /// ```
    pub fn trace(&self) -> T {
        let mut sum = T::zero();
        for i in 0..N {
            sum = sum + self[(i, i)];
        }
        sum
    }

    /// Extract the diagonal as a vector.
    ///
    /// ```
    /// use numeris::Matrix;
    /// let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let d = m.diag();
    /// assert_eq!(d[0], 1.0);
    /// assert_eq!(d[1], 4.0);
    /// ```
    pub fn diag(&self) -> Vector<T, N> {
        let mut v = Vector::zeros();
        for i in 0..N {
            v[i] = self[(i, i)];
        }
        v
    }

    /// Create a diagonal matrix from a vector.
    ///
    /// ```
    /// use numeris::{Matrix, Vector};
    /// let v = Vector::from_array([2.0, 3.0]);
    /// let m = Matrix::from_diag(&v);
    /// assert_eq!(m[(0, 0)], 2.0);
    /// assert_eq!(m[(1, 1)], 3.0);
    /// assert_eq!(m[(0, 1)], 0.0);
    /// ```
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
    ///
    /// ```
    /// use numeris::Matrix;
    /// let m = Matrix::new([[1.0, 1.0], [0.0, 1.0]]);
    /// let m3 = m.pow(3);
    /// assert_eq!(m3[(0, 1)], 3.0); // upper-triangular power
    /// assert_eq!(m.pow(0), Matrix::eye());
    /// ```
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

    /// Check if the matrix is symmetric (`A == A^T`).
    ///
    /// ```
    /// use numeris::Matrix;
    /// let sym = Matrix::new([[1.0, 2.0], [2.0, 3.0]]);
    /// assert!(sym.is_symmetric());
    ///
    /// let asym = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert!(!asym.is_symmetric());
    /// ```
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
    /// Determinant via direct formulas for N<=4, Gaussian elimination otherwise.
    ///
    /// ```
    /// use numeris::Matrix;
    /// let m = Matrix::new([[3.0_f64, 8.0], [4.0, 6.0]]);
    /// assert!((m.det() - (-14.0)).abs() < 1e-12);
    /// ```
    pub fn det(&self) -> T {
        // Direct formulas for small matrices — avoid full Gaussian elimination
        if N == 1 {
            return self.data[0][0];
        }
        if N == 2 {
            // ad - bc (column-major: data[col][row])
            return self.data[0][0] * self.data[1][1] - self.data[1][0] * self.data[0][1];
        }
        if N == 3 {
            let a = self.data[0][0]; let b = self.data[1][0]; let c = self.data[2][0];
            let d = self.data[0][1]; let e = self.data[1][1]; let f = self.data[2][1];
            let g = self.data[0][2]; let h = self.data[1][2]; let i = self.data[2][2];
            return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        }
        if N == 4 {
            // Sub-determinants from rows 0-1
            let a00 = self.data[0][0]; let a01 = self.data[1][0];
            let a02 = self.data[2][0]; let a03 = self.data[3][0];
            let a10 = self.data[0][1]; let a11 = self.data[1][1];
            let a12 = self.data[2][1]; let a13 = self.data[3][1];
            let a20 = self.data[0][2]; let a21 = self.data[1][2];
            let a22 = self.data[2][2]; let a23 = self.data[3][2];
            let a30 = self.data[0][3]; let a31 = self.data[1][3];
            let a32 = self.data[2][3]; let a33 = self.data[3][3];

            let s0 = a00 * a11 - a01 * a10;
            let s1 = a00 * a12 - a02 * a10;
            let s2 = a00 * a13 - a03 * a10;
            let s3 = a01 * a12 - a02 * a11;
            let s4 = a01 * a13 - a03 * a11;
            let s5 = a02 * a13 - a03 * a12;

            let c0 = a20 * a31 - a21 * a30;
            let c1 = a20 * a32 - a22 * a30;
            let c2 = a20 * a33 - a23 * a30;
            let c3 = a21 * a32 - a22 * a31;
            let c4 = a21 * a33 - a23 * a31;
            let c5 = a22 * a33 - a23 * a32;

            return s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
        }

        // General case: Gaussian elimination with partial pivoting
        let mut a = *self;
        let mut sign = T::one();

        for col in 0..N {
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

            if max_row != col {
                for j in 0..N {
                    let tmp = a[(col, j)];
                    a[(col, j)] = a[(max_row, j)];
                    a[(max_row, j)] = tmp;
                }
                sign = T::zero() - sign;
            }

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
