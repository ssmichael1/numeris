use num_traits::Zero;

use crate::traits::{LinalgScalar, Scalar};

use super::vector::DynVector;
use super::DynMatrix;

// ── Vector norms ────────────────────────────────────────────────────

impl<T: Scalar> DynVector<T> {
    /// Squared L2 norm (dot product with self).
    ///
    /// ```
    /// use numeris::DynVector;
    /// let v = DynVector::from_slice(&[3.0, 4.0]);
    /// assert_eq!(v.norm_squared(), 25.0);
    /// ```
    pub fn norm_squared(&self) -> T {
        self.dot(self)
    }
}

impl<T: LinalgScalar> DynVector<T> {
    /// L2 (Euclidean) norm.
    ///
    /// ```
    /// use numeris::DynVector;
    /// let v = DynVector::from_slice(&[3.0_f64, 4.0]);
    /// assert!((v.norm() - 5.0).abs() < 1e-12);
    /// ```
    pub fn norm(&self) -> T::Real {
        let mut sum = <T::Real as Zero>::zero();
        for i in 0..self.len() {
            sum = sum + self[i].modulus() * self[i].modulus();
        }
        sum.lsqrt()
    }

    /// L1 norm (sum of absolute values / moduli).
    ///
    /// ```
    /// use numeris::DynVector;
    /// let v = DynVector::from_slice(&[1.0_f64, -2.0, 3.0]);
    /// assert!((v.norm_l1() - 6.0).abs() < 1e-12);
    /// ```
    pub fn norm_l1(&self) -> T::Real {
        let mut sum = <T::Real as Zero>::zero();
        for i in 0..self.len() {
            sum = sum + self[i].modulus();
        }
        sum
    }

    /// Return a unit vector in the same direction.
    ///
    /// Panics if the norm is zero.
    ///
    /// ```
    /// use numeris::DynVector;
    /// let v = DynVector::from_slice(&[3.0_f64, 4.0]);
    /// let u = v.normalize();
    /// assert!((u.norm() - 1.0).abs() < 1e-12);
    /// assert!((u[0] - 0.6).abs() < 1e-12);
    /// ```
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        let inv = <T::Real as num_traits::One>::one() / n;
        let data = self
            .as_slice()
            .iter()
            .map(|&x| x * T::from_real(inv))
            .collect();
        DynVector {
            inner: DynMatrix {
                data,
                nrows: 1,
                ncols: self.len(),
            },
        }
    }
}

// ── Matrix norms ────────────────────────────────────────────────────

impl<T: Scalar> DynMatrix<T> {
    /// Squared Frobenius norm (sum of all elements squared).
    pub fn frobenius_norm_squared(&self) -> T {
        let mut sum = T::zero();
        for &x in &self.data {
            sum = sum + x * x;
        }
        sum
    }
}

impl<T: LinalgScalar> DynMatrix<T> {
    /// Frobenius norm (square root of sum of squared moduli).
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::from_rows(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]);
    /// assert!((m.frobenius_norm() - 30.0_f64.sqrt()).abs() < 1e-12);
    /// ```
    pub fn frobenius_norm(&self) -> T::Real {
        let mut sum = <T::Real as Zero>::zero();
        for &x in &self.data {
            let m = x.modulus();
            sum = sum + m * m;
        }
        sum.lsqrt()
    }

    /// Infinity norm (maximum row sum of moduli).
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::from_rows(2, 2, &[1.0_f64, -2.0, 3.0, 4.0]);
    /// assert!((m.norm_inf() - 7.0).abs() < 1e-12);
    /// ```
    pub fn norm_inf(&self) -> T::Real {
        let mut max = <T::Real as Zero>::zero();
        for i in 0..self.nrows {
            let mut row_sum = <T::Real as Zero>::zero();
            for j in 0..self.ncols {
                row_sum = row_sum + self[(i, j)].modulus();
            }
            if row_sum > max {
                max = row_sum;
            }
        }
        max
    }

    /// One norm (maximum column sum of moduli).
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::from_rows(2, 2, &[1.0_f64, -2.0, 3.0, 4.0]);
    /// assert!((m.norm_one() - 6.0).abs() < 1e-12);
    /// ```
    pub fn norm_one(&self) -> T::Real {
        let mut max = <T::Real as Zero>::zero();
        for j in 0..self.ncols {
            let mut col_sum = <T::Real as Zero>::zero();
            for i in 0..self.nrows {
                col_sum = col_sum + self[(i, j)].modulus();
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

    #[test]
    fn vector_norm_squared() {
        let v = DynVector::from_slice(&[3.0, 4.0]);
        assert_eq!(v.norm_squared(), 25.0);
    }

    #[test]
    fn vector_norm() {
        let v = DynVector::from_slice(&[3.0_f64, 4.0]);
        assert!((v.norm() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn vector_norm_l1() {
        let v = DynVector::from_slice(&[1.0_f64, -2.0, 3.0]);
        assert!((v.norm_l1() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn vector_normalize() {
        let v = DynVector::from_slice(&[3.0_f64, 4.0]);
        let u = v.normalize();
        assert!((u.norm() - 1.0).abs() < 1e-12);
        assert!((u[0] - 0.6).abs() < 1e-12);
        assert!((u[1] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn frobenius_norm() {
        let m = DynMatrix::from_rows(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]);
        assert!((m.frobenius_norm() - 30.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn frobenius_norm_squared_integer() {
        let m = DynMatrix::from_rows(2, 2, &[1, 2, 3, 4]);
        assert_eq!(m.frobenius_norm_squared(), 30);
    }

    #[test]
    fn norm_inf() {
        let m = DynMatrix::from_rows(2, 2, &[1.0_f64, -2.0, 3.0, 4.0]);
        assert!((m.norm_inf() - 7.0).abs() < 1e-12);
    }

    #[test]
    fn norm_one() {
        let m = DynMatrix::from_rows(2, 2, &[1.0_f64, -2.0, 3.0, 4.0]);
        assert!((m.norm_one() - 6.0).abs() < 1e-12);
    }
}
