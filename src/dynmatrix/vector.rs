use alloc::vec::Vec;
use core::ops::{Index, IndexMut};

use crate::matrix::vector::Vector;
use crate::traits::{MatrixMut, MatrixRef, Scalar};

use super::DynMatrix;

/// Dynamically-sized vector (wraps a 1×N `DynMatrix`).
///
/// Enforces single-row constraint and provides single-index access `v[i]`.
///
/// # Examples
///
/// ```
/// use numeris::DynVector;
///
/// let v = DynVector::from_slice(&[1.0_f64, 2.0, 3.0]);
/// assert_eq!(v[0], 1.0);
/// assert_eq!(v.len(), 3);
/// assert!((v.dot(&v) - 14.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DynVector<T> {
    pub(crate) inner: DynMatrix<T>,
}

impl<T: Scalar> DynVector<T> {
    /// Create a vector from a flat slice.
    ///
    /// ```
    /// use numeris::DynVector;
    /// let v = DynVector::from_slice(&[1.0, 2.0, 3.0]);
    /// assert_eq!(v[0], 1.0);
    /// assert_eq!(v.len(), 3);
    /// ```
    pub fn from_slice(data: &[T]) -> Self {
        Self {
            inner: DynMatrix::from_slice(1, data.len(), data),
        }
    }

    /// Create a vector from an owned `Vec`.
    ///
    /// ```
    /// use numeris::DynVector;
    /// let v = DynVector::from_vec(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(v[2], 3.0);
    /// ```
    pub fn from_vec(data: Vec<T>) -> Self {
        let n = data.len();
        Self {
            inner: DynMatrix::from_vec(1, n, data),
        }
    }

    /// Create a zero vector of length `n`.
    ///
    /// ```
    /// use numeris::DynVector;
    /// let v = DynVector::zeros(4, 0.0_f64);
    /// assert_eq!(v.len(), 4);
    /// assert_eq!(v[3], 0.0);
    /// ```
    pub fn zeros(n: usize, _zero: T) -> Self {
        Self {
            inner: DynMatrix::zeros(1, n, T::zero()),
        }
    }

    /// Create a vector filled with a value.
    pub fn fill(n: usize, value: T) -> Self {
        Self {
            inner: DynMatrix::fill(1, n, value),
        }
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.ncols()
    }

    /// Whether the vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Dot product.
    ///
    /// ```
    /// use numeris::DynVector;
    /// let a = DynVector::from_slice(&[1.0, 2.0, 3.0]);
    /// let b = DynVector::from_slice(&[4.0, 5.0, 6.0]);
    /// assert_eq!(a.dot(&b), 32.0);
    /// ```
    pub fn dot(&self, rhs: &Self) -> T {
        assert_eq!(self.len(), rhs.len(), "vector length mismatch");
        let mut sum = T::zero();
        for i in 0..self.len() {
            sum = sum + self[i] * rhs[i];
        }
        sum
    }

    /// View the vector data as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    /// View the vector data as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }
}

// ── Index ───────────────────────────────────────────────────────────

impl<T> Index<usize> for DynVector<T> {
    type Output = T;

    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.inner[(0, i)]
    }
}

impl<T> IndexMut<usize> for DynVector<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.inner[(0, i)]
    }
}

// ── MatrixRef / MatrixMut ───────────────────────────────────────────

impl<T> MatrixRef<T> for DynVector<T> {
    #[inline]
    fn nrows(&self) -> usize {
        1
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> &T {
        self.inner.get(row, col)
    }
}

impl<T> MatrixMut<T> for DynVector<T> {
    #[inline]
    fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        self.inner.get_mut(row, col)
    }
}

// ── Conversions: Vector ↔ DynVector ─────────────────────────────────

impl<T: Scalar, const N: usize> From<Vector<T, N>> for DynVector<T> {
    /// Convert a fixed-size `Vector` into a `DynVector`.
    ///
    /// ```
    /// use numeris::{Vector, DynVector};
    /// let v = Vector::from_array([1.0, 2.0, 3.0]);
    /// let dv: DynVector<f64> = v.into();
    /// assert_eq!(dv.len(), 3);
    /// assert_eq!(dv[0], 1.0);
    /// ```
    fn from(v: Vector<T, N>) -> Self {
        Self {
            inner: DynMatrix::from(v),
        }
    }
}

impl<T: Scalar, const N: usize> From<&Vector<T, N>> for DynVector<T> {
    fn from(v: &Vector<T, N>) -> Self {
        Self {
            inner: DynMatrix::from(v),
        }
    }
}

impl<T: Scalar> From<DynVector<T>> for DynMatrix<T> {
    fn from(v: DynVector<T>) -> Self {
        v.inner
    }
}

impl<T: Scalar> From<&DynVector<T>> for DynMatrix<T> {
    fn from(v: &DynVector<T>) -> Self {
        v.inner.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector;

    #[test]
    fn from_slice() {
        let v = DynVector::from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn from_vec() {
        let v = DynVector::from_vec(vec![10.0, 20.0]);
        assert_eq!(v.len(), 2);
        assert_eq!(v[1], 20.0);
    }

    #[test]
    fn zeros() {
        let v = DynVector::zeros(4, 0.0_f64);
        assert_eq!(v.len(), 4);
        for i in 0..4 {
            assert_eq!(v[i], 0.0);
        }
    }

    #[test]
    fn index_mut() {
        let mut v = DynVector::zeros(3, 0.0_f64);
        v[1] = 42.0;
        assert_eq!(v[1], 42.0);
    }

    #[test]
    fn dot_product() {
        let a = DynVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = DynVector::from_slice(&[4.0, 5.0, 6.0]);
        assert_eq!(a.dot(&b), 32.0);
    }

    #[test]
    fn from_fixed_vector() {
        let v = Vector::from_array([1.0, 2.0, 3.0]);
        let dv: DynVector<f64> = v.into();
        assert_eq!(dv.len(), 3);
        assert_eq!(dv[0], 1.0);
        assert_eq!(dv[2], 3.0);
    }

    #[test]
    fn matrix_ref_trait() {
        let v = DynVector::from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(v.nrows(), 1);
        assert_eq!(v.ncols(), 3);
        assert_eq!(*v.get(0, 1), 2.0);
    }

    #[test]
    fn as_slice() {
        let v = DynVector::from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn into_dynmatrix() {
        let v = DynVector::from_slice(&[1.0, 2.0, 3.0]);
        let m: DynMatrix<f64> = v.into();
        assert_eq!(m.nrows(), 1);
        assert_eq!(m.ncols(), 3);
        assert_eq!(m[(0, 1)], 2.0);
    }
}
