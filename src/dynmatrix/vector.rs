use alloc::vec::Vec;
use core::ops::{Index, IndexMut};

use crate::matrix::vector::Vector;
use crate::traits::{MatrixMut, MatrixRef, Scalar};

use super::DynMatrix;

/// Dynamically-sized column vector (wraps an N×1 `DynMatrix`).
///
/// Matches the fixed-size `Vector<T, N>` = `Matrix<T, N, 1>` convention.
/// Provides single-index access `v[i]`.
///
/// # Examples
///
/// ```
/// use numeris::{DynVector, MatrixRef};
///
/// let v = DynVector::from_slice(&[1.0_f64, 2.0, 3.0]);
/// assert_eq!(v[0], 1.0);
/// assert_eq!(v.len(), 3);
/// assert_eq!(v.nrows(), 3);
/// assert_eq!(v.ncols(), 1);
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
            inner: DynMatrix::from_slice(data.len(), 1, data),
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
            inner: DynMatrix::from_vec(n, 1, data),
        }
    }

    /// Create a zero vector of length `n`.
    ///
    /// ```
    /// use numeris::DynVector;
    /// let v = DynVector::<f64>::zeros(4);
    /// assert_eq!(v.len(), 4);
    /// assert_eq!(v[3], 0.0);
    /// ```
    pub fn zeros(n: usize) -> Self {
        Self {
            inner: DynMatrix::zeros(n, 1),
        }
    }

    /// Create a vector filled with a value.
    pub fn fill(n: usize, value: T) -> Self {
        Self {
            inner: DynMatrix::fill(n, 1, value),
        }
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.nrows()
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
        crate::simd::dot_dispatch(self.as_slice(), rhs.as_slice())
    }

    /// Cast every element to a different numeric type.
    ///
    /// ```
    /// use numeris::DynVector;
    /// let v = DynVector::from_slice(&[1.0_f64, 2.0, 3.0]);
    /// let v32: DynVector<f32> = v.cast();
    /// assert_eq!(v32[0], 1.0_f32);
    /// ```
    pub fn cast<U: Scalar + num_traits::NumCast>(&self) -> DynVector<U>
    where
        T: num_traits::ToPrimitive,
    {
        DynVector {
            inner: self.inner.cast(),
        }
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

    /// Iterate over elements.
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Iterate mutably over elements.
    #[inline]
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }
}

impl<T: Scalar> DynVector<T> {
    /// Create a vector from a function `f(index) -> value`.
    ///
    /// ```
    /// use numeris::DynVector;
    /// let v = DynVector::from_fn(4, |i| (i * i) as f64);
    /// assert_eq!(v[0], 0.0);
    /// assert_eq!(v[3], 9.0);
    /// ```
    pub fn from_fn(n: usize, f: impl Fn(usize) -> T) -> Self {
        let data: Vec<T> = (0..n).map(f).collect();
        Self::from_vec(data)
    }
}

// ── IntoIterator ────────────────────────────────────────────────────

impl<'a, T: Scalar> IntoIterator for &'a DynVector<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: Scalar> IntoIterator for &'a mut DynVector<T> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// ── Index ───────────────────────────────────────────────────────────

impl<T> Index<usize> for DynVector<T> {
    type Output = T;

    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.inner[(i, 0)]
    }
}

impl<T> IndexMut<usize> for DynVector<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.inner[(i, 0)]
    }
}

// ── MatrixRef / MatrixMut ───────────────────────────────────────────

impl<T> MatrixRef<T> for DynVector<T> {
    #[inline]
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    #[inline]
    fn ncols(&self) -> usize {
        1
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> &T {
        self.inner.get(row, col)
    }

    #[inline]
    fn col_as_slice(&self, col: usize, row_start: usize) -> &[T] {
        self.inner.col_as_slice(col, row_start)
    }
}

impl<T> MatrixMut<T> for DynVector<T> {
    #[inline]
    fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        self.inner.get_mut(row, col)
    }

    #[inline]
    fn col_as_mut_slice(&mut self, col: usize, row_start: usize) -> &mut [T] {
        self.inner.col_as_mut_slice(col, row_start)
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
        Self::from_slice(v.as_slice())
    }
}

impl<T: Scalar, const N: usize> From<&Vector<T, N>> for DynVector<T> {
    fn from(v: &Vector<T, N>) -> Self {
        Self::from_slice(v.as_slice())
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
        let v = DynVector::<f64>::zeros(4);
        assert_eq!(v.len(), 4);
        for i in 0..4 {
            assert_eq!(v[i], 0.0);
        }
    }

    #[test]
    fn index_mut() {
        let mut v = DynVector::<f64>::zeros(3);
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
    fn dimensions_match_vector() {
        // DynVector should be N×1 like Vector
        let v = DynVector::from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(v.nrows(), 3);
        assert_eq!(v.ncols(), 1);
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
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 1);
        assert_eq!(m[(1, 0)], 2.0);
    }
}
