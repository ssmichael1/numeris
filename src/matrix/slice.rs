use crate::traits::Scalar;
use crate::Matrix;

// ── Slice access ────────────────────────────────────────────────────

impl<T, const M: usize, const N: usize> Matrix<T, M, N> {
    /// View the entire matrix as a flat slice in column-major order.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data.as_flattened()
    }

    /// View the entire matrix as a mutable flat slice in column-major order.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_flattened_mut()
    }

    /// View column `j` as a slice. Zero-cost — columns are contiguous in memory.
    #[inline]
    pub fn col_slice(&self, j: usize) -> &[T] {
        &self.data[j]
    }

    /// View column `j` as a mutable slice.
    #[inline]
    pub fn col_slice_mut(&mut self, j: usize) -> &mut [T] {
        &mut self.data[j]
    }
}

impl<T: Scalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Create a matrix from a flat slice in column-major order.
    ///
    /// Panics if `slice.len() != M * N`.
    ///
    /// ```
    /// use numeris::Matrix;
    /// // Column-major: col0=[1,4], col1=[2,5], col2=[3,6]
    /// let m: Matrix<f64, 2, 3> = Matrix::from_slice(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    /// assert_eq!(m[(0, 0)], 1.0);
    /// assert_eq!(m[(1, 0)], 4.0);
    /// assert_eq!(m[(0, 2)], 3.0);
    /// ```
    pub fn from_slice(slice: &[T]) -> Self {
        assert_eq!(
            slice.len(),
            M * N,
            "slice length {} does not match {}x{} matrix",
            slice.len(),
            M,
            N
        );
        let mut m = Self::zeros();
        for j in 0..N {
            for i in 0..M {
                m[(i, j)] = slice[j * M + i];
            }
        }
        m
    }
}

// ── Iterators ───────────────────────────────────────────────────────

impl<T, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Iterate over all elements in column-major order.
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Iterate mutably over all elements in column-major order.
    #[inline]
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }

    /// Iterate over columns, each as an array `&[T; M]`.
    #[inline]
    pub fn iter_cols(&self) -> impl Iterator<Item = &[T; M]> {
        self.data.iter()
    }

    /// Iterate mutably over columns, each as `&mut [T; M]`.
    #[inline]
    pub fn iter_cols_mut(&mut self) -> impl Iterator<Item = &mut [T; M]> {
        self.data.iter_mut()
    }
}

impl<'a, T, const M: usize, const N: usize> IntoIterator for &'a Matrix<T, M, N> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const M: usize, const N: usize> IntoIterator for &'a mut Matrix<T, M, N> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::vector::Vector;

    #[test]
    fn as_slice_col_major() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let s = m.as_slice();
        // Column-major: col0=[1,3], col1=[2,4]
        assert_eq!(s, &[1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn as_mut_slice() {
        let mut m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        // Column-major: [0] is (0,0)
        m.as_mut_slice()[0] = 99.0;
        assert_eq!(m[(0, 0)], 99.0);
    }

    #[test]
    fn col_slice() {
        let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(m.col_slice(0), &[1.0, 4.0]);
        assert_eq!(m.col_slice(1), &[2.0, 5.0]);
        assert_eq!(m.col_slice(2), &[3.0, 6.0]);
    }

    #[test]
    fn col_slice_mut() {
        let mut m: Matrix<f64, 2, 3> = Matrix::zeros();
        m.col_slice_mut(1).copy_from_slice(&[7.0, 8.0]);
        assert_eq!(m[(0, 1)], 7.0);
        assert_eq!(m[(1, 1)], 8.0);
    }

    #[test]
    fn from_slice_col_major() {
        // Column-major: col0=[1,4], col1=[2,5], col2=[3,6]
        let m: Matrix<f64, 2, 3> = Matrix::from_slice(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(0, 2)], 3.0);
        assert_eq!(m[(1, 2)], 6.0);
    }

    #[test]
    #[should_panic]
    fn from_slice_wrong_length() {
        let _: Matrix<f64, 2, 2> = Matrix::from_slice(&[1.0, 2.0, 3.0]);
    }

    #[test]
    fn iter_elements() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let sum: f64 = m.iter().sum();
        assert_eq!(sum, 10.0);
    }

    #[test]
    fn iter_mut_elements() {
        let mut m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        for x in m.iter_mut() {
            *x *= 2.0;
        }
        assert_eq!(m[(0, 0)], 2.0);
        assert_eq!(m[(1, 1)], 8.0);
    }

    #[test]
    fn iter_cols() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let cols: Vec<&[f64; 2]> = m.iter_cols().collect();
        // Column 0 = [1, 3], Column 1 = [2, 4]
        assert_eq!(cols[0], &[1.0, 3.0]);
        assert_eq!(cols[1], &[2.0, 4.0]);
    }

    #[test]
    fn into_iter_ref() {
        let m = Matrix::new([[1, 2], [3, 4]]);
        let sum: i32 = (&m).into_iter().sum();
        assert_eq!(sum, 10);
    }

    #[test]
    fn into_iter_for_loop() {
        let m = Matrix::new([[1, 2], [3, 4]]);
        let mut sum = 0;
        for &x in &m {
            sum += x;
        }
        assert_eq!(sum, 10);
    }

    #[test]
    fn into_iter_mut_for_loop() {
        let mut m = Matrix::new([[1, 2], [3, 4]]);
        for x in &mut m {
            *x += 10;
        }
        assert_eq!(m[(0, 0)], 11);
        assert_eq!(m[(1, 1)], 14);
    }

    #[test]
    fn vector_as_slice() {
        let v = Vector::from_array([1.0, 2.0, 3.0]);
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn vector_iter() {
        let v = Vector::from_array([1.0, 2.0, 3.0]);
        let sum: f64 = v.iter().sum();
        assert_eq!(sum, 6.0);
    }
}
