use super::DynMatrix;

impl<T> DynMatrix<T> {
    /// View the entire matrix as a flat slice in row-major order.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(m.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    /// ```
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// View the entire matrix as a mutable flat slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// View row `i` as a slice.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(m.row_slice(0), &[1.0, 2.0, 3.0]);
    /// assert_eq!(m.row_slice(1), &[4.0, 5.0, 6.0]);
    /// ```
    #[inline]
    pub fn row_slice(&self, i: usize) -> &[T] {
        let start = i * self.ncols;
        &self.data[start..start + self.ncols]
    }

    /// View row `i` as a mutable slice.
    #[inline]
    pub fn row_slice_mut(&mut self, i: usize) -> &mut [T] {
        let start = i * self.ncols;
        let end = start + self.ncols;
        &mut self.data[start..end]
    }

    /// Iterate over all elements in row-major order.
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Iterate mutably over all elements in row-major order.
    #[inline]
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }
}

impl<'a, T> IntoIterator for &'a DynMatrix<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut DynMatrix<T> {
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

    #[test]
    fn as_slice() {
        let m = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn as_mut_slice() {
        let mut m = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        m.as_mut_slice()[0] = 99.0;
        assert_eq!(m[(0, 0)], 99.0);
    }

    #[test]
    fn row_slice() {
        let m = DynMatrix::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.row_slice(0), &[1.0, 2.0, 3.0]);
        assert_eq!(m.row_slice(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn iter() {
        let m = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let sum: f64 = m.iter().sum();
        assert_eq!(sum, 10.0);
    }

    #[test]
    fn iter_mut() {
        let mut m = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        for x in m.iter_mut() {
            *x *= 2.0;
        }
        assert_eq!(m[(0, 0)], 2.0);
        assert_eq!(m[(1, 1)], 8.0);
    }

    #[test]
    fn into_iter_ref() {
        let m = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let sum: f64 = (&m).into_iter().sum();
        assert_eq!(sum, 10.0);
    }

    #[test]
    fn into_iter_for_loop() {
        let m = DynMatrix::from_slice(2, 2, &[1, 2, 3, 4]);
        let mut sum = 0;
        for &x in &m {
            sum += x;
        }
        assert_eq!(sum, 10);
    }
}
