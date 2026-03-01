use crate::traits::Scalar;

use super::DynMatrix;

impl<T: Scalar> DynMatrix<T> {
    /// Extract a sub-matrix of size `rows x cols` starting at `(i, j)`.
    ///
    /// Panics if the block extends beyond the matrix bounds.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::from_fn(3, 3, |i, j| (i * 3 + j) as f64);
    /// let b = m.block(1, 1, 2, 2);
    /// assert_eq!(b[(0, 0)], 4.0);
    /// assert_eq!(b[(1, 1)], 8.0);
    /// ```
    pub fn block(&self, i: usize, j: usize, rows: usize, cols: usize) -> Self {
        assert!(
            i + rows <= self.nrows && j + cols <= self.ncols,
            "block ({},{}) size {}x{} out of bounds for {}x{} matrix",
            i, j, rows, cols, self.nrows, self.ncols,
        );
        DynMatrix::from_fn(rows, cols, |r, c| self[(i + r, j + c)])
    }

    /// Write a sub-matrix into self starting at position `(i, j)`.
    ///
    /// Panics if the block extends beyond the matrix bounds.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let mut m = DynMatrix::zeros(3, 3, 0.0_f64);
    /// let patch = DynMatrix::from_rows(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    /// m.set_block(1, 1, &patch);
    /// assert_eq!(m[(1, 1)], 1.0);
    /// assert_eq!(m[(2, 2)], 4.0);
    /// ```
    pub fn set_block(&mut self, i: usize, j: usize, src: &DynMatrix<T>) {
        assert!(
            i + src.nrows <= self.nrows && j + src.ncols <= self.ncols,
            "set_block ({},{}) size {}x{} out of bounds for {}x{} matrix",
            i, j, src.nrows, src.ncols, self.nrows, self.ncols,
        );
        for r in 0..src.nrows {
            for c in 0..src.ncols {
                self[(i + r, j + c)] = src[(r, c)];
            }
        }
    }

    /// Extract the top-left corner of size `rows x cols`.
    pub fn top_left(&self, rows: usize, cols: usize) -> Self {
        self.block(0, 0, rows, cols)
    }

    /// Extract the top-right corner of size `rows x cols`.
    pub fn top_right(&self, rows: usize, cols: usize) -> Self {
        self.block(0, self.ncols - cols, rows, cols)
    }

    /// Extract the bottom-left corner of size `rows x cols`.
    pub fn bottom_left(&self, rows: usize, cols: usize) -> Self {
        self.block(self.nrows - rows, 0, rows, cols)
    }

    /// Extract the bottom-right corner of size `rows x cols`.
    pub fn bottom_right(&self, rows: usize, cols: usize) -> Self {
        self.block(self.nrows - rows, self.ncols - cols, rows, cols)
    }

    /// Extract the first `n` rows.
    pub fn top_rows(&self, n: usize) -> Self {
        self.block(0, 0, n, self.ncols)
    }

    /// Extract the last `n` rows.
    pub fn bottom_rows(&self, n: usize) -> Self {
        self.block(self.nrows - n, 0, n, self.ncols)
    }

    /// Extract the first `n` columns.
    pub fn left_cols(&self, n: usize) -> Self {
        self.block(0, 0, self.nrows, n)
    }

    /// Extract the last `n` columns.
    pub fn right_cols(&self, n: usize) -> Self {
        self.block(0, self.ncols - n, self.nrows, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat4x5() -> DynMatrix<i32> {
        DynMatrix::from_fn(4, 5, |i, j| (i * 5 + j) as i32)
    }

    #[test]
    fn block_extract() {
        let m = mat4x5();
        let b = m.block(1, 1, 2, 3);
        assert_eq!(b[(0, 0)], 6);
        assert_eq!(b[(0, 2)], 8);
        assert_eq!(b[(1, 0)], 11);
        assert_eq!(b[(1, 2)], 13);
    }

    #[test]
    fn block_full() {
        let m = mat4x5();
        let full = m.block(0, 0, 4, 5);
        assert_eq!(full, m);
    }

    #[test]
    fn block_single() {
        let m = mat4x5();
        let s = m.block(2, 3, 1, 1);
        assert_eq!(s[(0, 0)], 13);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn block_out_of_bounds() {
        let m = mat4x5();
        let _ = m.block(3, 3, 2, 3);
    }

    #[test]
    fn set_block_basic() {
        let mut m = DynMatrix::zeros(4, 4, 0i32);
        let patch = DynMatrix::from_rows(2, 2, &[1, 2, 3, 4]);
        m.set_block(1, 1, &patch);
        assert_eq!(m[(1, 1)], 1);
        assert_eq!(m[(2, 2)], 4);
        assert_eq!(m[(0, 0)], 0);
    }

    #[test]
    fn block_roundtrip() {
        let m = mat4x5();
        let b = m.block(1, 2, 2, 3);
        let mut m2 = mat4x5();
        m2.set_block(1, 2, &b);
        assert_eq!(m, m2);
    }

    #[test]
    fn corners() {
        let m = mat4x5();
        let tl = m.top_left(2, 2);
        assert_eq!(tl[(0, 0)], 0);
        assert_eq!(tl[(1, 1)], 6);

        let tr = m.top_right(2, 2);
        assert_eq!(tr[(0, 0)], 3);
        assert_eq!(tr[(1, 1)], 9);

        let bl = m.bottom_left(2, 3);
        assert_eq!(bl[(0, 0)], 10);
        assert_eq!(bl[(1, 2)], 17);

        let br = m.bottom_right(2, 2);
        assert_eq!(br[(0, 0)], 13);
        assert_eq!(br[(1, 1)], 19);
    }

    #[test]
    fn row_col_spans() {
        let m = mat4x5();
        let top = m.top_rows(2);
        assert_eq!(top[(0, 0)], 0);
        assert_eq!(top[(1, 4)], 9);

        let bot = m.bottom_rows(1);
        assert_eq!(bot[(0, 0)], 15);

        let left = m.left_cols(2);
        assert_eq!(left[(0, 0)], 0);
        assert_eq!(left[(3, 1)], 16);

        let right = m.right_cols(3);
        assert_eq!(right[(0, 0)], 2);
        assert_eq!(right[(3, 2)], 19);
    }
}
