use crate::Matrix;
use crate::matrix::vector::Vector;
use crate::traits::Scalar;

// ── General block extraction & insertion ────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Extract a P×Q sub-matrix starting at position `(i, j)`.
    ///
    /// Panics if the block extends beyond the matrix bounds.
    ///
    /// ```
    /// use numeris::Matrix;
    /// let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    /// let b: Matrix<f64, 2, 2> = m.block(1, 1);
    /// assert_eq!(b[(0, 0)], 5.0);
    /// assert_eq!(b[(1, 1)], 9.0);
    /// ```
    pub fn block<const P: usize, const Q: usize>(&self, i: usize, j: usize) -> Matrix<T, P, Q> {
        assert!(
            i + P <= M && j + Q <= N,
            "block ({i},{j}) size {P}×{Q} out of bounds for {M}×{N} matrix"
        );
        let mut out = Matrix::<T, P, Q>::zeros();
        for r in 0..P {
            for c in 0..Q {
                out[(r, c)] = self[(i + r, j + c)];
            }
        }
        out
    }

    /// Write a P×Q sub-matrix into self starting at position `(i, j)`.
    ///
    /// Panics if the block extends beyond the matrix bounds.
    ///
    /// ```
    /// use numeris::Matrix;
    /// let mut m: Matrix<f64, 3, 3> = Matrix::zeros();
    /// let patch = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// m.set_block(1, 1, &patch);
    /// assert_eq!(m[(1, 1)], 1.0);
    /// assert_eq!(m[(2, 2)], 4.0);
    /// ```
    pub fn set_block<const P: usize, const Q: usize>(
        &mut self,
        i: usize,
        j: usize,
        src: &Matrix<T, P, Q>,
    ) {
        assert!(
            i + P <= M && j + Q <= N,
            "set_block ({i},{j}) size {P}×{Q} out of bounds for {M}×{N} matrix"
        );
        for r in 0..P {
            for c in 0..Q {
                self[(i + r, j + c)] = src[(r, c)];
            }
        }
    }
}

// ── Corner blocks ───────────────────────────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Extract the top-left P×Q corner.
    pub fn top_left<const P: usize, const Q: usize>(&self) -> Matrix<T, P, Q> {
        self.block(0, 0)
    }

    /// Extract the top-right P×Q corner.
    pub fn top_right<const P: usize, const Q: usize>(&self) -> Matrix<T, P, Q> {
        self.block(0, N - Q)
    }

    /// Extract the bottom-left P×Q corner.
    pub fn bottom_left<const P: usize, const Q: usize>(&self) -> Matrix<T, P, Q> {
        self.block(M - P, 0)
    }

    /// Extract the bottom-right P×Q corner.
    pub fn bottom_right<const P: usize, const Q: usize>(&self) -> Matrix<T, P, Q> {
        self.block(M - P, N - Q)
    }
}

// ── Row / column spans ──────────────────────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Extract the first P rows.
    pub fn top_rows<const P: usize>(&self) -> Matrix<T, P, N> {
        self.block(0, 0)
    }

    /// Extract the last P rows.
    pub fn bottom_rows<const P: usize>(&self) -> Matrix<T, P, N> {
        self.block(M - P, 0)
    }

    /// Extract the first Q columns.
    pub fn left_cols<const Q: usize>(&self) -> Matrix<T, M, Q> {
        self.block(0, 0)
    }

    /// Extract the last Q columns.
    pub fn right_cols<const Q: usize>(&self) -> Matrix<T, M, Q> {
        self.block(0, N - Q)
    }

    /// Extract P rows starting at row i.
    pub fn middle_rows<const P: usize>(&self, i: usize) -> Matrix<T, P, N> {
        self.block(i, 0)
    }

    /// Extract Q columns starting at column j.
    pub fn middle_cols<const Q: usize>(&self, j: usize) -> Matrix<T, M, Q> {
        self.block(0, j)
    }
}

// ── Vector segment operations ───────────────────────────────────────

impl<T: Scalar, const N: usize> Vector<T, N> {
    /// Extract the first P elements.
    ///
    /// ```
    /// use numeris::Vector;
    /// let v = Vector::from_array([10, 20, 30, 40, 50]);
    /// let h: Vector<i32, 3> = v.head();
    /// assert_eq!(h[0], 10);
    /// assert_eq!(h[2], 30);
    /// ```
    pub fn head<const P: usize>(&self) -> Vector<T, P> {
        self.block(0, 0)
    }

    /// Extract the last P elements.
    ///
    /// ```
    /// use numeris::Vector;
    /// let v = Vector::from_array([10, 20, 30, 40, 50]);
    /// let t: Vector<i32, 2> = v.tail();
    /// assert_eq!(t[0], 40);
    /// assert_eq!(t[1], 50);
    /// ```
    pub fn tail<const P: usize>(&self) -> Vector<T, P> {
        self.block(0, N - P)
    }

    /// Extract P elements starting at index i.
    pub fn segment<const P: usize>(&self, i: usize) -> Vector<T, P> {
        self.block(0, i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat4x5() -> Matrix<i32, 4, 5> {
        Matrix::from_fn(|i, j| (i * 5 + j) as i32)
    }

    // ── block / set_block ───────────────────────────────────────

    #[test]
    fn block_extract() {
        let m = mat4x5();
        let b: Matrix<i32, 2, 3> = m.block(1, 1);
        // Row 1 cols 1..4: [6, 7, 8]
        // Row 2 cols 1..4: [11, 12, 13]
        assert_eq!(b[(0, 0)], 6);
        assert_eq!(b[(0, 2)], 8);
        assert_eq!(b[(1, 0)], 11);
        assert_eq!(b[(1, 2)], 13);
    }

    #[test]
    fn block_full_matrix() {
        let m = mat4x5();
        let full: Matrix<i32, 4, 5> = m.block(0, 0);
        assert_eq!(full, m);
    }

    #[test]
    fn block_single_element() {
        let m = mat4x5();
        let s: Matrix<i32, 1, 1> = m.block(2, 3);
        assert_eq!(s[(0, 0)], 13);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn block_out_of_bounds() {
        let m = mat4x5();
        let _: Matrix<i32, 2, 3> = m.block(3, 3);
    }

    #[test]
    fn set_block_basic() {
        let mut m: Matrix<i32, 4, 4> = Matrix::zeros();
        let patch = Matrix::new([[1, 2], [3, 4]]);
        m.set_block(1, 1, &patch);
        assert_eq!(m[(1, 1)], 1);
        assert_eq!(m[(1, 2)], 2);
        assert_eq!(m[(2, 1)], 3);
        assert_eq!(m[(2, 2)], 4);
        // Surrounding elements unchanged
        assert_eq!(m[(0, 0)], 0);
        assert_eq!(m[(3, 3)], 0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn set_block_out_of_bounds() {
        let mut m: Matrix<i32, 3, 3> = Matrix::zeros();
        let patch = Matrix::new([[1, 2], [3, 4]]);
        m.set_block(2, 2, &patch);
    }

    #[test]
    fn block_roundtrip() {
        let m = mat4x5();
        let b: Matrix<i32, 2, 3> = m.block(1, 2);
        let mut m2 = mat4x5();
        m2.set_block(1, 2, &b);
        assert_eq!(m, m2);
    }

    // ── Corner blocks ───────────────────────────────────────────

    #[test]
    fn top_left() {
        let m = mat4x5();
        let tl: Matrix<i32, 2, 2> = m.top_left();
        assert_eq!(tl[(0, 0)], 0);
        assert_eq!(tl[(0, 1)], 1);
        assert_eq!(tl[(1, 0)], 5);
        assert_eq!(tl[(1, 1)], 6);
    }

    #[test]
    fn top_right() {
        let m = mat4x5();
        let tr: Matrix<i32, 2, 2> = m.top_right();
        assert_eq!(tr[(0, 0)], 3);
        assert_eq!(tr[(0, 1)], 4);
        assert_eq!(tr[(1, 0)], 8);
        assert_eq!(tr[(1, 1)], 9);
    }

    #[test]
    fn bottom_left() {
        let m = mat4x5();
        let bl: Matrix<i32, 2, 3> = m.bottom_left();
        assert_eq!(bl[(0, 0)], 10);
        assert_eq!(bl[(1, 2)], 17);
    }

    #[test]
    fn bottom_right() {
        let m = mat4x5();
        let br: Matrix<i32, 2, 2> = m.bottom_right();
        assert_eq!(br[(0, 0)], 13);
        assert_eq!(br[(1, 1)], 19);
    }

    // ── Row / column spans ──────────────────────────────────────

    #[test]
    fn top_rows() {
        let m = mat4x5();
        let t: Matrix<i32, 2, 5> = m.top_rows();
        assert_eq!(t[(0, 0)], 0);
        assert_eq!(t[(1, 4)], 9);
    }

    #[test]
    fn bottom_rows() {
        let m = mat4x5();
        let b: Matrix<i32, 1, 5> = m.bottom_rows();
        assert_eq!(b[(0, 0)], 15);
        assert_eq!(b[(0, 4)], 19);
    }

    #[test]
    fn left_cols() {
        let m = mat4x5();
        let l: Matrix<i32, 4, 2> = m.left_cols();
        assert_eq!(l[(0, 0)], 0);
        assert_eq!(l[(3, 1)], 16);
    }

    #[test]
    fn right_cols() {
        let m = mat4x5();
        let r: Matrix<i32, 4, 3> = m.right_cols();
        assert_eq!(r[(0, 0)], 2);
        assert_eq!(r[(3, 2)], 19);
    }

    #[test]
    fn middle_rows() {
        let m = mat4x5();
        let mid: Matrix<i32, 2, 5> = m.middle_rows(1);
        assert_eq!(mid[(0, 0)], 5);
        assert_eq!(mid[(1, 4)], 14);
    }

    #[test]
    fn middle_cols() {
        let m = mat4x5();
        let mid: Matrix<i32, 4, 2> = m.middle_cols(2);
        assert_eq!(mid[(0, 0)], 2);
        assert_eq!(mid[(3, 1)], 18);
    }

    // ── Vector segments ─────────────────────────────────────────

    #[test]
    fn head() {
        let v = Vector::from_array([10, 20, 30, 40, 50]);
        let h: Vector<i32, 3> = v.head();
        assert_eq!(h[0], 10);
        assert_eq!(h[1], 20);
        assert_eq!(h[2], 30);
    }

    #[test]
    fn tail() {
        let v = Vector::from_array([10, 20, 30, 40, 50]);
        let t: Vector<i32, 2> = v.tail();
        assert_eq!(t[0], 40);
        assert_eq!(t[1], 50);
    }

    #[test]
    fn segment() {
        let v = Vector::from_array([10, 20, 30, 40, 50]);
        let s: Vector<i32, 3> = v.segment(1);
        assert_eq!(s[0], 20);
        assert_eq!(s[1], 30);
        assert_eq!(s[2], 40);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn segment_out_of_bounds() {
        let v = Vector::from_array([10, 20, 30]);
        let _: Vector<i32, 2> = v.segment(2);
    }

    // ── Float tests ─────────────────────────────────────────────

    #[test]
    fn block_f64() {
        let m = Matrix::new([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);
        let b: Matrix<f64, 2, 2> = m.block(1, 1);
        assert_eq!(b[(0, 0)], 5.0);
        assert_eq!(b[(0, 1)], 6.0);
        assert_eq!(b[(1, 0)], 8.0);
        assert_eq!(b[(1, 1)], 9.0);
    }

    #[test]
    fn set_block_identity_corner() {
        let mut m: Matrix<f64, 4, 4> = Matrix::zeros();
        let eye2: Matrix<f64, 2, 2> = Matrix::eye();
        m.set_block(0, 0, &eye2);
        m.set_block(2, 2, &eye2);
        assert_eq!(m, Matrix::new([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]));
    }
}
