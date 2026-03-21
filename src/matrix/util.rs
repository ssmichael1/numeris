use core::fmt::{self, Write as _};

use crate::Matrix;
use crate::matrix::vector::Vector;
use crate::traits::{FloatScalar, Scalar};

// ── Constructors ────────────────────────────────────────────────────

impl<T, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Create a matrix by calling `f(row, col)` for each element.
    ///
    /// ```
    /// use numeris::Matrix;
    /// let m: Matrix<f64, 3, 3> = Matrix::from_fn(|i, j| {
    ///     if i == j { 1.0 } else { 0.0 }
    /// });
    /// assert_eq!(m, Matrix::eye());
    /// ```
    pub fn from_fn(f: impl Fn(usize, usize) -> T) -> Self
    where
        T: Copy + Default,
    {
        let mut data = [[T::default(); M]; N];
        for j in 0..N {
            for i in 0..M {
                data[j][i] = f(i, j);
            }
        }
        Self { data }
    }

    /// Cast every element to a different numeric type.
    ///
    /// Uses `NumCast` from `num-traits` for safe numeric conversion.
    /// Panics if any element cannot be represented in the target type.
    ///
    /// ```
    /// use numeris::Matrix;
    /// let m = Matrix::new([[1.0_f64, 2.0], [3.0, 4.0]]);
    /// let m32: Matrix<f32, 2, 2> = m.cast();
    /// assert_eq!(m32[(0, 0)], 1.0_f32);
    ///
    /// let mi = Matrix::new([[1_i32, 2], [3, 4]]);
    /// let mf: Matrix<f64, 2, 2> = mi.cast();
    /// assert_eq!(mf[(1, 1)], 4.0);
    /// ```
    pub fn cast<U: Copy + Default + num_traits::NumCast>(&self) -> Matrix<U, M, N>
    where
        T: Copy + num_traits::ToPrimitive,
    {
        let mut data = [[U::default(); M]; N];
        for j in 0..N {
            for i in 0..M {
                data[j][i] = U::from(self[(i, j)]).expect("numeric cast failed");
            }
        }
        Matrix { data }
    }

    /// Apply a function to every element, producing a new matrix.
    ///
    /// ```
    /// use numeris::Matrix;
    /// let m = Matrix::new([[1.0_f64, 4.0], [9.0, 16.0]]);
    /// let r = m.map(|x: f64| x.sqrt());
    /// assert_eq!(r[(0, 0)], 1.0);
    /// assert_eq!(r[(1, 1)], 4.0);
    /// ```
    pub fn map<U: Copy + Default>(&self, f: impl Fn(T) -> U) -> Matrix<U, M, N>
    where
        T: Copy,
    {
        let mut data = [[U::default(); M]; N];
        for j in 0..N {
            for i in 0..M {
                data[j][i] = f(self[(i, j)]);
            }
        }
        Matrix { data }
    }
}

// ── Aggregation ─────────────────────────────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Sum of all elements.
    ///
    /// ```
    /// use numeris::Matrix;
    /// let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(m.sum(), 10.0);
    /// ```
    pub fn sum(&self) -> T {
        let mut s = T::zero();
        for j in 0..N {
            for i in 0..M {
                s = s + self[(i, j)];
            }
        }
        s
    }
}

// ── Element-wise operations ─────────────────────────────────────────

impl<T: FloatScalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Element-wise absolute value.
    ///
    /// ```
    /// use numeris::Matrix;
    /// let m = Matrix::new([[1.0_f64, -2.0], [-3.0, 4.0]]);
    /// let a = m.abs();
    /// assert_eq!(a[(0, 1)], 2.0);
    /// assert_eq!(a[(1, 0)], 3.0);
    /// ```
    pub fn abs(&self) -> Self {
        let mut out = *self;
        for j in 0..N {
            for i in 0..M {
                out[(i, j)] = self[(i, j)].abs();
            }
        }
        out
    }
}

// ── Element-wise max ────────────────────────────────────────────────

impl<T: FloatScalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Element-wise maximum: `c[i][j] = max(a[i][j], b[i][j])`.
    ///
    /// ```
    /// use numeris::Matrix;
    /// let a = Matrix::new([[1.0_f64, 5.0], [3.0, 2.0]]);
    /// let b = Matrix::new([[4.0, 2.0], [1.0, 6.0]]);
    /// let c = a.element_max(&b);
    /// assert_eq!(c[(0, 0)], 4.0);
    /// assert_eq!(c[(0, 1)], 5.0);
    /// assert_eq!(c[(1, 0)], 3.0);
    /// assert_eq!(c[(1, 1)], 6.0);
    /// ```
    pub fn element_max(&self, rhs: &Self) -> Self {
        let mut out = *self;
        for j in 0..N {
            for i in 0..M {
                if rhs[(i, j)] > self[(i, j)] {
                    out[(i, j)] = rhs[(i, j)];
                }
            }
        }
        out
    }
}

// ── Row / Column manipulation ───────────────────────────────────────

impl<T: Copy, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Swap two rows in place.
    ///
    /// ```
    /// use numeris::Matrix;
    /// let mut m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// m.swap_rows(0, 1);
    /// assert_eq!(m[(0, 0)], 3.0);
    /// assert_eq!(m[(1, 0)], 1.0);
    /// ```
    pub fn swap_rows(&mut self, a: usize, b: usize) {
        if a != b {
            for j in 0..N {
                let tmp = self.data[j][a];
                self.data[j][a] = self.data[j][b];
                self.data[j][b] = tmp;
            }
        }
    }
}

impl<T, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Swap two columns in place.
    ///
    /// ```
    /// use numeris::Matrix;
    /// let mut m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// m.swap_cols(0, 1);
    /// assert_eq!(m[(0, 0)], 2.0);
    /// assert_eq!(m[(0, 1)], 1.0);
    /// ```
    pub fn swap_cols(&mut self, a: usize, b: usize) {
        if a != b {
            self.data.swap(a, b);
        }
    }
}

// ── Row / Column access ─────────────────────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Extract row `i` as a row vector.
    ///
    /// ```
    /// use numeris::Matrix;
    /// let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let r = m.row(0);
    /// assert_eq!(r[0], 1.0);
    /// assert_eq!(r[1], 2.0);
    /// ```
    pub fn row(&self, i: usize) -> Vector<T, N> {
        let mut v = Vector::zeros();
        for j in 0..N {
            v[j] = self[(i, j)];
        }
        v
    }

    /// Set row `i` from a row vector.
    pub fn set_row(&mut self, i: usize, v: &Vector<T, N>) {
        for j in 0..N {
            self[(i, j)] = v[j];
        }
    }

    /// Extract column `j` as a row vector.
    ///
    /// ```
    /// use numeris::Matrix;
    /// let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let c = m.col(1);
    /// assert_eq!(c[0], 2.0);
    /// assert_eq!(c[1], 4.0);
    /// ```
    pub fn col(&self, j: usize) -> Vector<T, M> {
        let mut v = Vector::zeros();
        for i in 0..M {
            v[i] = self[(i, j)];
        }
        v
    }

    /// Set column `j` from a row vector.
    pub fn set_col(&mut self, j: usize, v: &Vector<T, M>) {
        for i in 0..M {
            self[(i, j)] = v[i];
        }
    }
}

// ── Display ─────────────────────────────────────────────────────────

impl<T: fmt::Display, const M: usize, const N: usize> fmt::Display for Matrix<T, M, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prec = f.precision();

        // Format each element once (respecting precision), find max width per column
        // Use a flat array to avoid allocation (M*N elements).
        // For very large matrices this is stack-heavy, but Matrix is const-generic
        // and typically small (≤ 6×6).
        let mut widths = [0usize; N];
        for j in 0..N {
            for i in 0..M {
                let w = WriteCounting::count(|wc| {
                    if let Some(p) = prec {
                        write!(wc, "{:.prec$}", self[(i, j)], prec = p)
                    } else {
                        write!(wc, "{}", self[(i, j)])
                    }
                });
                if w > widths[j] {
                    widths[j] = w;
                }
            }
        }

        // Special case: column vector (N×1) — compact bracket format
        if N == 1 {
            write!(f, "[")?;
            for i in 0..M {
                if i > 0 {
                    write!(f, ", ")?;
                }
                if let Some(p) = prec {
                    write!(f, "{:.prec$}", self[(i, 0)], prec = p)?;
                } else {
                    write!(f, "{}", self[(i, 0)])?;
                }
            }
            write!(f, "]")?;
            return Ok(());
        }

        for i in 0..M {
            write!(f, "│ ")?;
            for j in 0..N {
                if j > 0 {
                    write!(f, "  ")?;
                }
                if let Some(p) = prec {
                    write!(f, "{:>width$.prec$}", self[(i, j)], width = widths[j], prec = p)?;
                } else {
                    write!(f, "{:>width$}", self[(i, j)], width = widths[j])?;
                }
            }
            write!(f, " │")?;
            if i < M - 1 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

/// Helper to count characters written, without allocating.
struct WriteCounting {
    count: usize,
}

impl WriteCounting {
    fn count(f: impl FnOnce(&mut Self) -> fmt::Result) -> usize {
        let mut wc = WriteCounting { count: 0 };
        let _ = f(&mut wc);
        wc.count
    }
}

impl fmt::Write for WriteCounting {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.count += s.len();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_fn() {
        let m: Matrix<f64, 3, 3> = Matrix::from_fn(|i, j| if i == j { 1.0 } else { 0.0 });
        assert_eq!(m, Matrix::eye());
    }

    #[test]
    fn map() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let doubled = m.map(|x| x * 2.0);
        assert_eq!(doubled[(0, 0)], 2.0);
        assert_eq!(doubled[(1, 1)], 8.0);
    }

    #[test]
    fn map_type_change() {
        let m = Matrix::new([[1.0_f64, 2.0], [3.0, 4.0]]);
        let rounded = m.map(|x| x as i32);
        assert_eq!(rounded[(0, 0)], 1);
        assert_eq!(rounded[(1, 1)], 4);
    }

    #[test]
    fn row_col_access() {
        let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let r0 = m.row(0);
        assert_eq!(r0[0], 1.0);
        assert_eq!(r0[2], 3.0);

        let c1 = m.col(1);
        assert_eq!(c1[0], 2.0);
        assert_eq!(c1[1], 5.0);
    }

    #[test]
    fn set_row_col() {
        let mut m: Matrix<f64, 2, 2> = Matrix::zeros();

        m.set_row(0, &Vector::from_array([1.0, 2.0]));
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);

        m.set_col(1, &Vector::from_array([7.0, 8.0]));
        assert_eq!(m[(0, 1)], 7.0);
        assert_eq!(m[(1, 1)], 8.0);
    }

    #[test]
    fn display_2x2() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let s = format!("{}", m);
        assert!(s.contains("1"));
        assert!(s.contains("4"));
        // Verify it has two lines
        assert_eq!(s.lines().count(), 2);
    }

    #[test]
    fn display_alignment() {
        let m = Matrix::new([[1, 100], [1000, 2]]);
        let s = format!("{}", m);
        // Both rows should have the same length due to alignment
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines[0].len(), lines[1].len());
    }

    #[test]
    fn display_vector() {
        let v = Vector::from_array([1.0, 2.0, 3.0]);
        let s = format!("{}", v);
        // Vector uses compact bracket format on one line
        assert_eq!(s, "[1, 2, 3]");
    }

    #[test]
    fn display_precision() {
        let m = Matrix::new([[1.0_f64, 2.0], [3.0, 4.0]]);
        let s = format!("{:.2}", m);
        assert!(s.contains("1.00"));
        assert!(s.contains("4.00"));
    }

    #[test]
    fn display_vector_precision() {
        let v = Vector::from_array([1.0_f64, 2.5]);
        let s = format!("{:.3}", v);
        assert_eq!(s, "[1.000, 2.500]");
    }

    #[test]
    fn sum() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(m.sum(), 10.0);
    }

    #[test]
    fn sum_integer() {
        let m = Matrix::new([[1, 2, 3], [4, 5, 6]]);
        assert_eq!(m.sum(), 21);
    }

    #[test]
    fn abs() {
        let m = Matrix::new([[1.0_f64, -2.0], [-3.0, 4.0]]);
        let a = m.abs();
        assert_eq!(a[(0, 0)], 1.0);
        assert_eq!(a[(0, 1)], 2.0);
        assert_eq!(a[(1, 0)], 3.0);
        assert_eq!(a[(1, 1)], 4.0);
    }

    #[test]
    fn swap_rows() {
        let mut m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        m.swap_rows(0, 1);
        assert_eq!(m[(0, 0)], 3.0);
        assert_eq!(m[(0, 1)], 4.0);
        assert_eq!(m[(1, 0)], 1.0);
        assert_eq!(m[(1, 1)], 2.0);
    }

    #[test]
    fn swap_rows_same() {
        let mut m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let original = m;
        m.swap_rows(0, 0);
        assert_eq!(m, original);
    }

    #[test]
    fn swap_cols() {
        let mut m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        m.swap_cols(0, 1);
        assert_eq!(m[(0, 0)], 2.0);
        assert_eq!(m[(0, 1)], 1.0);
        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(1, 1)], 3.0);
    }
}
