use alloc::vec::Vec;
use core::fmt::{self, Write as _};

use crate::traits::{FloatScalar, Scalar};

use super::vector::DynVector;
use super::DynMatrix;

// ── Aggregation ─────────────────────────────────────────────────────

impl<T: Scalar> DynMatrix<T> {
    /// Sum of all elements.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(m.sum(), 10.0);
    /// ```
    pub fn sum(&self) -> T {
        let mut s = T::zero();
        for &x in &self.data {
            s = s + x;
        }
        s
    }
}

// ── Map ─────────────────────────────────────────────────────────────

impl<T> DynMatrix<T> {
    /// Apply a function to every element, producing a new matrix.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::from_slice(2, 2, &[1.0_f64, 4.0, 9.0, 16.0]);
    /// let r = m.map(|x: f64| x.sqrt());
    /// assert_eq!(r[(0, 0)], 1.0);
    /// assert_eq!(r[(1, 1)], 4.0);
    /// ```
    pub fn map<U>(&self, f: impl Fn(T) -> U) -> DynMatrix<U>
    where
        T: Copy,
    {
        let data: Vec<U> = self.data.iter().map(|&x| f(x)).collect();
        DynMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

// ── Element-wise operations ─────────────────────────────────────────

impl<T: FloatScalar> DynMatrix<T> {
    /// Element-wise absolute value.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::from_slice(2, 2, &[1.0_f64, -2.0, -3.0, 4.0]);
    /// let a = m.abs();
    /// assert_eq!(a[(0, 1)], 2.0);
    /// assert_eq!(a[(1, 0)], 3.0);
    /// ```
    pub fn abs(&self) -> Self {
        self.map(|x| x.abs())
    }

    /// Element-wise maximum: `c[i][j] = max(a[i][j], b[i][j])`.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let a = DynMatrix::from_slice(2, 2, &[1.0_f64, 5.0, 3.0, 2.0]);
    /// let b = DynMatrix::from_slice(2, 2, &[4.0, 2.0, 1.0, 6.0]);
    /// let c = a.element_max(&b);
    /// assert_eq!(c[(0, 0)], 4.0);
    /// assert_eq!(c[(0, 1)], 5.0);
    /// ```
    pub fn element_max(&self, rhs: &Self) -> Self {
        assert_eq!(
            (self.nrows, self.ncols),
            (rhs.nrows, rhs.ncols),
            "dimension mismatch",
        );
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&a, &b)| if b > a { b } else { a })
            .collect();
        DynMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

// ── Row / Column manipulation ───────────────────────────────────────

impl<T> DynMatrix<T> {
    /// Swap two rows in place.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let mut m = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    /// m.swap_rows(0, 1);
    /// assert_eq!(m[(0, 0)], 3.0);
    /// assert_eq!(m[(1, 0)], 1.0);
    /// ```
    pub fn swap_rows(&mut self, a: usize, b: usize) {
        if a != b {
            let n = self.ncols;
            for j in 0..n {
                self.data.swap(a * n + j, b * n + j);
            }
        }
    }
}

impl<T: Copy> DynMatrix<T> {
    /// Swap two columns in place.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let mut m = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    /// m.swap_cols(0, 1);
    /// assert_eq!(m[(0, 0)], 2.0);
    /// assert_eq!(m[(0, 1)], 1.0);
    /// ```
    pub fn swap_cols(&mut self, a: usize, b: usize) {
        if a != b {
            for i in 0..self.nrows {
                let ia = i * self.ncols + a;
                let ib = i * self.ncols + b;
                self.data.swap(ia, ib);
            }
        }
    }
}

// ── Row / Column access ─────────────────────────────────────────────

impl<T: Scalar> DynMatrix<T> {
    /// Extract row `i` as a `DynVector`.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    /// let r = m.row(0);
    /// assert_eq!(r[0], 1.0);
    /// assert_eq!(r[1], 2.0);
    /// ```
    pub fn row(&self, i: usize) -> DynVector<T> {
        DynVector::from_slice(self.row_slice(i))
    }

    /// Set row `i` from a `DynVector`.
    pub fn set_row(&mut self, i: usize, v: &DynVector<T>) {
        assert_eq!(v.len(), self.ncols, "vector length mismatch");
        for j in 0..self.ncols {
            self[(i, j)] = v[j];
        }
    }

    /// Extract column `j` as a `DynVector`.
    ///
    /// ```
    /// use numeris::DynMatrix;
    /// let m = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    /// let c = m.col(1);
    /// assert_eq!(c[0], 2.0);
    /// assert_eq!(c[1], 4.0);
    /// ```
    pub fn col(&self, j: usize) -> DynVector<T> {
        let mut data = Vec::with_capacity(self.nrows);
        for i in 0..self.nrows {
            data.push(self[(i, j)]);
        }
        DynVector::from_vec(data)
    }

    /// Set column `j` from a `DynVector`.
    pub fn set_col(&mut self, j: usize, v: &DynVector<T>) {
        assert_eq!(v.len(), self.nrows, "vector length mismatch");
        for i in 0..self.nrows {
            self[(i, j)] = v[i];
        }
    }
}

// ── Display ─────────────────────────────────────────────────────────

impl<T: fmt::Display> fmt::Display for DynMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let m = self.nrows;
        let n = self.ncols;

        // Measure column widths
        let mut widths: Vec<usize> = alloc::vec![0; n];
        for j in 0..n {
            for i in 0..m {
                let w = WriteCounting::count(|wc| write!(wc, "{}", self[(i, j)]));
                if w > widths[j] {
                    widths[j] = w;
                }
            }
        }

        for i in 0..m {
            write!(f, "│")?;
            for j in 0..n {
                if j > 0 {
                    write!(f, "  ")?;
                }
                write!(f, "{:>width$}", self[(i, j)], width = widths[j])?;
            }
            write!(f, "│")?;
            if i < m - 1 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

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
    fn sum() {
        let m = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m.sum(), 10.0);
    }

    #[test]
    fn map() {
        let m = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let doubled = m.map(|x| x * 2.0);
        assert_eq!(doubled[(0, 0)], 2.0);
        assert_eq!(doubled[(1, 1)], 8.0);
    }

    #[test]
    fn map_type_change() {
        let m = DynMatrix::from_slice(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]);
        let rounded = m.map(|x| x as i32);
        assert_eq!(rounded[(0, 0)], 1);
        assert_eq!(rounded[(1, 1)], 4);
    }

    #[test]
    fn abs() {
        let m = DynMatrix::from_slice(2, 2, &[1.0_f64, -2.0, -3.0, 4.0]);
        let a = m.abs();
        assert_eq!(a[(0, 0)], 1.0);
        assert_eq!(a[(0, 1)], 2.0);
        assert_eq!(a[(1, 0)], 3.0);
        assert_eq!(a[(1, 1)], 4.0);
    }

    #[test]
    fn element_max() {
        let a = DynMatrix::from_slice(2, 2, &[1.0_f64, 5.0, 3.0, 2.0]);
        let b = DynMatrix::from_slice(2, 2, &[4.0, 2.0, 1.0, 6.0]);
        let c = a.element_max(&b);
        assert_eq!(c[(0, 0)], 4.0);
        assert_eq!(c[(0, 1)], 5.0);
        assert_eq!(c[(1, 0)], 3.0);
        assert_eq!(c[(1, 1)], 6.0);
    }

    #[test]
    fn swap_rows() {
        let mut m = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        m.swap_rows(0, 1);
        assert_eq!(m[(0, 0)], 3.0);
        assert_eq!(m[(1, 0)], 1.0);
    }

    #[test]
    fn swap_cols() {
        let mut m = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        m.swap_cols(0, 1);
        assert_eq!(m[(0, 0)], 2.0);
        assert_eq!(m[(0, 1)], 1.0);
    }

    #[test]
    fn row_col() {
        let m = DynMatrix::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = m.row(0);
        assert_eq!(r[0], 1.0);
        assert_eq!(r[2], 3.0);

        let c = m.col(1);
        assert_eq!(c[0], 2.0);
        assert_eq!(c[1], 5.0);
    }

    #[test]
    fn set_row_col() {
        let mut m = DynMatrix::zeros(2, 2, 0.0_f64);
        m.set_row(0, &DynVector::from_slice(&[1.0, 2.0]));
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);

        m.set_col(1, &DynVector::from_slice(&[7.0, 8.0]));
        assert_eq!(m[(0, 1)], 7.0);
        assert_eq!(m[(1, 1)], 8.0);
    }

    #[test]
    fn display() {
        let m = DynMatrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let s = format!("{}", m);
        assert!(s.contains("1"));
        assert!(s.contains("4"));
        assert_eq!(s.lines().count(), 2);
    }

    #[test]
    fn display_alignment() {
        let m = DynMatrix::from_slice(2, 2, &[1.0, 100.0, 1000.0, 2.0]);
        let s = format!("{}", m);
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines[0].len(), lines[1].len());
    }
}
