use core::fmt::{self, Write as _};

use crate::matrix::vector::Vector;
use crate::traits::Scalar;
use crate::Matrix;

// ── Constructors ────────────────────────────────────────────────────

impl<T, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Create a matrix by calling `f(row, col)` for each element.
    pub fn from_fn(f: impl Fn(usize, usize) -> T) -> Self
    where
        T: Copy + Default,
    {
        let mut data = [[T::default(); N]; M];
        for i in 0..M {
            for j in 0..N {
                data[i][j] = f(i, j);
            }
        }
        Self::new(data)
    }

    /// Apply a function to every element, producing a new matrix.
    pub fn map<U: Copy + Default>(&self, f: impl Fn(T) -> U) -> Matrix<U, M, N>
    where
        T: Copy,
    {
        let mut data = [[U::default(); N]; M];
        for i in 0..M {
            for j in 0..N {
                data[i][j] = f(self[(i, j)]);
            }
        }
        Matrix::new(data)
    }
}

// ── Row / Column access ─────────────────────────────────────────────

impl<T: Scalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Extract row `i` as a row vector.
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
        // Find max width per column for alignment
        // Use a fixed buffer to avoid alloc
        let mut widths = [0usize; N];
        // We'll format twice: once to measure, once to print.
        // For no_std, measure by formatting to a counting sink.
        for j in 0..N {
            for i in 0..M {
                let w = WriteCounting::count(|wc| write!(wc, "{}", self[(i, j)]));
                if w > widths[j] {
                    widths[j] = w;
                }
            }
        }

        for i in 0..M {
            write!(f, "│")?;
            for j in 0..N {
                if j > 0 {
                    write!(f, "  ")?;
                }
                write!(f, "{:>width$}", self[(i, j)], width = widths[j])?;
            }
            write!(f, "│")?;
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
        let m: Matrix<f64, 3, 3> = Matrix::from_fn(|i, j| {
            if i == j { 1.0 } else { 0.0 }
        });
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
        // Vector is 1×N, so should be a single line
        assert_eq!(s.lines().count(), 1);
    }
}
