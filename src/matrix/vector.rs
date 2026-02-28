use core::ops::{Index, IndexMut};

use crate::traits::Scalar;
use crate::Matrix;

/// A row vector (1×N matrix).
///
/// Vectors support single-index access (`v[i]`), dot products, norms, and
/// cross products (3-element vectors). Use [`ColumnVector`] for column vectors.
///
/// # Examples
///
/// ```
/// use numeris::Vector;
///
/// let v = Vector::from_array([3.0_f64, 4.0]);
/// assert_eq!(v[0], 3.0);
/// assert_eq!(v.dot(&v), 25.0);
/// assert!((v.norm() - 5.0).abs() < 1e-12);
/// ```
pub type Vector<T, const N: usize> = Matrix<T, 1, N>;

impl<T: Scalar, const N: usize> Vector<T, N> {
    /// Create a vector from a 1D array.
    ///
    /// ```
    /// use numeris::Vector;
    /// let v = Vector::from_array([1.0, 2.0, 3.0]);
    /// assert_eq!(v[0], 1.0);
    /// ```
    #[inline]
    pub fn from_array(data: [T; N]) -> Self {
        Self::new([data])
    }

    /// Create a vector filled with a single value.
    #[inline]
    pub fn fill(value: T) -> Self {
        Self::new([[value; N]])
    }

    /// Number of elements.
    #[inline]
    pub const fn len(&self) -> usize {
        N
    }

    /// Dot product of two vectors.
    ///
    /// ```
    /// use numeris::Vector;
    /// let a = Vector::from_array([1.0, 2.0, 3.0]);
    /// let b = Vector::from_array([4.0, 5.0, 6.0]);
    /// assert_eq!(a.dot(&b), 32.0); // 1*4 + 2*5 + 3*6
    /// ```
    #[inline]
    pub fn dot(&self, rhs: &Self) -> T {
        let mut sum = T::zero();
        for i in 0..N {
            sum = sum + self[(0, i)] * rhs[(0, i)];
        }
        sum
    }
}

impl<T: Scalar, const N: usize> Vector<T, N> {
    /// Outer product: `v.outer(w)` → N×P matrix where `result[i][j] = v[i] * w[j]`.
    ///
    /// ```
    /// use numeris::Vector;
    /// let a = Vector::from_array([1.0, 2.0]);
    /// let b = Vector::from_array([3.0, 4.0, 5.0]);
    /// let m = a.outer(&b);
    /// assert_eq!(m[(0, 0)], 3.0);  // 1*3
    /// assert_eq!(m[(1, 2)], 10.0); // 2*5
    /// ```
    pub fn outer<const P: usize>(&self, rhs: &Vector<T, P>) -> Matrix<T, N, P> {
        let mut out = Matrix::<T, N, P>::zeros();
        for i in 0..N {
            for j in 0..P {
                out[(i, j)] = self[i] * rhs[j];
            }
        }
        out
    }
}

// ── Vector size aliases ─────────────────────────────────────────────

/// A 1-element row vector.
pub type Vector1<T> = Vector<T, 1>;
/// A 2-element row vector.
pub type Vector2<T> = Vector<T, 2>;
/// A 3-element row vector.
///
/// Adds `cross()` for cross product in addition to all `Vector` methods.
pub type Vector3<T> = Vector<T, 3>;
/// A 4-element row vector.
pub type Vector4<T> = Vector<T, 4>;
/// A 5-element row vector.
pub type Vector5<T> = Vector<T, 5>;
/// A 6-element row vector.
pub type Vector6<T> = Vector<T, 6>;

impl<T: Scalar> Vector3<T> {
    /// Cross product of two 3-vectors.
    ///
    /// ```
    /// use numeris::Vector3;
    /// let x = Vector3::from_array([1.0, 0.0, 0.0]);
    /// let y = Vector3::from_array([0.0, 1.0, 0.0]);
    /// let z = x.cross(&y);
    /// assert_eq!(z[2], 1.0); // x × y = z
    /// ```
    #[inline]
    pub fn cross(&self, rhs: &Self) -> Self {
        Self::from_array([
            self[1] * rhs[2] - self[2] * rhs[1],
            self[2] * rhs[0] - self[0] * rhs[2],
            self[0] * rhs[1] - self[1] * rhs[0],
        ])
    }
}

// Single-index access: v[i] instead of v[(0, i)]
impl<T, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;

    #[inline]
    fn index(&self, i: usize) -> &T {
        &self[(0, i)]
    }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self[(0, i)]
    }
}

// ── Column vector ───────────────────────────────────────────────────

/// A column vector (N×1 matrix).
///
/// Enables natural `Matrix * ColumnVector` multiplication:
/// `(M×N) * (N×1) → (M×1)`.
///
/// Convert between row and column vectors with `.transpose()`.
/// Single-element access uses `cv[(i, 0)]`.
pub type ColumnVector<T, const N: usize> = Matrix<T, N, 1>;

// ── Column vector size aliases ──────────────────────────────────────

/// A 1-element column vector.
pub type ColumnVector1<T> = ColumnVector<T, 1>;
/// A 2-element column vector.
pub type ColumnVector2<T> = ColumnVector<T, 2>;
/// A 3-element column vector.
pub type ColumnVector3<T> = ColumnVector<T, 3>;
/// A 4-element column vector.
pub type ColumnVector4<T> = ColumnVector<T, 4>;
/// A 5-element column vector.
pub type ColumnVector5<T> = ColumnVector<T, 5>;
/// A 6-element column vector.
pub type ColumnVector6<T> = ColumnVector<T, 6>;

impl<T: Scalar, const N: usize> ColumnVector<T, N> {
    /// Create a column vector from a 1D array.
    ///
    /// ```
    /// use numeris::ColumnVector;
    /// let cv = ColumnVector::from_column([1.0, 2.0, 3.0]);
    /// assert_eq!(cv[(0, 0)], 1.0);
    /// assert_eq!(cv[(2, 0)], 3.0);
    /// ```
    #[inline]
    pub fn from_column(data: [T; N]) -> Self {
        Self::new(data.map(|x| [x]))
    }
}

impl<T: Scalar> ColumnVector3<T> {
    /// Cross product of two 3-column-vectors.
    #[inline]
    pub fn cross_col(&self, rhs: &Self) -> Self {
        Self::from_column([
            self[(1, 0)] * rhs[(2, 0)] - self[(2, 0)] * rhs[(1, 0)],
            self[(2, 0)] * rhs[(0, 0)] - self[(0, 0)] * rhs[(2, 0)],
            self[(0, 0)] * rhs[(1, 0)] - self[(1, 0)] * rhs[(0, 0)],
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_array_and_index() {
        let v = Vector::from_array([1.0, 2.0, 3.0]);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn index_mut() {
        let mut v = Vector::<f64, 3>::zeros();
        v[1] = 5.0;
        assert_eq!(v[1], 5.0);
    }

    #[test]
    fn fill() {
        let v = Vector::<f64, 4>::fill(7.0);
        for i in 0..4 {
            assert_eq!(v[i], 7.0);
        }
    }

    #[test]
    fn dot_product() {
        let a = Vector::from_array([1.0, 2.0, 3.0]);
        let b = Vector::from_array([4.0, 5.0, 6.0]);
        assert_eq!(a.dot(&b), 32.0); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn len() {
        let v = Vector::from_array([1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn vector_arithmetic() {
        let a = Vector::from_array([1.0, 2.0, 3.0]);
        let b = Vector::from_array([4.0, 5.0, 6.0]);

        let c = a + b;
        assert_eq!(c[0], 5.0);
        assert_eq!(c[2], 9.0);

        let d = a * 2.0;
        assert_eq!(d[0], 2.0);
        assert_eq!(d[2], 6.0);
    }

    #[test]
    fn cross_product() {
        let x = Vector3::from_array([1.0, 0.0, 0.0]);
        let y = Vector3::from_array([0.0, 1.0, 0.0]);
        let z = x.cross(&y);
        assert_eq!(z[0], 0.0);
        assert_eq!(z[1], 0.0);
        assert_eq!(z[2], 1.0);
    }

    #[test]
    fn cross_product_anticommutative() {
        let a = Vector3::from_array([1.0, 2.0, 3.0]);
        let b = Vector3::from_array([4.0, 5.0, 6.0]);
        let ab = a.cross(&b);
        let ba = b.cross(&a);
        assert_eq!(ab[0], -ba[0]);
        assert_eq!(ab[1], -ba[1]);
        assert_eq!(ab[2], -ba[2]);
    }

    #[test]
    fn cross_product_self_is_zero() {
        let a = Vector3::from_array([3.0, -1.0, 4.0]);
        let c = a.cross(&a);
        assert_eq!(c[0], 0.0);
        assert_eq!(c[1], 0.0);
        assert_eq!(c[2], 0.0);
    }

    #[test]
    fn vector3_as_vector() {
        // Vector3 is just a Vector<T, 3>, so dot/arithmetic work
        let a = Vector3::from_array([1.0, 2.0, 3.0]);
        let b = Vector3::from_array([4.0, 5.0, 6.0]);
        assert_eq!(a.dot(&b), 32.0);
        assert_eq!((a + b)[0], 5.0);
    }

    // ── Outer product tests ─────────────────────────────────────

    #[test]
    fn outer_product() {
        let a = Vector::from_array([1.0, 2.0, 3.0]);
        let b = Vector::from_array([4.0, 5.0]);
        let m = a.outer(&b);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 2);
        assert_eq!(m[(0, 0)], 4.0);  // 1*4
        assert_eq!(m[(0, 1)], 5.0);  // 1*5
        assert_eq!(m[(1, 0)], 8.0);  // 2*4
        assert_eq!(m[(2, 1)], 15.0); // 3*5
    }

    #[test]
    fn outer_product_square() {
        let v = Vector::from_array([1.0, 2.0]);
        let m = v.outer(&v);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 0)], 2.0);
        assert_eq!(m[(1, 1)], 4.0);
        assert!(m.is_symmetric());
    }

    // ── Column vector tests ──────────────────────────────────────

    #[test]
    fn column_vector_from_column() {
        let cv = ColumnVector::from_column([1.0, 2.0, 3.0]);
        assert_eq!(cv[(0, 0)], 1.0);
        assert_eq!(cv[(1, 0)], 2.0);
        assert_eq!(cv[(2, 0)], 3.0);
        assert_eq!(cv.nrows(), 3);
        assert_eq!(cv.ncols(), 1);
    }

    #[test]
    fn matrix_times_column_vector() {
        // (2×3) * (3×1) → (2×1)
        let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let v = ColumnVector::from_column([7.0, 8.0, 9.0]);

        let result = m * v;
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 1);
        assert_eq!(result[(0, 0)], 50.0);  // 1*7 + 2*8 + 3*9
        assert_eq!(result[(1, 0)], 122.0); // 4*7 + 5*8 + 6*9
    }

    #[test]
    fn row_column_transpose_roundtrip() {
        let row = Vector::from_array([1.0, 2.0, 3.0]);
        let col: ColumnVector<f64, 3> = row.transpose();
        let back: Vector<f64, 3> = col.transpose();
        assert_eq!(row, back);
    }

    #[test]
    fn column_vector3_cross() {
        let x = ColumnVector3::from_column([1.0, 0.0, 0.0]);
        let y = ColumnVector3::from_column([0.0, 1.0, 0.0]);
        let z = x.cross_col(&y);
        assert_eq!(z[(0, 0)], 0.0);
        assert_eq!(z[(1, 0)], 0.0);
        assert_eq!(z[(2, 0)], 1.0);
    }

    #[test]
    fn square_matrix_times_column_vector() {
        // Ax = b style
        let a = Matrix::new([[2.0, 1.0], [5.0, 3.0]]);
        let x = ColumnVector::from_column([1.0, 2.0]);
        let b = a * x;
        assert_eq!(b[(0, 0)], 4.0);  // 2*1 + 1*2
        assert_eq!(b[(1, 0)], 11.0); // 5*1 + 3*2
    }

    // ── Row vector tests ────────────────────────────────────────

    #[test]
    fn vector_matrix_multiply() {
        // (1×2) * (2×3) → (1×3)
        let v = Vector::from_array([1.0, 2.0]);
        let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let result = v * m;
        assert_eq!(result[0], 9.0);  // 1*1 + 2*4
        assert_eq!(result[1], 12.0); // 1*2 + 2*5
        assert_eq!(result[2], 15.0); // 1*3 + 2*6
    }
}
