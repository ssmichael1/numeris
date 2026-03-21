use core::ops::{Index, IndexMut};

use crate::traits::Scalar;
use crate::Matrix;

/// A column vector (N×1 matrix).
///
/// Vectors support single-index access (`v[i]`), dot products, norms, and
/// cross products (3-element vectors). Natural matrix-vector multiplication:
/// `(M×N) * (N×1) → (M×1)`.
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
pub type Vector<T, const N: usize> = Matrix<T, N, 1>;

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
        // Vector is N×1: column-major storage is [[T;N]; 1]
        Self { data: [data] }
    }

    /// Create a vector filled with a single value.
    #[inline]
    pub fn fill(value: T) -> Self {
        Self {
            data: [[value; N]],
        }
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
        crate::simd::dot_dispatch(self.as_slice(), rhs.as_slice())
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

/// A 3-element vector.
///
/// Adds `cross()` for cross product in addition to all `Vector` methods.
pub type Vector3<T> = Vector<T, 3>;

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

// ── Named component accessors ──────────────────────────────────────

impl<T: Copy, const N: usize> Vector<T, N> {
    /// First component.
    ///
    /// ```
    /// use numeris::Vector;
    /// let v = Vector::from_array([1.0, 2.0, 3.0]);
    /// assert_eq!(v.x(), 1.0);
    /// ```
    #[inline]
    pub fn x(&self) -> T { self[0] }

    /// Set first component.
    #[inline]
    pub fn set_x(&mut self, val: T) { self[0] = val; }

    /// Second component.
    ///
    /// ```
    /// use numeris::Vector;
    /// let v = Vector::from_array([1.0, 2.0, 3.0]);
    /// assert_eq!(v.y(), 2.0);
    /// ```
    #[inline]
    pub fn y(&self) -> T { self[1] }

    /// Set second component.
    #[inline]
    pub fn set_y(&mut self, val: T) { self[1] = val; }
}

impl<T: Copy> Vector3<T> {
    /// Third component.
    ///
    /// ```
    /// use numeris::Vector3;
    /// let v = Vector3::from_array([1.0, 2.0, 3.0]);
    /// assert_eq!(v.z(), 3.0);
    /// ```
    #[inline]
    pub fn z(&self) -> T { self[2] }

    /// Set third component.
    #[inline]
    pub fn set_z(&mut self, val: T) { self[2] = val; }
}

// Single-index access: v[i] instead of v[(i, 0)]
impl<T, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;

    #[inline]
    fn index(&self, i: usize) -> &T {
        &self[(i, 0)]
    }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self[(i, 0)]
    }
}

// ── From<[T; N]> ────────────────────────────────────────────────────

impl<T: Scalar, const N: usize> From<[T; N]> for Vector<T, N> {
    /// Create a vector from an array.
    ///
    /// ```
    /// use numeris::Vector;
    /// let v: Vector<f64, 3> = [1.0, 2.0, 3.0].into();
    /// assert_eq!(v[0], 1.0);
    /// ```
    #[inline]
    fn from(data: [T; N]) -> Self {
        Self::from_array(data)
    }
}

// ── Ordering (lexicographic) ────────────────────────────────────────

impl<T: PartialOrd, const N: usize> PartialOrd for Vector<T, N> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        for i in 0..N {
            match self.data[0][i].partial_cmp(&other.data[0][i]) {
                Some(core::cmp::Ordering::Equal) => continue,
                ord => return ord,
            }
        }
        Some(core::cmp::Ordering::Equal)
    }
}

impl<T: Ord, const N: usize> Ord for Vector<T, N> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        for i in 0..N {
            match self.data[0][i].cmp(&other.data[0][i]) {
                core::cmp::Ordering::Equal => continue,
                ord => return ord,
            }
        }
        core::cmp::Ordering::Equal
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

    // ── Matrix-vector multiplication ────────────────────────────

    #[test]
    fn matrix_times_vector() {
        // (2×3) * (3×1) → (2×1)
        let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let v = Vector::from_array([7.0, 8.0, 9.0]);

        let result = m * v;
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 1);
        assert_eq!(result[0], 50.0);  // 1*7 + 2*8 + 3*9
        assert_eq!(result[1], 122.0); // 4*7 + 5*8 + 6*9
    }

    #[test]
    fn square_matrix_times_vector() {
        // Ax = b style
        let a = Matrix::new([[2.0, 1.0], [5.0, 3.0]]);
        let x = Vector::from_array([1.0, 2.0]);
        let b = a * x;
        assert_eq!(b[0], 4.0);  // 2*1 + 1*2
        assert_eq!(b[1], 11.0); // 5*1 + 3*2
    }

    #[test]
    fn vector_is_column() {
        let v = Vector::from_array([1.0, 2.0, 3.0]);
        assert_eq!(v.nrows(), 3);
        assert_eq!(v.ncols(), 1);
    }
}
