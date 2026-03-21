//! Conversions between numeris and nalgebra matrix/vector types.
//!
//! Enabled by the `nalgebra` cargo feature. Provides zero-copy borrows
//! via `as_slice` where possible, and cheap `memcpy`-level owned conversions
//! for fixed-size and dynamic matrices.
//!
//! # Layout compatibility
//!
//! Both numeris and nalgebra use column-major contiguous storage:
//! - `Matrix<T, M, N>` stores `[[T; M]; N]` — N columns of M rows
//! - `nalgebra::SMatrix<T, R, C>` stores `[T; R*C]` column-major
//!
//! Owned conversions go through `from_column_slice` / `as_slice`, which is a
//! single `memcpy` for fixed-size types (≤ 288 bytes for a 6×6 f64 matrix).
//!
//! # Vector convention
//!
//! numeris `Vector<T, N>` is `Matrix<T, N, 1>` — an N×1 column vector,
//! matching nalgebra's `SVector<T, N>` = `SMatrix<T, N, 1>` exactly.
//! The `From` impls for `Matrix<T, M, N>` cover vectors automatically.
//!
//! # Examples
//!
//! ```
//! # #[cfg(feature = "nalgebra")] {
//! use numeris::Matrix;
//!
//! let nm = Matrix::new([[1.0_f64, 2.0], [3.0, 4.0]]);
//! let na: nalgebra::SMatrix<f64, 2, 2> = nm.into();
//! assert_eq!(na[(0, 0)], 1.0);
//! assert_eq!(na[(1, 0)], 3.0);
//!
//! let back: Matrix<f64, 2, 2> = na.into();
//! assert_eq!(back, nm);
//! # }
//! ```

use crate::traits::{MatrixMut, MatrixRef, Scalar};
use crate::Matrix;

// ── Fixed-size Matrix ↔ SMatrix ────────────────────────────────────

impl<T: Scalar + nalgebra::Scalar, const M: usize, const N: usize>
    From<Matrix<T, M, N>> for nalgebra::SMatrix<T, M, N>
where
    nalgebra::Const<M>: nalgebra::DimName,
    nalgebra::Const<N>: nalgebra::DimName,
{
    #[inline]
    fn from(m: Matrix<T, M, N>) -> Self {
        nalgebra::SMatrix::from_column_slice(m.as_slice())
    }
}

impl<T: Scalar + nalgebra::Scalar, const M: usize, const N: usize>
    From<&Matrix<T, M, N>> for nalgebra::SMatrix<T, M, N>
where
    nalgebra::Const<M>: nalgebra::DimName,
    nalgebra::Const<N>: nalgebra::DimName,
{
    #[inline]
    fn from(m: &Matrix<T, M, N>) -> Self {
        nalgebra::SMatrix::from_column_slice(m.as_slice())
    }
}

impl<T: Scalar + nalgebra::Scalar, const M: usize, const N: usize>
    From<nalgebra::SMatrix<T, M, N>> for Matrix<T, M, N>
where
    nalgebra::Const<M>: nalgebra::DimName,
    nalgebra::Const<N>: nalgebra::DimName,
{
    #[inline]
    fn from(m: nalgebra::SMatrix<T, M, N>) -> Self {
        Matrix::from_slice(m.as_slice())
    }
}

impl<T: Scalar + nalgebra::Scalar, const M: usize, const N: usize>
    From<&nalgebra::SMatrix<T, M, N>> for Matrix<T, M, N>
where
    nalgebra::Const<M>: nalgebra::DimName,
    nalgebra::Const<N>: nalgebra::DimName,
{
    #[inline]
    fn from(m: &nalgebra::SMatrix<T, M, N>) -> Self {
        Matrix::from_slice(m.as_slice())
    }
}

// ── Vector ↔ SVector ───────────────────────────────────────────────
//
// Vector<T,N> = Matrix<T,N,1> and SVector<T,N> = SMatrix<T,N,1>:
// already covered by the Matrix<T,M,N> ↔ SMatrix<T,M,N> impls above.
// No special convenience methods needed.

// ── MatrixRef / MatrixMut for nalgebra::SMatrix ────────────────────

impl<T: Scalar + nalgebra::Scalar, const M: usize, const N: usize>
    MatrixRef<T> for nalgebra::SMatrix<T, M, N>
where
    nalgebra::Const<M>: nalgebra::DimName,
    nalgebra::Const<N>: nalgebra::DimName,
{
    #[inline]
    fn nrows(&self) -> usize {
        M
    }

    #[inline]
    fn ncols(&self) -> usize {
        N
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> &T {
        &self[(row, col)]
    }

    #[inline]
    fn col_as_slice(&self, col: usize, row_start: usize) -> &[T] {
        let start = col * M + row_start;
        let end = (col + 1) * M;
        &self.as_slice()[start..end]
    }
}

impl<T: Scalar + nalgebra::Scalar, const M: usize, const N: usize>
    MatrixMut<T> for nalgebra::SMatrix<T, M, N>
where
    nalgebra::Const<M>: nalgebra::DimName,
    nalgebra::Const<N>: nalgebra::DimName,
{
    #[inline]
    fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self[(row, col)]
    }

    #[inline]
    fn col_as_mut_slice(&mut self, col: usize, row_start: usize) -> &mut [T] {
        let start = col * M + row_start;
        let end = (col + 1) * M;
        &mut self.as_mut_slice()[start..end]
    }
}

// ── Dynamic: DynMatrix ↔ DMatrix, DynVector ↔ DVector ─────────────

#[cfg(feature = "alloc")]
mod dyn_interop {
    use crate::dynmatrix::{DynMatrix, DynVector};
    use crate::traits::{MatrixMut, MatrixRef, Scalar};

    impl<T: Scalar + nalgebra::Scalar> From<DynMatrix<T>> for nalgebra::DMatrix<T> {
        #[inline]
        fn from(m: DynMatrix<T>) -> Self {
            nalgebra::DMatrix::from_column_slice(m.nrows(), m.ncols(), m.as_slice())
        }
    }

    impl<T: Scalar + nalgebra::Scalar> From<&DynMatrix<T>> for nalgebra::DMatrix<T> {
        #[inline]
        fn from(m: &DynMatrix<T>) -> Self {
            nalgebra::DMatrix::from_column_slice(m.nrows(), m.ncols(), m.as_slice())
        }
    }

    impl<T: Scalar + nalgebra::Scalar> From<nalgebra::DMatrix<T>> for DynMatrix<T> {
        #[inline]
        fn from(m: nalgebra::DMatrix<T>) -> Self {
            DynMatrix::from_slice(m.nrows(), m.ncols(), m.as_slice())
        }
    }

    impl<T: Scalar + nalgebra::Scalar> From<&nalgebra::DMatrix<T>> for DynMatrix<T> {
        #[inline]
        fn from(m: &nalgebra::DMatrix<T>) -> Self {
            DynMatrix::from_slice(m.nrows(), m.ncols(), m.as_slice())
        }
    }

    impl<T: Scalar + nalgebra::Scalar> From<DynVector<T>> for nalgebra::DVector<T> {
        #[inline]
        fn from(v: DynVector<T>) -> Self {
            nalgebra::DVector::from_column_slice(v.as_slice())
        }
    }

    impl<T: Scalar + nalgebra::Scalar> From<&DynVector<T>> for nalgebra::DVector<T> {
        #[inline]
        fn from(v: &DynVector<T>) -> Self {
            nalgebra::DVector::from_column_slice(v.as_slice())
        }
    }

    impl<T: Scalar + nalgebra::Scalar> From<nalgebra::DVector<T>> for DynVector<T> {
        #[inline]
        fn from(v: nalgebra::DVector<T>) -> Self {
            DynVector::from_slice(v.as_slice())
        }
    }

    impl<T: Scalar + nalgebra::Scalar> From<&nalgebra::DVector<T>> for DynVector<T> {
        #[inline]
        fn from(v: &nalgebra::DVector<T>) -> Self {
            DynVector::from_slice(v.as_slice())
        }
    }

    // ── MatrixRef / MatrixMut for nalgebra::DMatrix ────────────────

    impl<T: Scalar + nalgebra::Scalar> MatrixRef<T> for nalgebra::DMatrix<T> {
        #[inline]
        fn nrows(&self) -> usize {
            self.nrows()
        }

        #[inline]
        fn ncols(&self) -> usize {
            self.ncols()
        }

        #[inline]
        fn get(&self, row: usize, col: usize) -> &T {
            &self[(row, col)]
        }

        #[inline]
        fn col_as_slice(&self, col: usize, row_start: usize) -> &[T] {
            let nrows = self.nrows();
            let start = col * nrows + row_start;
            let end = (col + 1) * nrows;
            &self.as_slice()[start..end]
        }
    }

    impl<T: Scalar + nalgebra::Scalar> MatrixMut<T> for nalgebra::DMatrix<T> {
        #[inline]
        fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
            &mut self[(row, col)]
        }

        #[inline]
        fn col_as_mut_slice(&mut self, col: usize, row_start: usize) -> &mut [T] {
            let nrows = self.nrows();
            let start = col * nrows + row_start;
            let end = (col + 1) * nrows;
            &mut self.as_mut_slice()[start..end]
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_to_smatrix_roundtrip() {
        let nm = Matrix::new([[1.0_f64, 2.0], [3.0, 4.0]]);
        let na: nalgebra::SMatrix<f64, 2, 2> = nm.into();
        assert_eq!(na[(0, 0)], 1.0);
        assert_eq!(na[(0, 1)], 2.0);
        assert_eq!(na[(1, 0)], 3.0);
        assert_eq!(na[(1, 1)], 4.0);
        let back: Matrix<f64, 2, 2> = na.into();
        assert_eq!(back, nm);
    }

    #[test]
    fn matrix_ref_roundtrip() {
        let nm = Matrix::new([[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let na: nalgebra::SMatrix<f64, 2, 3> = (&nm).into();
        let back: Matrix<f64, 2, 3> = (&na).into();
        assert_eq!(back, nm);
    }

    #[test]
    fn vector_to_svector_roundtrip() {
        // Vector<T,N> = Matrix<T,N,1> and SVector<T,N> = SMatrix<T,N,1>:
        // From impls work directly, no transpose needed.
        use crate::Vector;
        let nv = Vector::from_array([1.0_f64, 2.0, 3.0]);
        let na: nalgebra::SVector<f64, 3> = nv.into();
        assert_eq!(na[0], 1.0);
        assert_eq!(na[1], 2.0);
        assert_eq!(na[2], 3.0);
        assert_eq!(na.nrows(), 3);
        assert_eq!(na.ncols(), 1);
        let back: Vector<f64, 3> = na.into();
        assert_eq!(back, nv);
    }

    #[test]
    fn vector_ref_roundtrip() {
        use crate::Vector;
        let nv = Vector::from_array([10.0_f32, 20.0, 30.0, 40.0]);
        let na: nalgebra::SMatrix<f32, 4, 1> = (&nv).into();
        let back: Vector<f32, 4> = (&na).into();
        assert_eq!(back, nv);
    }

    #[test]
    fn smatrix_matrix_ref_trait() {
        let na = nalgebra::SMatrix::<f64, 3, 3>::new(
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        );
        assert_eq!(na.nrows(), 3);
        assert_eq!(na.ncols(), 3);
        assert_eq!(*MatrixRef::get(&na, 0, 0), 1.0);
        assert_eq!(*MatrixRef::get(&na, 1, 0), 4.0);
        assert_eq!(*MatrixRef::get(&na, 0, 1), 2.0);

        let col1 = MatrixRef::col_as_slice(&na, 1, 0);
        assert_eq!(col1, &[2.0, 5.0, 8.0]);

        let col2_from1 = MatrixRef::col_as_slice(&na, 2, 1);
        assert_eq!(col2_from1, &[6.0, 9.0]);
    }

    #[test]
    fn smatrix_matrix_mut_trait() {
        let mut na = nalgebra::SMatrix::<f64, 2, 2>::zeros();
        *MatrixMut::get_mut(&mut na, 0, 1) = 42.0;
        assert_eq!(na[(0, 1)], 42.0);

        let col = MatrixMut::col_as_mut_slice(&mut na, 0, 0);
        col[0] = 10.0;
        col[1] = 20.0;
        assert_eq!(na[(0, 0)], 10.0);
        assert_eq!(na[(1, 0)], 20.0);
    }

    #[test]
    fn lu_on_nalgebra_smatrix() {
        use crate::linalg::lu::lu_in_place;

        let mut na = nalgebra::SMatrix::<f64, 3, 3>::new(
            2.0, 1.0, -1.0,
            -3.0, -1.0, 2.0,
            -2.0, 1.0, 2.0,
        );
        let mut perm = [0usize; 3];
        let result = lu_in_place(&mut na, &mut perm);
        assert!(result.is_ok());
    }

    #[test]
    fn cholesky_on_nalgebra_smatrix() {
        use crate::linalg::cholesky::cholesky_in_place;

        // SPD matrix: [[4, 2], [2, 3]]
        let mut na = nalgebra::SMatrix::<f64, 2, 2>::new(4.0, 2.0, 2.0, 3.0);
        let result = cholesky_in_place(&mut na);
        assert!(result.is_ok());
    }

    #[test]
    fn nonsquare_matrix_roundtrip() {
        let nm = Matrix::new([[1.0_f64, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]);
        let na: nalgebra::SMatrix<f64, 3, 4> = nm.into();
        let back: Matrix<f64, 3, 4> = na.into();
        assert_eq!(back, nm);
    }

    #[test]
    fn i32_matrix_roundtrip() {
        let nm = Matrix::new([[1_i32, 2], [3, 4]]);
        let na: nalgebra::SMatrix<i32, 2, 2> = nm.into();
        let back: Matrix<i32, 2, 2> = na.into();
        assert_eq!(back, nm);
    }

    #[cfg(feature = "alloc")]
    mod dyn_tests {
        use crate::dynmatrix::{DynMatrix, DynVector};
        use crate::traits::{MatrixMut, MatrixRef};

        #[test]
        fn dynmatrix_to_dmatrix_roundtrip() {
            let dm = DynMatrix::from_rows(2, 3, &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
            let na: nalgebra::DMatrix<f64> = (&dm).into();
            assert_eq!(na.nrows(), 2);
            assert_eq!(na.ncols(), 3);
            assert_eq!(na[(0, 0)], 1.0);
            assert_eq!(na[(0, 2)], 3.0);
            assert_eq!(na[(1, 0)], 4.0);
            let back: DynMatrix<f64> = na.into();
            assert_eq!(back, dm);
        }

        #[test]
        fn dynvector_to_dvector_roundtrip() {
            let dv = DynVector::from_slice(&[1.0_f64, 2.0, 3.0]);
            let na: nalgebra::DVector<f64> = (&dv).into();
            assert_eq!(na.len(), 3);
            assert_eq!(na[0], 1.0);
            assert_eq!(na[2], 3.0);
            let back: DynVector<f64> = na.into();
            assert_eq!(back, dv);
        }

        #[test]
        fn dmatrix_matrix_ref_trait() {
            let na = nalgebra::DMatrix::from_row_slice(2, 3, &[
                1.0_f64, 2.0, 3.0,
                4.0, 5.0, 6.0,
            ]);
            assert_eq!(MatrixRef::nrows(&na), 2);
            assert_eq!(MatrixRef::ncols(&na), 3);
            assert_eq!(*MatrixRef::get(&na, 0, 0), 1.0);
            assert_eq!(*MatrixRef::get(&na, 1, 0), 4.0);
            assert_eq!(*MatrixRef::get(&na, 0, 2), 3.0);

            let col1 = MatrixRef::col_as_slice(&na, 1, 0);
            assert_eq!(col1, &[2.0, 5.0]);
        }

        #[test]
        fn dmatrix_matrix_mut_trait() {
            let mut na = nalgebra::DMatrix::zeros(3, 3);
            *MatrixMut::get_mut(&mut na, 1, 2) = 99.0;
            assert_eq!(na[(1, 2)], 99.0);

            let col = MatrixMut::col_as_mut_slice(&mut na, 0, 1);
            col[0] = 10.0;
            col[1] = 20.0;
            assert_eq!(na[(1, 0)], 10.0);
            assert_eq!(na[(2, 0)], 20.0);
        }

        #[test]
        fn lu_on_nalgebra_dmatrix() {
            use crate::linalg::lu::lu_in_place;

            let mut na = nalgebra::DMatrix::from_row_slice(3, 3, &[
                2.0_f64, 1.0, -1.0,
                -3.0, -1.0, 2.0,
                -2.0, 1.0, 2.0,
            ]);
            let mut perm = [0usize; 3];
            let result = lu_in_place(&mut na, &mut perm);
            assert!(result.is_ok());
        }
    }
}
