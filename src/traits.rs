use core::fmt::Debug;
use num_traits::{Float, Num, One, Zero};

/// Trait for types that can be used as matrix elements.
///
/// Blanket-implemented for all types satisfying the bounds.
/// Covers `f32`, `f64`, and all integer types.
pub trait Scalar: Copy + PartialEq + Debug + Zero + One + Num {}

impl<T: Copy + PartialEq + Debug + Zero + One + Num> Scalar for T {}

/// Trait for floating-point matrix elements.
///
/// Required by operations that need `sqrt`, `sin`, `abs`, etc.
/// (decompositions, norms, trigonometric functions).
pub trait FloatScalar: Scalar + Float {}

impl<T: Scalar + Float> FloatScalar for T {}

/// Read-only access to a matrix-like type.
///
/// This trait allows algorithms to operate generically over
/// both fixed-size `Matrix` and future `DynMatrix` types.
pub trait MatrixRef<T> {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn get(&self, row: usize, col: usize) -> &T;
}

/// Mutable access to a matrix-like type.
///
/// Extends `MatrixRef` with mutable element access, enabling
/// in-place algorithms (Cholesky, LU, etc.) to work generically.
pub trait MatrixMut<T>: MatrixRef<T> {
    fn get_mut(&mut self, row: usize, col: usize) -> &mut T;
}
