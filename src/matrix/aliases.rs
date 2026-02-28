//! Pre-defined type aliases for common matrix, vector, and column vector sizes.

use crate::matrix::vector::{ColumnVector, Vector};
use crate::Matrix;

// ── Square matrix aliases ──────────────────────────────────────────

/// 1×1 matrix.
pub type Matrix1<T> = Matrix<T, 1, 1>;
/// 2×2 matrix.
pub type Matrix2<T> = Matrix<T, 2, 2>;
/// 3×3 matrix.
pub type Matrix3<T> = Matrix<T, 3, 3>;
/// 4×4 matrix.
pub type Matrix4<T> = Matrix<T, 4, 4>;
/// 5×5 matrix.
pub type Matrix5<T> = Matrix<T, 5, 5>;
/// 6×6 matrix.
pub type Matrix6<T> = Matrix<T, 6, 6>;

// ── Rectangular matrix aliases ─────────────────────────────────────

/// 1×2 matrix.
pub type Matrix1x2<T> = Matrix<T, 1, 2>;
/// 1×3 matrix.
pub type Matrix1x3<T> = Matrix<T, 1, 3>;
/// 1×4 matrix.
pub type Matrix1x4<T> = Matrix<T, 1, 4>;
/// 1×5 matrix.
pub type Matrix1x5<T> = Matrix<T, 1, 5>;
/// 1×6 matrix.
pub type Matrix1x6<T> = Matrix<T, 1, 6>;

/// 2×1 matrix.
pub type Matrix2x1<T> = Matrix<T, 2, 1>;
/// 2×3 matrix.
pub type Matrix2x3<T> = Matrix<T, 2, 3>;
/// 2×4 matrix.
pub type Matrix2x4<T> = Matrix<T, 2, 4>;
/// 2×5 matrix.
pub type Matrix2x5<T> = Matrix<T, 2, 5>;
/// 2×6 matrix.
pub type Matrix2x6<T> = Matrix<T, 2, 6>;

/// 3×1 matrix.
pub type Matrix3x1<T> = Matrix<T, 3, 1>;
/// 3×2 matrix.
pub type Matrix3x2<T> = Matrix<T, 3, 2>;
/// 3×4 matrix.
pub type Matrix3x4<T> = Matrix<T, 3, 4>;
/// 3×5 matrix.
pub type Matrix3x5<T> = Matrix<T, 3, 5>;
/// 3×6 matrix.
pub type Matrix3x6<T> = Matrix<T, 3, 6>;

/// 4×1 matrix.
pub type Matrix4x1<T> = Matrix<T, 4, 1>;
/// 4×2 matrix.
pub type Matrix4x2<T> = Matrix<T, 4, 2>;
/// 4×3 matrix.
pub type Matrix4x3<T> = Matrix<T, 4, 3>;
/// 4×5 matrix.
pub type Matrix4x5<T> = Matrix<T, 4, 5>;
/// 4×6 matrix.
pub type Matrix4x6<T> = Matrix<T, 4, 6>;

/// 5×1 matrix.
pub type Matrix5x1<T> = Matrix<T, 5, 1>;
/// 5×2 matrix.
pub type Matrix5x2<T> = Matrix<T, 5, 2>;
/// 5×3 matrix.
pub type Matrix5x3<T> = Matrix<T, 5, 3>;
/// 5×4 matrix.
pub type Matrix5x4<T> = Matrix<T, 5, 4>;
/// 5×6 matrix.
pub type Matrix5x6<T> = Matrix<T, 5, 6>;

/// 6×1 matrix.
pub type Matrix6x1<T> = Matrix<T, 6, 1>;
/// 6×2 matrix.
pub type Matrix6x2<T> = Matrix<T, 6, 2>;
/// 6×3 matrix.
pub type Matrix6x3<T> = Matrix<T, 6, 3>;
/// 6×4 matrix.
pub type Matrix6x4<T> = Matrix<T, 6, 4>;
/// 6×5 matrix.
pub type Matrix6x5<T> = Matrix<T, 6, 5>;

// ── Row vector aliases ─────────────────────────────────────────────

/// 1-element row vector.
pub type Vector1<T> = Vector<T, 1>;
/// 2-element row vector.
pub type Vector2<T> = Vector<T, 2>;
/// 4-element row vector.
pub type Vector4<T> = Vector<T, 4>;
/// 5-element row vector.
pub type Vector5<T> = Vector<T, 5>;
/// 6-element row vector.
pub type Vector6<T> = Vector<T, 6>;

// ── Column vector aliases ──────────────────────────────────────────

/// 1-element column vector.
pub type ColumnVector1<T> = ColumnVector<T, 1>;
/// 2-element column vector.
pub type ColumnVector2<T> = ColumnVector<T, 2>;
/// 4-element column vector.
pub type ColumnVector4<T> = ColumnVector<T, 4>;
/// 5-element column vector.
pub type ColumnVector5<T> = ColumnVector<T, 5>;
/// 6-element column vector.
pub type ColumnVector6<T> = ColumnVector<T, 6>;
