use super::Matrix;

// Matrix types
/// 2x2 Matrix of generic type
pub type Matrix2<T> = Matrix<2, 2, T>;
/// 3x3 Matrix of generic type
pub type Matrix3<T> = Matrix<3, 3, T>;
/// 4x4 Matrix of generic type
pub type Matrix4<T> = Matrix<4, 4, T>;
/// 5x5 Matrix of generic type
pub type Matrix5<T> = Matrix<5, 5, T>;
/// 6x6 Matrix of generic type
pub type Matrix6<T> = Matrix<6, 6, T>;

/// 2x2 double-precision matrix
pub type Matrix2d = Matrix<2, 2, f64>;
/// 2x2 single-precision matrix
pub type Matrix2f = Matrix<2, 2, f32>;

/// 3x3 double-precision matrix
pub type Matrix3d = Matrix<3, 3, f64>;
/// 3x3 single-precision matrix
pub type Matrix3f = Matrix<3, 3, f32>;

/// 4x4 double-precision matrix
pub type Matrix4d = Matrix<4, 4, f64>;
/// 4x4 single-precision matrix
pub type Matrix4f = Matrix<4, 4, f32>;

/// 5x5 double-precision matrix
pub type Matrix5d = Matrix<5, 5, f64>;
/// 5x5 single-precision matrix
pub type Matrix5f = Matrix<5, 5, f32>;

/// 6x6 double-precision matrix
pub type Matrix6d = Matrix<6, 6, f64>;
/// 6x6 single-precision matrix
pub type Matrix6f = Matrix<6, 6, f32>;

/// 1D Vectors (Row Matrix) types
pub type Vector<const N: usize, T> = Matrix<N, 1, T>;

/// Fixed-length 2D vector type
pub type Vector2<T> = Matrix<2, 1, T>;
/// Fixed-length 3D vector type
pub type Vector3<T> = Matrix<3, 1, T>;
/// Fixed-length 4D vector type
pub type Vector4<T> = Matrix<4, 1, T>;

// 1D Vectors (Row Matrix) types with scalar specification

/// Fixed-length double-precision vector type
pub type VectorNd<const N: usize> = Matrix<N, 1, f64>;
/// Fixed-length single-precision vector type
pub type VectorNf<const N: usize> = Matrix<N, 1, f32>;

/// Fixed-length 2D double-precision vector type
pub type Vector2d = Matrix<2, 1, f64>;
/// Fixed-length 2D single-precision vector type
pub type Vector2f = Matrix<2, 1, f32>;
/// Fixed-length 3D double-precision vector type
pub type Vector3d = Matrix<3, 1, f64>;
/// Fixed-length 3D single-precision vector type
pub type Vector3f = Matrix<3, 1, f32>;
/// Fixed-length 4D double-precision vector type
pub type Vector4d = Matrix<4, 1, f64>;
/// Fixed-length 4D single-precision vector type
pub type Vector4f = Matrix<4, 1, f32>;
