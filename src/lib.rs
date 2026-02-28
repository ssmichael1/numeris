//! # numeris
//!
//! Pure-Rust numerical algorithms library, no-std compatible. Similar in scope
//! to SciPy, suitable for embedded targets (no heap allocation, no FPU assumptions).
//!
//! ## Quick start
//!
//! ```
//! use numeris::{Matrix, Vector};
//!
//! // Solve a linear system Ax = b
//! let a = Matrix::new([
//!     [2.0_f64, 1.0, -1.0],
//!     [-3.0, -1.0, 2.0],
//!     [-2.0, 1.0, 2.0],
//! ]);
//! let b = Vector::from_array([8.0, -11.0, -3.0]);
//! let x = a.solve(&b).unwrap(); // x = [2, 3, -1]
//! ```
//!
//! ## Modules
//!
//! - [`matrix`] — Fixed-size `Matrix<T, M, N>` with const-generic dimensions.
//!   Stack-allocated `[[T; N]; M]` row-major storage. Includes arithmetic,
//!   indexing, norms, block operations, and iteration. [`Vector<T, N>`] and
//!   [`ColumnVector<T, N>`] are type aliases for 1-row and 1-column matrices.
//!
//! - [`linalg`] — LU (partial pivoting), Cholesky (A = LL^H), and QR
//!   (Householder) decompositions. Each provides `solve()`, `inverse()`, and
//!   `det()`. Free functions operate on `&mut impl MatrixMut<T>` for in-place
//!   use; wrapper structs offer a higher-level API. Convenience methods on
//!   `Matrix`: `a.solve(&b)`, `a.inverse()`, `a.det()`, `a.cholesky()`, `a.qr()`.
//!
//! - [`quaternion`] — Unit quaternion for 3D rotations. Scalar-first `[w, x, y, z]`.
//!   Construct from axis-angle, Euler angles, or rotation matrices. Supports
//!   Hamilton product, vector rotation, SLERP, and conversion back to matrices.
//!
//! - [`traits`] — Element trait hierarchy:
//!   - [`Scalar`] — all matrix elements (`Copy + PartialEq + Debug + Zero + One + Num`)
//!   - [`FloatScalar`] — real floats (`Scalar + Float`), used by quaternions
//!   - [`LinalgScalar`] — real floats and complex numbers, used by decompositions and norms
//!   - [`MatrixRef`] / [`MatrixMut`] — generic read/write access for algorithms
//!
//! ## Complex matrices
//!
//! Enable the `complex` feature to use decompositions with `Complex<f32>` /
//! `Complex<f64>`. Cholesky generalizes to Hermitian (A = LL^H), QR uses
//! complex Householder reflections, and norms return real values. Zero overhead
//! for real-only code paths.
//!
//! ## Cargo features
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std` | yes | Hardware FPU via system libm |
//! | `libm` | baseline | Pure-Rust software float fallback |
//! | `complex` | no | `Complex<f32>` / `Complex<f64>` support via `num-complex` |

#![cfg_attr(not(feature = "std"), no_std)]

pub mod linalg;
pub mod matrix;
pub mod quaternion;
pub mod traits;

pub use matrix::vector::{ColumnVector, ColumnVector3, Vector, Vector3};
pub use matrix::Matrix;
pub use quaternion::Quaternion;
pub use traits::{FloatScalar, LinalgScalar, MatrixMut, MatrixRef, Scalar};

#[cfg(feature = "complex")]
pub use num_complex::Complex;
