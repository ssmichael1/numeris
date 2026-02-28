#![cfg_attr(not(feature = "std"), no_std)]

pub mod linalg;
pub mod matrix;
pub mod traits;

pub use matrix::vector::{ColumnVector, ColumnVector3, Vector, Vector3};
pub use matrix::Matrix;
pub use traits::{FloatScalar, MatrixMut, MatrixRef, Scalar};
