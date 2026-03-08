//! Convenience re-exports for common types and traits.
//!
//! ```
//! use numeris::prelude::*;
//! use numeris::{matrix, vector};
//!
//! let m = matrix![1.0, 2.0; 3.0, 4.0];
//! let v = vector![1.0_f64, 2.0];
//! let r = m.vecmul(&v);
//! ```

pub use crate::Matrix;
pub use crate::matrix::vector::{ColumnVector, ColumnVector3, Vector, Vector3};
pub use crate::matrix::aliases::{
    Matrix1, Matrix2, Matrix3, Matrix4, Matrix5, Matrix6,
    Vector1, Vector2, Vector4, Vector5, Vector6,
    ColumnVector1, ColumnVector2, ColumnVector4, ColumnVector5, ColumnVector6,
};
pub use crate::traits::{FloatScalar, LinalgScalar, Scalar};
pub use crate::linalg::LinalgError;
pub use crate::quaternion::Quaternion;
