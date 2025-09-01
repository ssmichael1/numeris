//! Compact, performant representation of fixed-size matrices and vector
//!
//! Matrix and vector sizes must be known at compile time.  This eliminates
//! the need for dynamic memory allocation and allows for matrix data allocation
//! on the stack.
//!
//! This module provides a Matrix struct and associated methods for performing
//! linear algebra operations on fixed-size matrices and vectors.
//!
//! Vectors and matrices can hold somewhat arbitrary types (see `MatrixElem`), but
//! in general are designed for `f32` and `f64` types.
//!
//! # Included Functionality
//
//! ## Matrices:
//!
//! * Creation from arrays
//! * Matrix multiplication
//! * Matrix transposition
//! * Matrix inversion
//! * Matrix determinant
//!
//! ## Vectors:
//! * Creation from arrays
//! * Vector addition
//! * Vector subtraction
//! * Dot product
//! * Cross product (for 3D vectors)
//!
//! # Examples
//!
//! ```rust
//! use tiny_matrix::prelude::*;
//! let m = Matrix::from_row_major([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//! //let v = vec!([1.0, 2.0, 3.0]);
//! //let result = m * v;
//! ```

mod display;
mod macros;
mod mat_impl;
mod mat_ops;
mod mat_square_impl;
mod types;
mod vector;

pub use types::*;

pub trait MatrixElem:
    Copy
    + num_traits::Zero
    + num_traits::One
    + Default
    + std::fmt::Debug
    + std::ops::AddAssign
    + std::ops::MulAssign
    + std::ops::Neg<Output = Self>
    + std::cmp::PartialEq
    + std::ops::Div<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::DivAssign
    + std::iter::Sum
    + std::fmt::Display
{
}

impl<T> MatrixElem for T where
    T: Copy
        + num_traits::Zero
        + num_traits::One
        + Default
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::Neg<Output = Self>
        + std::cmp::PartialEq
        + std::ops::Div<Output = Self>
        + std::ops::Sub<Output = Self>
        + std::ops::DivAssign
        + std::iter::Sum
        + std::fmt::Display
{
}

/// A fixed-size matrix generic, with rows and columns known at compile time
/// This struct is designed for performance and minimal memory usage in embedded systems.
/// It achieves this by using a simple array representation and avoiding dynamic memory allocation.
///
/// Data is stored natively in column-major order
///
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix<const ROWS: usize, const COLS: usize, T> {
    pub(crate) data: [[T; ROWS]; COLS],
}
