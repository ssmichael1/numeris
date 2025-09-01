//!
//! Multi-dimensional arrays with dynamic sizes
//!
//! This module provides a way to create and manipulate multi-dimensional arrays
//! where the sizes of each dimension can be determined at runtime.
//!
//! The arrays support a wide-range of element-wise functions
//!   (e.g., add, multiply, subtract, divide, trigonometric functions, exp, log, sqrt)
//!   as well as mapping functions.
//!
//! # Notes:
//!
//! - Arrays can be operated on element-wise using standard operators,
//!   however when operators are between two arrays, the arrays must have
//!   the same shape.  For such functions, a `Result` type is returned to
//!   indicate success or failure.
//!
//! - For operators with scalars, the operators always succeed and return a new array.
//!
//! # Example
//!
//!
//!

mod dynarr_err;
mod dynarr_funcs;
mod dynarr_impl;
mod dynarr_ops;
mod dynarr_types;

pub use dynarr_err::*;
pub use dynarr_types::*;

pub trait ArrayElem: Copy + Default {}
impl<T> ArrayElem for T where T: Copy + Default {}

/// Dynamically-sized multi-dimensional arrays
///
/// # Overview
///
/// The `DynArray` type provides a flexible and efficient way to work with
/// multi-dimensional arrays of arbitrary size at runtime. It is designed for
/// ease of use and performance, making it suitable for a wide range of
/// applications.
///
/// # Features
///
/// - Dynamic sizing: Create arrays of any size without knowing the
///   dimensions at compile time.
/// - Efficient memory management: The array data is stored in a
///   contiguous vector, ensuring good cache performance.
/// - Comprehensive API: The `DynArray` type implements a wide range
///   of array operations, making it easy to perform common tasks.
///
/// # Notes
///
/// - common operators (+, -, / *) act element-wise on the array if the right-hand
///   side is another array.  Because the sizes must match, the operators return a `Result`
///   indicating success or failure.
///
/// - for operators with scalars, the operators always succeed and return a new array.
///
#[derive(Debug, Clone, PartialEq)]
pub struct DynArray<T>
where
    T: ArrayElem,
{
    pub(crate) data: Vec<T>,
    pub(crate) shape_: Vec<usize>,
}
