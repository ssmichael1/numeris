//! Quaternion module
//!
//! # Quaternions
//! Quaternions are a number system that extends complex numbers.
//! They are used in 3D graphics and robotics for representing rotations in 3D space.
//!
//! This module provides a Quaternion struct and associated methods
//! for representing attitude and rotations in 3 spatial dimensions
//!
//! # References:
//! - [Quaternions and Spatial Rotation](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation)
//! - [Understanding Quaternions](https://www.oreilly.com/library/view/understanding-computer-graphics/9780133001725/ch04s02.html)
//!
//!
//! # Notes:
//! * A quaternion rotation of a 3D vector is represented by the Mul operator
//!
//! # Examples:
//!
//! ```rust
//! use numeris::prelude::*;
//! let q = Quaternion::rotz(std::f64::consts::PI / 2.0);
//! // Right-handed rotation of xhat vector by Pi/2 radians about zhat
//! // This should produce yhat
//! let rotated = q * Vector3d::xhat();
//! // Check approximate equality of rotated vector with yhat
//! for idx in 0..3 {
//!     assert!((rotated - Vector3d::yhat())[idx].abs() < 1e-15);
//! }
//! ```

mod quat_impl;
mod quat_ops;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion<T>
where
    T: num_traits::Float,
{
    pub w: T,
    pub x: T,
    pub y: T,
    pub z: T,
}
