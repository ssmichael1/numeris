pub mod dynarr;
pub mod dynmatrix;
pub mod matrix;
pub mod utils;

#[cfg(feature = "quaternion")]
mod quaternion;

/// Ordinary differential equations
#[cfg(feature = "ode")]
pub mod ode;

// Image processin
#[cfg(feature = "image")]
pub mod image;

pub mod prelude {
    pub use crate::dynarr::*;
    pub use crate::dynmatrix::*;
    pub use crate::matrix::*;

    pub use crate::mat;
    pub use crate::mat_col_major;
    pub use crate::mat_row_major;
    pub use crate::rowmat;

    #[cfg(feature = "quaternion")]
    pub use crate::quaternion::*;

    #[cfg(feature = "image")]
    pub use crate::image::*;

    pub use crate::utils;
}
