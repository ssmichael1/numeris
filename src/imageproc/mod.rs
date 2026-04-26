//! Image processing: 2D convolution, Gaussian/box blur, Sobel gradients,
//! and kernel generators.
//!
//! Operates on heap-allocated [`DynMatrix<T>`](crate::DynMatrix) buffers where
//! the row index is the vertical (y) axis and the column index is the
//! horizontal (x) axis. `T` must be a [`FloatScalar`](crate::FloatScalar)
//! (`f32` or `f64`).
//!
//! Column-major storage means a single image column is contiguous in memory,
//! so convolution inner loops are implemented as column-wise AXPY accumulations
//! and dispatch automatically to SIMD kernels (NEON / SSE2 / AVX / AVX-512).
//!
//! # Convolution convention
//!
//! All convolution functions apply the kernel as *correlation* (no flip),
//! matching the convention used by OpenCV `filter2D` and MATLAB `imfilter`.
//! For symmetric kernels (Gaussian, box) this is equivalent to true
//! convolution. Asymmetric kernels (Sobel) are provided pre-flipped so
//! correlation produces the intended gradient sign.
//!
//! # Examples
//!
//! ```
//! use numeris::DynMatrix;
//! use numeris::imageproc::{gaussian_blur, BorderMode};
//!
//! let img = DynMatrix::<f64>::zeros(16, 16);
//! let blurred = gaussian_blur(&img, 1.5, BorderMode::Replicate);
//! assert_eq!(blurred.nrows(), 16);
//! assert_eq!(blurred.ncols(), 16);
//! ```
//!
//! # Typical pipeline
//!
//! ```
//! use numeris::DynMatrix;
//! use numeris::imageproc::{sobel_gradients, BorderMode};
//!
//! let img = DynMatrix::<f64>::zeros(32, 32);
//! let (gx, gy) = sobel_gradients(&img, BorderMode::Replicate);
//! ```

mod border;
mod canny;
mod connected;
mod convolve;
mod corners;
mod filters;
mod geometric;
mod integral;
mod kernels;
mod local_stats;
mod multiscale;
mod morphology;
mod pool;
mod rank;
mod resize;
mod threshold;

#[cfg(test)]
mod tests;

pub use border::{fetch_border, BorderMode};
pub use canny::canny;
pub use connected::{
    connected_components, connected_components_labeled,
    connected_components_with_label_buffer, Component, Connectivity,
};
pub use convolve::{convolve2d, convolve2d_separable};
pub use corners::{harris_corners, shi_tomasi_corners};
pub use filters::{
    box_blur, gaussian_blur, gradient_magnitude, laplacian, laplacian_of_gaussian,
    scharr_gradients, sobel_gradients, unsharp_mask,
};
pub use integral::{integral_image, integral_rect_sum};
pub use kernels::{
    box_kernel_1d, gaussian_kernel_1d, laplacian_3x3, laplacian_3x3_diag, scharr_x_3x3,
    scharr_y_3x3, sobel_x_3x3, sobel_y_3x3,
};
pub use geometric::{
    crop, flip_horizontal, flip_vertical, pad, resize_nearest, rotate_180, rotate_270, rotate_90,
};
pub use local_stats::{local_mean, local_stddev, local_variance};
pub use multiscale::{difference_of_gaussians, gaussian_pyramid};
pub use morphology::{
    black_hat, closing, dilate, erode, max_filter, min_filter, morphology_gradient, opening,
    top_hat,
};
pub use pool::{median_pool, median_pool_upsampled};
pub use rank::{median_filter, median_filter_u16, percentile_filter, rank_filter};
pub use resize::resize_bilinear;
pub use threshold::{adaptive_threshold, threshold, threshold_otsu};

/// Errors from image processing operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImageError {
    /// Kernel has zero length, or even length (an odd length is required to
    /// center the kernel on a pixel).
    InvalidKernelSize,
    /// A numeric parameter was non-positive or non-finite (e.g. `sigma <= 0`).
    InvalidParameter,
}

impl core::fmt::Display for ImageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ImageError::InvalidKernelSize => write!(f, "kernel length must be odd and nonzero"),
            ImageError::InvalidParameter => write!(f, "parameter must be positive and finite"),
        }
    }
}
