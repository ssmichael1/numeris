use alloc::vec::Vec;

use crate::dynmatrix::DynMatrix;
use crate::traits::FloatScalar;

use super::border::BorderMode;
use super::filters::gaussian_blur;
use super::resize::resize_bilinear;

/// Difference of Gaussians: `gaussian_blur(src, σ₁) − gaussian_blur(src, σ₂)`.
///
/// A band-pass filter that approximates the Laplacian of Gaussian. Peaks
/// localize blobs at the scale `σ = √(σ₁ · σ₂)` (geometric mean). The
/// convention here follows the common edge/blob-detector form: the smaller
/// σ is subtracted from the larger, so bright blobs near the scale produce
/// positive response.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
/// use numeris::imageproc::{difference_of_gaussians, BorderMode};
///
/// let img = DynMatrix::<f64>::fill(32, 32, 5.0);
/// let dog = difference_of_gaussians(&img, 1.0, 1.6, BorderMode::Replicate);
/// // A uniform image has zero DoG response everywhere.
/// for i in 0..32 { for j in 0..32 { assert!(dog[(i, j)].abs() < 1e-10); } }
/// ```
pub fn difference_of_gaussians<T: FloatScalar>(
    src: &DynMatrix<T>,
    sigma_small: T,
    sigma_large: T,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    let a = gaussian_blur(src, sigma_small, border);
    let b = gaussian_blur(src, sigma_large, border);
    let h = src.nrows();
    let w = src.ncols();
    let mut out = DynMatrix::<T>::zeros(h, w);
    for j in 0..w {
        for i in 0..h {
            out[(i, j)] = a[(i, j)] - b[(i, j)];
        }
    }
    out
}

/// Build a Gaussian pyramid: `levels` images, each half the size of the
/// previous, pre-blurred to suppress aliasing from the 2× downsample.
///
/// The first element is the input image at its original size (no blur); each
/// subsequent element is the previous blurred by `sigma` then downsampled.
/// The pyramid stops early if either dimension reaches 1.
///
/// # Parameters
///
/// * `sigma` — typical value `1.0`, matching a cutoff near Nyquist for a
///   2× decimation. Increase for gentler rolloff.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
/// use numeris::imageproc::{gaussian_pyramid, BorderMode};
///
/// let img = DynMatrix::<f64>::fill(128, 128, 1.0);
/// let pyr = gaussian_pyramid(&img, 4, 1.0, BorderMode::Replicate);
/// assert_eq!(pyr.len(), 4);
/// assert_eq!(pyr[0].nrows(), 128);
/// assert_eq!(pyr[1].nrows(), 64);
/// assert_eq!(pyr[2].nrows(), 32);
/// assert_eq!(pyr[3].nrows(), 16);
/// ```
pub fn gaussian_pyramid<T: FloatScalar>(
    src: &DynMatrix<T>,
    levels: usize,
    sigma: T,
    border: BorderMode<T>,
) -> Vec<DynMatrix<T>> {
    let mut out = Vec::with_capacity(levels);
    if levels == 0 {
        return out;
    }
    out.push(src.clone());
    for _ in 1..levels {
        let prev = out.last().unwrap();
        if prev.nrows() <= 1 || prev.ncols() <= 1 {
            break;
        }
        let blurred = gaussian_blur(prev, sigma, border);
        // Downsample by 2 using bilinear — smoother than nearest-neighbour.
        let down = resize_bilinear(&blurred, prev.nrows() / 2, prev.ncols() / 2);
        out.push(down);
    }
    out
}
