use alloc::vec::Vec;

use crate::dynmatrix::DynMatrix;
use crate::traits::{FloatScalar, MatrixRef};

use super::border::BorderMode;
use super::convolve::{
    convolve2d, convolve2d_separable, convolve2d_separable_cols, convolve2d_separable_into,
};
use super::kernels::{
    box_kernel_1d, gaussian_kernel_1d, scharr_x_3x3, scharr_y_3x3, sobel_x_3x3, sobel_y_3x3,
};
use super::ImageError;

/// The 1D kernel [`gaussian_blur`] uses for standard deviation `sigma`: a
/// Gaussian truncated at `3 σ` on each side, i.e.
/// [`gaussian_kernel_1d`]`(sigma, 3.0)`, of length `2 · ceil(3 σ) + 1`.
///
/// Exposed so callers of the streaming variant [`gaussian_blur_cols`] can
/// size their own halos or bands from the kernel length, and so the blur's
/// kernel can be reused with [`convolve2d_separable`] and friends.
///
/// # Errors
///
/// Returns [`ImageError::InvalidParameter`] if `sigma` is non-positive or
/// not finite.
///
/// # Example
///
/// ```
/// use numeris::imageproc::gaussian_blur_kernel;
///
/// let k = gaussian_blur_kernel::<f32>(1.5).unwrap();
/// assert_eq!(k.len(), 11); // 2 * ceil(4.5) + 1
/// ```
pub fn gaussian_blur_kernel<T: FloatScalar>(sigma: T) -> Result<Vec<T>, ImageError> {
    let three = T::from(3.0_f64).unwrap();
    gaussian_kernel_1d(sigma, three)
}

/// Gaussian blur with standard deviation `sigma`.
///
/// Implemented as two 1D separable passes with a Gaussian kernel truncated at
/// `3 σ` on each side ([`gaussian_blur_kernel`]). Pixels within `3 σ` of the
/// image edge use the chosen border mode.
///
/// A non-positive or non-finite `sigma` is clamped to returning the input
/// unchanged.
///
/// Allocates the intermediate and output images on every call; see
/// [`gaussian_blur_into`] to reuse caller-owned buffers and
/// [`gaussian_blur_cols`] to stream the result column by column without
/// materializing it. All three produce bit-identical results.
pub fn gaussian_blur<T: FloatScalar + crate::par::MaybeSync>(
    src: &DynMatrix<T>,
    sigma: T,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    let kernel = match gaussian_blur_kernel(sigma) {
        Ok(k) => k,
        Err(_) => return src.clone(),
    };
    convolve2d_separable(src, &kernel, &kernel, border)
}

/// Allocation-free [`gaussian_blur`]: writes the result into `dst`, using
/// `tmp` for the intermediate image.
///
/// Both buffers are resized to `src`'s shape if needed (reallocating only
/// when their capacity is insufficient) and their prior contents are
/// discarded; with correctly sized buffers the call performs no allocation.
/// See [`convolve2d_separable_into`] for why this matters on large images.
/// Bit-for-bit identical to [`gaussian_blur`].
///
/// A non-positive or non-finite `sigma` copies `src` into `dst` unchanged
/// (`tmp` is left untouched).
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
/// use numeris::imageproc::{gaussian_blur, gaussian_blur_into, BorderMode};
///
/// let img = DynMatrix::<f32>::from_fn(40, 30, |i, j| (i * 3 + j) as f32);
/// let mut tmp = DynMatrix::<f32>::zeros(40, 30);
/// let mut dst = DynMatrix::<f32>::zeros(40, 30);
/// gaussian_blur_into(&img, 1.5, BorderMode::Reflect, &mut tmp, &mut dst);
/// assert_eq!(dst.as_slice(), gaussian_blur(&img, 1.5, BorderMode::Reflect).as_slice());
/// ```
pub fn gaussian_blur_into<T: FloatScalar + crate::par::MaybeSync>(
    src: &DynMatrix<T>,
    sigma: T,
    border: BorderMode<T>,
    tmp: &mut DynMatrix<T>,
    dst: &mut DynMatrix<T>,
) {
    let kernel = match gaussian_blur_kernel(sigma) {
        Ok(k) => k,
        Err(_) => {
            dst.resize_discard(src.nrows(), src.ncols());
            dst.as_mut_slice().copy_from_slice(src.as_slice());
            return;
        }
    };
    convolve2d_separable_into(src, &kernel, &kernel, border, tmp, dst);
}

/// Banded [`gaussian_blur`] that hands each output **column** to a callback
/// instead of materializing the blurred image.
///
/// `f(j, col)` receives every column `j` of the blurred image exactly once,
/// bit-for-bit identical to column `j` of [`gaussian_blur`]'s output. The
/// image is processed in bands of `band` columns with a
/// `(band + K − 1) · nrows` scratch buffer (`K` = kernel length, see
/// [`gaussian_blur_kernel`]); under the `rayon` feature the bands run in
/// parallel and the callback is invoked concurrently, in no particular
/// order. `band` = 64–128 is a good default. Full details — the scratch
/// layout, threading contract, and the column-major note that a row-major
/// caller sees its image *rows* in the callback — are in
/// [`convolve2d_separable_cols`], which this wraps.
///
/// A non-positive or non-finite `sigma` streams the columns of `src`
/// unchanged.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
/// use numeris::imageproc::{gaussian_blur_cols, BorderMode};
/// use std::sync::Mutex;
///
/// let img = DynMatrix::<f32>::from_fn(64, 100, |i, j| ((i ^ j) % 5) as f32);
/// // Per-column maxima of the blurred image, without the blurred image.
/// let maxima = Mutex::new(vec![0.0f32; 100]);
/// gaussian_blur_cols(&img, 1.5, BorderMode::Replicate, 32, |j, col| {
///     let m = col.iter().cloned().fold(f32::MIN, f32::max);
///     maxima.lock().unwrap()[j] = m;
/// });
/// assert!(maxima.into_inner().unwrap().iter().all(|&m| m > 0.0));
/// ```
pub fn gaussian_blur_cols<T, F>(
    src: &DynMatrix<T>,
    sigma: T,
    border: BorderMode<T>,
    band: usize,
    f: F,
) where
    T: FloatScalar + crate::par::MaybeSync,
    F: Fn(usize, &[T]) + Sync,
{
    let kernel = match gaussian_blur_kernel(sigma) {
        Ok(k) => k,
        Err(_) => {
            for j in 0..src.ncols() {
                f(j, src.col_as_slice(j, 0));
            }
            return;
        }
    };
    convolve2d_separable_cols(src, &kernel, &kernel, border, band, f);
}

/// Box (mean) blur with odd radius `radius`, i.e. kernel length `2·radius + 1`.
///
/// Equivalent to averaging every pixel over a `(2r+1) × (2r+1)` window via
/// two separable 1D box filters.
pub fn box_blur<T: FloatScalar + crate::par::MaybeSync>(
    src: &DynMatrix<T>,
    radius: usize,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    if radius == 0 {
        return src.clone();
    }
    let n = 2 * radius + 1;
    let kernel = match box_kernel_1d::<T>(n) {
        Ok(k) => k,
        Err(_) => return src.clone(),
    };
    convolve2d_separable(src, &kernel, &kernel, border)
}

/// Apply the 3×3 Laplacian operator (4-neighbour variant).
///
/// The output is the discrete second-order derivative `∂²/∂x² + ∂²/∂y²`,
/// useful as an edge or blob detector (zero-crossings mark edges).
pub fn laplacian<T: FloatScalar>(src: &DynMatrix<T>, border: BorderMode<T>) -> DynMatrix<T> {
    let k = super::kernels::laplacian_3x3::<T>();
    convolve2d(src, &k, border)
}

/// Laplacian-of-Gaussian (LoG): smooth with a Gaussian of scale `sigma`, then
/// apply the 3×3 Laplacian operator.
///
/// Pre-smoothing suppresses the noise amplification inherent to raw
/// Laplacians. Zero-crossings of the result localize edges at scale `sigma`.
pub fn laplacian_of_gaussian<T: FloatScalar + crate::par::MaybeSync>(
    src: &DynMatrix<T>,
    sigma: T,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    let smoothed = gaussian_blur(src, sigma, border);
    laplacian(&smoothed, border)
}

/// Unsharp mask: sharpen `src` by adding `amount × (src − gaussian_blur(src, sigma))`.
///
/// `amount = 1.0` doubles the high-frequency content; typical values range
/// from 0.5 (subtle) to 2.0 (aggressive). A non-positive `sigma` returns the
/// input unchanged.
pub fn unsharp_mask<T: FloatScalar + crate::par::MaybeSync>(
    src: &DynMatrix<T>,
    sigma: T,
    amount: T,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    if !sigma.is_finite() || sigma <= T::zero() {
        return src.clone();
    }
    // Reuse the blurred buffer: out = src + amount·(src − blurred)
    let mut out = gaussian_blur(src, sigma, border);
    for (o, &s) in out.as_mut_slice().iter_mut().zip(src.as_slice()) {
        let b = *o;
        *o = s + amount * (s - b);
    }
    out
}

/// Scharr gradients: returns `(Gx, Gy)` using the 3×3 Scharr operator, a
/// Sobel variant with better rotational symmetry.
pub fn scharr_gradients<T: FloatScalar>(
    src: &DynMatrix<T>,
    border: BorderMode<T>,
) -> (DynMatrix<T>, DynMatrix<T>) {
    let kx = scharr_x_3x3::<T>();
    let ky = scharr_y_3x3::<T>();
    let gx = convolve2d(src, &kx, border);
    let gy = convolve2d(src, &ky, border);
    (gx, gy)
}

/// Per-pixel gradient magnitude `√(gx² + gy²)`.
///
/// `gx` and `gy` must have the same dimensions (typically the outputs of
/// [`sobel_gradients`] or [`scharr_gradients`]).
///
/// # Panics
///
/// Panics if `gx` and `gy` have different dimensions.
pub fn gradient_magnitude<T: FloatScalar>(gx: &DynMatrix<T>, gy: &DynMatrix<T>) -> DynMatrix<T> {
    assert_eq!(
        (gx.nrows(), gx.ncols()),
        (gy.nrows(), gy.ncols()),
        "gradient magnitude inputs must have the same shape",
    );
    let mut out = DynMatrix::<T>::zeros(gx.nrows(), gx.ncols());
    for j in 0..gx.ncols() {
        for i in 0..gx.nrows() {
            let a = gx[(i, j)];
            let b = gy[(i, j)];
            out[(i, j)] = (a * a + b * b).sqrt();
        }
    }
    out
}

/// Sobel gradients: returns `(Gx, Gy)` where `Gx` is the horizontal
/// derivative and `Gy` is the vertical derivative, computed with the 3×3
/// Sobel operators.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
/// use numeris::imageproc::{sobel_gradients, BorderMode};
///
/// // A vertical dark→bright step: left half 0, right half 1.
/// let mut img = DynMatrix::<f64>::zeros(8, 8);
/// for i in 0..8 {
///     for j in 4..8 {
///         img[(i, j)] = 1.0;
///     }
/// }
/// let (gx, _gy) = sobel_gradients(&img, BorderMode::Replicate);
/// // The strong positive response lies along the step column.
/// assert!(gx[(4, 3)] > 0.0);
/// assert!(gx[(4, 4)] > 0.0);
/// ```
pub fn sobel_gradients<T: FloatScalar>(
    src: &DynMatrix<T>,
    border: BorderMode<T>,
) -> (DynMatrix<T>, DynMatrix<T>) {
    let kx = sobel_x_3x3::<T>();
    let ky = sobel_y_3x3::<T>();
    let gx = convolve2d(src, &kx, border);
    let gy = convolve2d(src, &ky, border);
    (gx, gy)
}
