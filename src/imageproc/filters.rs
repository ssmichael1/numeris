use crate::dynmatrix::DynMatrix;
use crate::traits::FloatScalar;

use super::border::BorderMode;
use super::convolve::{convolve2d, convolve2d_separable};
use super::kernels::{
    box_kernel_1d, gaussian_kernel_1d, scharr_x_3x3, scharr_y_3x3, sobel_x_3x3, sobel_y_3x3,
};

/// Gaussian blur with standard deviation `sigma`.
///
/// Implemented as two 1D separable passes with a Gaussian kernel truncated at
/// `3 σ` on each side. Pixels within `3 σ` of the image edge use the chosen
/// border mode.
///
/// A non-positive or non-finite `sigma` is clamped to returning the input
/// unchanged.
pub fn gaussian_blur<T: FloatScalar>(
    src: &DynMatrix<T>,
    sigma: T,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    if !sigma.is_finite() || sigma <= T::zero() {
        return src.clone();
    }
    let three = T::from(3.0_f64).unwrap();
    let kernel = match gaussian_kernel_1d(sigma, three) {
        Ok(k) => k,
        Err(_) => return src.clone(),
    };
    convolve2d_separable(src, &kernel, &kernel, border)
}

/// Box (mean) blur with odd radius `radius`, i.e. kernel length `2·radius + 1`.
///
/// Equivalent to averaging every pixel over a `(2r+1) × (2r+1)` window via
/// two separable 1D box filters.
pub fn box_blur<T: FloatScalar>(
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
pub fn laplacian_of_gaussian<T: FloatScalar>(
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
pub fn unsharp_mask<T: FloatScalar>(
    src: &DynMatrix<T>,
    sigma: T,
    amount: T,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    if !sigma.is_finite() || sigma <= T::zero() {
        return src.clone();
    }
    let blurred = gaussian_blur(src, sigma, border);
    let mut out = DynMatrix::<T>::zeros(src.nrows(), src.ncols());
    for j in 0..src.ncols() {
        for i in 0..src.nrows() {
            let s = src[(i, j)];
            let b = blurred[(i, j)];
            out[(i, j)] = s + amount * (s - b);
        }
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
pub fn gradient_magnitude<T: FloatScalar>(
    gx: &DynMatrix<T>,
    gy: &DynMatrix<T>,
) -> DynMatrix<T> {
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
