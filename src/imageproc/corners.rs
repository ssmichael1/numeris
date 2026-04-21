use crate::dynmatrix::DynMatrix;
use crate::traits::FloatScalar;

use super::border::BorderMode;
use super::filters::{gaussian_blur, sobel_gradients};

/// **Harris corner response** map.
///
/// Builds the structure tensor `M = G_σ * [gx² gxgy; gxgy gy²]` (Gaussian-
/// smoothed products of Sobel gradients) and returns the Harris scoring
/// function `det(M) − k · trace(M)²` at every pixel. Corners yield large
/// positive responses; edges yield large negative values; flat regions
/// yield ≈ 0.
///
/// Typical parameters: `sigma ≈ 1.0`, `k ∈ [0.04, 0.06]`.
///
/// The returned map is **not** thresholded or non-max-suppressed — the
/// caller selects a response threshold and either takes connected components
/// or a simple local-maximum pass depending on what they want.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
/// use numeris::imageproc::{harris_corners, BorderMode};
///
/// // A black square on a white background has 4 strong corners.
/// let mut img = DynMatrix::<f64>::fill(32, 32, 1.0);
/// for i in 10..22 { for j in 10..22 { img[(i, j)] = 0.0; } }
/// let r = harris_corners(&img, 1.0, 0.05, BorderMode::Replicate);
/// // Each of the four corners should have a strong positive response.
/// assert!(r[(10, 10)] > 0.0);
/// assert!(r[(21, 21)] > 0.0);
/// ```
pub fn harris_corners<T: FloatScalar>(
    src: &DynMatrix<T>,
    sigma: T,
    k: T,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    let (sxx, sxy, syy) = structure_tensor(src, sigma, border);
    let h = src.nrows();
    let w = src.ncols();
    let mut out = DynMatrix::<T>::zeros(h, w);
    for j in 0..w {
        for i in 0..h {
            let a = sxx[(i, j)];
            let b = sxy[(i, j)];
            let c = syy[(i, j)];
            let det = a * c - b * b;
            let tr = a + c;
            out[(i, j)] = det - k * tr * tr;
        }
    }
    out
}

/// **Shi-Tomasi corner response** map (a.k.a. "Good Features to Track").
///
/// Same structure tensor as [`harris_corners`], but scored by the smaller
/// eigenvalue `min(λ₁, λ₂)`. This is more selective than Harris for corner-
/// versus-edge discrimination.
///
/// For a symmetric 2×2 matrix `M = [a b; b c]`, the smaller eigenvalue is
/// `((a + c) − √((a − c)² + 4 b²)) / 2`, clamped to 0 if negative rounding
/// slips through.
pub fn shi_tomasi_corners<T: FloatScalar>(
    src: &DynMatrix<T>,
    sigma: T,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    let (sxx, sxy, syy) = structure_tensor(src, sigma, border);
    let h = src.nrows();
    let w = src.ncols();
    let half = T::from(0.5_f64).unwrap();
    let four = T::from(4.0_f64).unwrap();
    let zero = T::zero();
    let mut out = DynMatrix::<T>::zeros(h, w);
    for j in 0..w {
        for i in 0..h {
            let a = sxx[(i, j)];
            let b = sxy[(i, j)];
            let c = syy[(i, j)];
            let diff = a - c;
            let disc = (diff * diff + four * b * b).sqrt();
            let lam_min = half * (a + c - disc);
            out[(i, j)] = if lam_min < zero { zero } else { lam_min };
        }
    }
    out
}

// ── internal ──────────────────────────────────────────────────────────

/// Compute the three independent entries of the Gaussian-smoothed
/// structure tensor: `(G * gx², G * gxgy, G * gy²)`.
fn structure_tensor<T: FloatScalar>(
    src: &DynMatrix<T>,
    sigma: T,
    border: BorderMode<T>,
) -> (DynMatrix<T>, DynMatrix<T>, DynMatrix<T>) {
    let (gx, gy) = sobel_gradients(src, border);
    let h = src.nrows();
    let w = src.ncols();
    let ixx = DynMatrix::from_fn(h, w, |i, j| gx[(i, j)] * gx[(i, j)]);
    let ixy = DynMatrix::from_fn(h, w, |i, j| gx[(i, j)] * gy[(i, j)]);
    let iyy = DynMatrix::from_fn(h, w, |i, j| gy[(i, j)] * gy[(i, j)]);
    let sxx = gaussian_blur(&ixx, sigma, border);
    let sxy = gaussian_blur(&ixy, sigma, border);
    let syy = gaussian_blur(&iyy, sigma, border);
    (sxx, sxy, syy)
}
