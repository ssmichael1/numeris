use crate::dynmatrix::DynMatrix;
use crate::traits::FloatScalar;

use super::integral::{integral_image, integral_rect_sum};

/// Local (moving-window) mean over a `(2·radius + 1) × (2·radius + 1)`
/// window, computed in **O(1) per pixel** via a summed-area table.
///
/// Boundary pixels see a clipped window: the mean is taken over only the
/// portion of the window that intersects the image (a.k.a. "no border
/// padding" semantics). This differs slightly from a convolution with a
/// normalized box kernel under [`BorderMode::Replicate`], which extends the
/// image with edge values; prefer [`box_blur`](super::box_blur) if you need
/// that behavior.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
/// use numeris::imageproc::local_mean;
///
/// let img = DynMatrix::<f64>::fill(8, 8, 5.0);
/// let mean = local_mean(&img, 2);
/// for i in 0..8 { for j in 0..8 { assert!((mean[(i, j)] - 5.0).abs() < 1e-12); } }
/// ```
pub fn local_mean<T: FloatScalar>(src: &DynMatrix<T>, radius: usize) -> DynMatrix<T> {
    let h = src.nrows();
    let w = src.ncols();
    if h == 0 || w == 0 {
        return DynMatrix::<T>::zeros(h, w);
    }
    let sat = integral_image(src);
    let mut out = DynMatrix::<T>::zeros(h, w);
    let r = radius;
    for j in 0..w {
        let c0 = j.saturating_sub(r);
        let c1 = (j + r + 1).min(w);
        for i in 0..h {
            let r0 = i.saturating_sub(r);
            let r1 = (i + r + 1).min(h);
            let count = T::from((r1 - r0) * (c1 - c0)).unwrap();
            let sum = integral_rect_sum(&sat, r0, c0, r1, c1);
            out[(i, j)] = sum / count;
        }
    }
    out
}

/// Local (moving-window) **variance** over a `(2·radius + 1) × (2·radius + 1)`
/// window, O(1) per pixel via two summed-area tables (one of `x`, one of
/// `x²`). Boundary windows are clipped to the image (same semantics as
/// [`local_mean`]).
///
/// The *population* variance `E[x²] − (E[x])²` is returned (not Bessel-
/// corrected). Negative results due to floating-point roundoff are clamped
/// to zero.
pub fn local_variance<T: FloatScalar>(src: &DynMatrix<T>, radius: usize) -> DynMatrix<T> {
    let h = src.nrows();
    let w = src.ncols();
    if h == 0 || w == 0 {
        return DynMatrix::<T>::zeros(h, w);
    }
    // Integral of x, and of x².
    let sat = integral_image(src);
    let squared = DynMatrix::from_fn(h, w, |i, j| {
        let v = src[(i, j)];
        v * v
    });
    let sat2 = integral_image(&squared);

    let mut out = DynMatrix::<T>::zeros(h, w);
    let zero = T::zero();
    for j in 0..w {
        let c0 = j.saturating_sub(radius);
        let c1 = (j + radius + 1).min(w);
        for i in 0..h {
            let r0 = i.saturating_sub(radius);
            let r1 = (i + radius + 1).min(h);
            let count = T::from((r1 - r0) * (c1 - c0)).unwrap();
            let s1 = integral_rect_sum(&sat, r0, c0, r1, c1);
            let s2 = integral_rect_sum(&sat2, r0, c0, r1, c1);
            let mean = s1 / count;
            let var = s2 / count - mean * mean;
            out[(i, j)] = if var < zero { zero } else { var };
        }
    }
    out
}

/// Local (moving-window) **standard deviation**: `sqrt(local_variance)`.
pub fn local_stddev<T: FloatScalar>(src: &DynMatrix<T>, radius: usize) -> DynMatrix<T> {
    let var = local_variance(src, radius);
    let h = var.nrows();
    let w = var.ncols();
    let mut out = DynMatrix::<T>::zeros(h, w);
    for j in 0..w {
        for i in 0..h {
            out[(i, j)] = var[(i, j)].sqrt();
        }
    }
    out
}
