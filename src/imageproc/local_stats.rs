use crate::dynmatrix::DynMatrix;
use crate::traits::FloatScalar;

use super::integral::{integral_image, integral_rect_sum};

/// Approximate per-pixel-query work (`h · w`) above which the local-statistics
/// query loops fan their output columns out across threads under `rayon`. Each
/// pixel is an O(1) integral-image rectangle sum, so this is a coarse pixel-
/// count gate. (The summed-area table build itself stays sequential — it is a
/// prefix-sum scan, parallelized only via a separate two-pass decomposition.)
const LOCAL_STATS_WORK_BUDGET: usize = 250_000;

/// Floor on parallel column chunks (see [`crate::par::work_col_threshold`]).
const LOCAL_STATS_PAR_MIN_COLS: usize = 8;

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
pub fn local_mean<T: FloatScalar + crate::par::MaybeSync>(
    src: &DynMatrix<T>,
    radius: usize,
) -> DynMatrix<T> {
    let h = src.nrows();
    let w = src.ncols();
    if h == 0 || w == 0 {
        return DynMatrix::<T>::zeros(h, w);
    }
    let sat = integral_image(src);
    let mut out = DynMatrix::<T>::zeros(h, w);
    let r = radius;
    // Each output column reads only the shared summed-area table and writes its
    // own column, so the query runs in parallel under `rayon`.
    let threshold = crate::par::work_col_threshold(
        h,
        LOCAL_STATS_WORK_BUDGET,
        LOCAL_STATS_PAR_MIN_COLS,
    );
    crate::par::for_each_chunk_mut(out.as_mut_slice(), h, threshold, |j, out_col| {
        let c0 = j.saturating_sub(r);
        let c1 = (j + r + 1).min(w);
        for (i, cell) in out_col.iter_mut().enumerate() {
            let r0 = i.saturating_sub(r);
            let r1 = (i + r + 1).min(h);
            let count = T::from((r1 - r0) * (c1 - c0)).unwrap();
            let sum = integral_rect_sum(&sat, r0, c0, r1, c1);
            *cell = sum / count;
        }
    });
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
pub fn local_variance<T: FloatScalar + crate::par::MaybeSync>(
    src: &DynMatrix<T>,
    radius: usize,
) -> DynMatrix<T> {
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
    let threshold = crate::par::work_col_threshold(
        h,
        LOCAL_STATS_WORK_BUDGET,
        LOCAL_STATS_PAR_MIN_COLS,
    );
    crate::par::for_each_chunk_mut(out.as_mut_slice(), h, threshold, |j, out_col| {
        let c0 = j.saturating_sub(radius);
        let c1 = (j + radius + 1).min(w);
        for (i, cell) in out_col.iter_mut().enumerate() {
            let r0 = i.saturating_sub(radius);
            let r1 = (i + radius + 1).min(h);
            let count = T::from((r1 - r0) * (c1 - c0)).unwrap();
            let s1 = integral_rect_sum(&sat, r0, c0, r1, c1);
            let s2 = integral_rect_sum(&sat2, r0, c0, r1, c1);
            let mean = s1 / count;
            let var = s2 / count - mean * mean;
            *cell = if var < zero { zero } else { var };
        }
    });
    out
}

/// Local (moving-window) **standard deviation**: `sqrt(local_variance)`.
pub fn local_stddev<T: FloatScalar + crate::par::MaybeSync>(
    src: &DynMatrix<T>,
    radius: usize,
) -> DynMatrix<T> {
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
