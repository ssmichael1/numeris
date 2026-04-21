use alloc::vec;
use alloc::vec::Vec;

use crate::dynmatrix::DynMatrix;
use crate::traits::FloatScalar;

use super::local_stats::local_mean;

/// Binary threshold: output is `1.0` where `src > t`, else `0.0`.
pub fn threshold<T: FloatScalar>(src: &DynMatrix<T>, t: T) -> DynMatrix<T> {
    let h = src.nrows();
    let w = src.ncols();
    let one = T::one();
    let zero = T::zero();
    let mut out = DynMatrix::<T>::zeros(h, w);
    for j in 0..w {
        for i in 0..h {
            out[(i, j)] = if src[(i, j)] > t { one } else { zero };
        }
    }
    out
}

/// Automatic threshold selection via **Otsu's method**: finds the scalar
/// cutoff that maximizes between-class variance in the image histogram.
///
/// Builds a 256-bin histogram over `[min(src), max(src)]` and returns the
/// scalar boundary value between the two classes. Typical use:
///
/// ```rust
/// use numeris::DynMatrix;
/// use numeris::imageproc::{threshold, threshold_otsu};
///
/// let img = DynMatrix::<f64>::from_fn(16, 16, |i, j| if (i + j) % 5 < 2 { 1.0 } else { 9.0 });
/// let t = threshold_otsu(&img);
/// let binary = threshold(&img, t);
/// ```
///
/// Uniform or empty images return the image's single value.
pub fn threshold_otsu<T: FloatScalar>(src: &DynMatrix<T>) -> T {
    let h = src.nrows();
    let w = src.ncols();
    let n = h * w;
    if n == 0 {
        return T::zero();
    }
    // Find min and max.
    let mut lo = src[(0, 0)];
    let mut hi = lo;
    for j in 0..w {
        for i in 0..h {
            let v = src[(i, j)];
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
    }
    if hi <= lo {
        return lo;
    }

    const BINS: usize = 256;
    let bins_m1 = T::from(BINS - 1).unwrap();
    let range = hi - lo;

    // Build histogram.
    let mut hist: Vec<u64> = vec![0; BINS];
    for j in 0..w {
        for i in 0..h {
            let v = src[(i, j)];
            let f = ((v - lo) / range) * bins_m1;
            let idx = f
                .to_usize()
                .unwrap_or(0)
                .min(BINS - 1);
            hist[idx] += 1;
        }
    }

    // Otsu: maximize between-class variance σ_b² = w0 · w1 · (μ0 − μ1)².
    let total = n as f64;
    let mut sum_all = 0.0_f64;
    for (b, &h) in hist.iter().enumerate() {
        sum_all += b as f64 * h as f64;
    }

    let mut w0 = 0.0_f64;
    let mut sum0 = 0.0_f64;
    let mut best_var = -1.0_f64;
    // Track the full run of tied-maximum bins and return the midpoint —
    // important when clusters are well-separated and many intermediate
    // thresholds are mathematically equivalent.
    let mut tie_start: usize = 0;
    let mut tie_end: usize = 0;
    for b in 0..BINS {
        w0 += hist[b] as f64;
        if w0 <= 0.0 || w0 >= total {
            continue;
        }
        sum0 += b as f64 * hist[b] as f64;
        let w1 = total - w0;
        let mu0 = sum0 / w0;
        let mu1 = (sum_all - sum0) / w1;
        let var = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);
        if var > best_var {
            best_var = var;
            tie_start = b;
            tie_end = b;
        } else if var == best_var {
            tie_end = b;
        }
    }
    let best_bin = (tie_start + tie_end) / 2;

    // Map bin index back to threshold value at bin centre.
    let frac = (best_bin as f64 + 0.5) / BINS as f64;
    lo + T::from(frac).unwrap() * range
}

/// **Adaptive threshold**: compares each pixel against its local mean over
/// a `(2·radius + 1) × (2·radius + 1)` window, plus an offset:
/// output is `1.0` where `src(i, j) > local_mean(i, j) + offset`, else `0.0`.
///
/// Useful for uneven-illumination scenes where a single global threshold
/// fails. Local mean is computed via the integral image (O(1) per pixel).
pub fn adaptive_threshold<T: FloatScalar>(
    src: &DynMatrix<T>,
    radius: usize,
    offset: T,
) -> DynMatrix<T> {
    let mean = local_mean(src, radius);
    let h = src.nrows();
    let w = src.ncols();
    let one = T::one();
    let zero = T::zero();
    let mut out = DynMatrix::<T>::zeros(h, w);
    for j in 0..w {
        for i in 0..h {
            out[(i, j)] = if src[(i, j)] > mean[(i, j)] + offset { one } else { zero };
        }
    }
    out
}
