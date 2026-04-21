use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::dynmatrix::DynMatrix;
use crate::traits::{FloatScalar, MatrixMut, MatrixRef};

use super::border::BorderMode;

/// Sliding rank filter: each output pixel is the `rank`-th smallest value in
/// the `(2·radius + 1) × (2·radius + 1)` window centered on it.
///
/// `rank` is a 0-indexed order statistic. With window size `K = (2r+1)²`:
/// - `rank = 0` — minimum (erosion-like)
/// - `rank = K - 1` — maximum (dilation-like)
/// - `rank = K / 2` — median
///
/// Implemented with `slice::select_nth_unstable_by` (expected `O(K)` per
/// pixel), not a full sort. A single heap buffer is reused across all output
/// pixels, so there is no per-pixel allocation.
///
/// Median is **not separable**, so unlike Gaussian blur this cannot be
/// decomposed into 1D passes. Expect `O(H · W · K)` cost; large radii are
/// slow.
///
/// # Panics
///
/// Panics (debug) if `rank ≥ (2·radius + 1)²`.
/// Any NaN in a window will panic via `partial_cmp().unwrap()`; if your input
/// may contain NaNs, clean them up first.
pub fn rank_filter<T: FloatScalar>(
    src: &DynMatrix<T>,
    radius: usize,
    rank: usize,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    let nrows = src.nrows();
    let ncols = src.ncols();
    let mut dst = DynMatrix::<T>::zeros(nrows, ncols);
    if nrows == 0 || ncols == 0 {
        return dst;
    }
    if radius == 0 {
        return src.clone();
    }

    let k_side = 2 * radius + 1;
    let k_total = k_side * k_side;
    debug_assert!(rank < k_total, "rank {rank} out of range for window size {k_total}");

    let r = radius as isize;
    let mut buf: Vec<T> = Vec::with_capacity(k_total);

    for j in 0..ncols {
        for i in 0..nrows {
            buf.clear();
            for dj in -r..=r {
                let sj = j as isize + dj;
                for di in -r..=r {
                    let si = i as isize + di;
                    buf.push(super::convolve::fetch_border_2d(src, si, sj, border));
                }
            }
            // Quickselect: partitions the buffer so buf[rank] is the k-th smallest.
            buf.select_nth_unstable_by(rank, |a, b| {
                a.partial_cmp(b).unwrap_or(Ordering::Equal)
            });
            dst[(i, j)] = buf[rank];
        }
    }
    dst
}

/// Sliding percentile filter. `percentile` is clamped to `[0, 1]`.
///
/// `percentile = 0.0` → local minimum, `0.5` → median, `1.0` → local maximum.
/// For window size `K`, the rank used is `floor(percentile · (K − 1))`.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
/// use numeris::imageproc::{percentile_filter, BorderMode};
///
/// // Suppress salt-and-pepper noise by taking the 25th percentile in a 5×5
/// // window — robust to impulse noise brighter than the signal.
/// let img = DynMatrix::<f64>::zeros(16, 16);
/// let bg = percentile_filter(&img, 2, 0.25, BorderMode::Replicate);
/// assert_eq!(bg.nrows(), 16);
/// ```
pub fn percentile_filter<T: FloatScalar>(
    src: &DynMatrix<T>,
    radius: usize,
    percentile: T,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    if radius == 0 {
        return src.clone();
    }
    let zero = T::zero();
    let one = T::one();
    let p = if percentile < zero {
        zero
    } else if percentile > one {
        one
    } else {
        percentile
    };
    let k_total = (2 * radius + 1) * (2 * radius + 1);
    let k_minus_1 = T::from(k_total - 1).unwrap();
    let rank_f = (p * k_minus_1).floor();
    let rank = rank_f.to_usize().unwrap_or(0).min(k_total - 1);
    rank_filter(src, radius, rank, border)
}

/// Sliding median filter: each output pixel is the median of its
/// `(2·radius + 1) × (2·radius + 1)` window.
///
/// Dispatches to fast specializations at `radius = 1` (3×3) and `radius = 2`
/// (5×5), which split the image into an interior path with inlined
/// contiguous-column gathers and stack-allocated window buffers, and a
/// border path that uses the generic border-aware fetch. For larger radii
/// the generic [`rank_filter`] (quickselect) is used.
///
/// Excellent for salt-and-pepper noise removal; preserves edges unlike
/// Gaussian blur.
pub fn median_filter<T: FloatScalar>(
    src: &DynMatrix<T>,
    radius: usize,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    match radius {
        0 => src.clone(),
        1 => median_3x3(src, border),
        2 => median_5x5(src, border),
        r => {
            let k_total = (2 * r + 1) * (2 * r + 1);
            rank_filter(src, r, k_total / 2, border)
        }
    }
}

/// 3×3 median filter (radius = 1), with split interior/border loops and a
/// stack-allocated 9-element window buffer. ~2–3× faster than the generic
/// quickselect path on large images.
fn median_3x3<T: FloatScalar>(src: &DynMatrix<T>, border: BorderMode<T>) -> DynMatrix<T> {
    let nrows = src.nrows();
    let ncols = src.ncols();
    let mut dst = DynMatrix::<T>::zeros(nrows, ncols);
    if nrows == 0 || ncols == 0 {
        return dst;
    }
    if nrows < 3 || ncols < 3 {
        // Image too small for an interior region — fall through to the
        // generic border-aware path for every pixel.
        let r = 1usize;
        let k_total = 9;
        return rank_filter(src, r, k_total / 2, border);
    }

    // Interior: [1, nrows-1) × [1, ncols-1).
    for j in 1..ncols - 1 {
        let col_m = src.col_as_slice(j - 1, 0);
        let col_c = src.col_as_slice(j, 0);
        let col_p = src.col_as_slice(j + 1, 0);
        let dst_col = dst.col_as_mut_slice(j, 0);
        for i in 1..nrows - 1 {
            let mut w: [T; 9] = [
                col_m[i - 1], col_c[i - 1], col_p[i - 1],
                col_m[i],     col_c[i],     col_p[i],
                col_m[i + 1], col_c[i + 1], col_p[i + 1],
            ];
            w.select_nth_unstable_by(4, |a, b| {
                a.partial_cmp(b).unwrap_or(Ordering::Equal)
            });
            dst_col[i] = w[4];
        }
    }

    // Border: every pixel outside the interior rectangle, via fetch_border.
    let border_pixel = |i: usize, j: usize| -> T {
        let mut w: [T; 9] = [T::zero(); 9];
        let mut idx = 0;
        for di in -1_isize..=1 {
            for dj in -1_isize..=1 {
                w[idx] = super::convolve::fetch_border_2d(
                    src,
                    i as isize + di,
                    j as isize + dj,
                    border,
                );
                idx += 1;
            }
        }
        w.select_nth_unstable_by(4, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        w[4]
    };
    for j in 0..ncols {
        dst[(0, j)] = border_pixel(0, j);
        dst[(nrows - 1, j)] = border_pixel(nrows - 1, j);
    }
    for i in 1..nrows - 1 {
        dst[(i, 0)] = border_pixel(i, 0);
        dst[(i, ncols - 1)] = border_pixel(i, ncols - 1);
    }
    dst
}

/// 5×5 median filter (radius = 2) — same interior/border split as
/// [`median_3x3`] with a 25-element stack window.
fn median_5x5<T: FloatScalar>(src: &DynMatrix<T>, border: BorderMode<T>) -> DynMatrix<T> {
    let nrows = src.nrows();
    let ncols = src.ncols();
    let mut dst = DynMatrix::<T>::zeros(nrows, ncols);
    if nrows == 0 || ncols == 0 {
        return dst;
    }
    if nrows < 5 || ncols < 5 {
        let r = 2usize;
        let k_total = 25;
        return rank_filter(src, r, k_total / 2, border);
    }

    for j in 2..ncols - 2 {
        let c0 = src.col_as_slice(j - 2, 0);
        let c1 = src.col_as_slice(j - 1, 0);
        let c2 = src.col_as_slice(j, 0);
        let c3 = src.col_as_slice(j + 1, 0);
        let c4 = src.col_as_slice(j + 2, 0);
        let dst_col = dst.col_as_mut_slice(j, 0);
        for i in 2..nrows - 2 {
            let mut w: [T; 25] = [
                c0[i - 2], c1[i - 2], c2[i - 2], c3[i - 2], c4[i - 2],
                c0[i - 1], c1[i - 1], c2[i - 1], c3[i - 1], c4[i - 1],
                c0[i],     c1[i],     c2[i],     c3[i],     c4[i],
                c0[i + 1], c1[i + 1], c2[i + 1], c3[i + 1], c4[i + 1],
                c0[i + 2], c1[i + 2], c2[i + 2], c3[i + 2], c4[i + 2],
            ];
            w.select_nth_unstable_by(12, |a, b| {
                a.partial_cmp(b).unwrap_or(Ordering::Equal)
            });
            dst_col[i] = w[12];
        }
    }

    // Border rows/columns: generic border-aware gather.
    let border_pixel = |i: usize, j: usize| -> T {
        let mut w: [T; 25] = [T::zero(); 25];
        let mut idx = 0;
        for di in -2_isize..=2 {
            for dj in -2_isize..=2 {
                w[idx] = super::convolve::fetch_border_2d(
                    src,
                    i as isize + di,
                    j as isize + dj,
                    border,
                );
                idx += 1;
            }
        }
        w.select_nth_unstable_by(12, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        w[12]
    };
    for j in 0..ncols {
        for i in 0..2 {
            dst[(i, j)] = border_pixel(i, j);
            dst[(nrows - 1 - i, j)] = border_pixel(nrows - 1 - i, j);
        }
    }
    for i in 2..nrows - 2 {
        for j in 0..2 {
            dst[(i, j)] = border_pixel(i, j);
            dst[(i, ncols - 1 - j)] = border_pixel(i, ncols - 1 - j);
        }
    }
    dst
}

/// Fast sliding median filter for `u16` images using Huang's sliding
/// histogram (1979).
///
/// Complexity is `O(H · W · radius)` — independent of the window area — by
/// maintaining a 65 536-bin histogram of the current window, decrementing the
/// leaving column's pixels and incrementing the entering column's pixels as
/// the window slides horizontally. A running count tracks how many window
/// pixels are strictly below the current median so the median value only
/// walks a handful of bins per step.
///
/// Suitable for any data quantized to ≤ 16 bits (8-, 10-, 12-, 14-, 16-bit).
/// For float images, quantize to `u16` first; for 8-bit, cast up via
/// `DynMatrix::from_fn(h, w, |i, j| u8_img[(i, j)] as u16)`.
///
/// Memory: a single 256 KB histogram reused across all pixels (plus 4·W
/// bytes of running state), not per row.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
/// use numeris::imageproc::{median_filter_u16, BorderMode};
///
/// // A uniform 12-bit image with a single saturated pixel — Huang's median
/// // recovers the background exactly.
/// let mut img = DynMatrix::<u16>::fill(9, 9, 800);
/// img[(4, 4)] = 4095;
/// let out = median_filter_u16(&img, 1, BorderMode::Replicate);
/// assert_eq!(out[(4, 4)], 800);
/// ```
pub fn median_filter_u16(
    src: &DynMatrix<u16>,
    radius: usize,
    border: BorderMode<u16>,
) -> DynMatrix<u16> {
    let nrows = src.nrows();
    let ncols = src.ncols();
    let mut dst = DynMatrix::<u16>::zeros(nrows, ncols);
    if nrows == 0 || ncols == 0 {
        return dst;
    }
    if radius == 0 {
        return src.clone();
    }

    const L: usize = 65536;
    let r = radius as isize;
    let k_total = (2 * radius + 1) * (2 * radius + 1);
    let target = (k_total / 2) as u32;

    let mut hist: Vec<u32> = alloc::vec![0u32; L];

    for i in 0..nrows {
        // Reset histogram and seed the window at column 0.
        for h in hist.iter_mut() {
            *h = 0;
        }
        for dj in -r..=r {
            let sj = dj; // j = 0
            for di in -r..=r {
                let si = i as isize + di;
                let v = super::convolve::fetch_border_2d(src, si, sj, border) as usize;
                hist[v] += 1;
            }
        }
        // Initial median: walk from 0 until cumulative count exceeds target.
        let mut cum: u32 = 0;
        let mut m: usize = 0;
        while cum + hist[m] <= target {
            cum += hist[m];
            m += 1;
        }
        dst[(i, 0)] = m as u16;

        // Slide horizontally. Invariant: cum = #{v in window : v < m}.
        for j in 1..ncols {
            let sj_out = j as isize - 1 - r;
            let sj_in = j as isize + r;
            for di in -r..=r {
                let si = i as isize + di;
                let v_out = super::convolve::fetch_border_2d(src, si, sj_out, border) as usize;
                let v_in = super::convolve::fetch_border_2d(src, si, sj_in, border) as usize;
                hist[v_out] -= 1;
                if v_out < m {
                    cum -= 1;
                }
                hist[v_in] += 1;
                if v_in < m {
                    cum += 1;
                }
            }
            // Re-anchor m so cum ≤ target < cum + hist[m].
            while cum > target {
                m -= 1;
                cum -= hist[m];
            }
            while cum + hist[m] <= target {
                cum += hist[m];
                m += 1;
            }
            dst[(i, j)] = m as u16;
        }
    }
    dst
}
