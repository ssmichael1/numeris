use crate::dynmatrix::DynMatrix;
use crate::simd;
use crate::traits::{FloatScalar, MatrixMut, MatrixRef, Scalar};

use super::border::{fetch_border, BorderMode};

/// 2D convolution (correlation convention: the kernel is **not** flipped).
///
/// Computes `out[i, j] = Σ_{ki, kj} kernel[ki, kj] · src[i + ki - hy, j + kj - hx]`
/// where `hy = (kernel.nrows() - 1) / 2` and `hx = (kernel.ncols() - 1) / 2`.
/// Out-of-bounds reads are resolved by `border`.
///
/// The kernel may be any type implementing [`MatrixRef`] (e.g. a fixed-size
/// [`Matrix`](crate::Matrix) or a [`DynMatrix`]). Kernel dimensions must be
/// odd and nonzero; a debug assertion is triggered otherwise.
///
/// For large separable kernels (Gaussian, box), prefer
/// [`convolve2d_separable`] — it runs in `O(h·w·(K_y + K_x))` instead of
/// `O(h·w·K_y·K_x)`.
///
/// # Panics
///
/// Panics (debug) if either kernel dimension is zero or even.
pub fn convolve2d<T: FloatScalar, K: MatrixRef<T>>(
    src: &DynMatrix<T>,
    kernel: &K,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    let ky = kernel.nrows();
    let kx = kernel.ncols();
    debug_assert!(ky > 0 && ky % 2 == 1, "kernel row count must be odd");
    debug_assert!(kx > 0 && kx % 2 == 1, "kernel col count must be odd");

    let nrows = src.nrows();
    let ncols = src.ncols();
    let hy = ky / 2;
    let hx = kx / 2;

    let mut dst = DynMatrix::<T>::zeros(nrows, ncols);
    if nrows == 0 || ncols == 0 {
        return dst;
    }

    // Iterate over each kernel tap and accumulate into dst via column-wise
    // AXPYs. For each tap (tky, tkx) the contribution to output pixel (i, j)
    // is kernel[tky, tkx] * src[i + (tky - hy), j + (tkx - hx)]. Rows where
    // the vertical shift stays in-bounds use a SIMD AXPY on contiguous column
    // slices; the top/bottom border rows fall back to a scalar border-aware
    // fetch.
    for tkx in 0..kx {
        let dx = tkx as isize - hx as isize;
        for tky in 0..ky {
            let dy = tky as isize - hy as isize;
            let w = *kernel.get(tky, tkx);
            if w == T::zero() {
                continue;
            }
            accumulate_shifted(&mut dst, src, w, dy, dx, border);
        }
    }
    dst
}

/// Separable 2D convolution: apply `kernel_y` along each column, then
/// `kernel_x` along each row.
///
/// Equivalent to [`convolve2d`] with the outer-product kernel
/// `kernel_y ⊗ kernel_x` but runs in `O(h·w·(K_y + K_x))` instead of
/// `O(h·w·K_y·K_x)`.
///
/// Both 1D kernels must have odd, nonzero length.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
/// use numeris::imageproc::{convolve2d_separable, gaussian_kernel_1d, BorderMode};
///
/// let img = DynMatrix::<f64>::fill(16, 16, 1.0);
/// let k = gaussian_kernel_1d::<f64>(1.0, 3.0).unwrap();
/// let blurred = convolve2d_separable(&img, &k, &k, BorderMode::Replicate);
/// // A uniform image convolved with a normalized kernel is unchanged.
/// for i in 0..16 {
///     for j in 0..16 {
///         assert!((blurred[(i, j)] - 1.0).abs() < 1e-12);
///     }
/// }
/// ```
pub fn convolve2d_separable<T: FloatScalar>(
    src: &DynMatrix<T>,
    kernel_y: &[T],
    kernel_x: &[T],
    border: BorderMode<T>,
) -> DynMatrix<T> {
    debug_assert!(
        !kernel_y.is_empty() && kernel_y.len() % 2 == 1,
        "kernel_y length must be odd and nonzero"
    );
    debug_assert!(
        !kernel_x.is_empty() && kernel_x.len() % 2 == 1,
        "kernel_x length must be odd and nonzero"
    );

    let tmp = convolve_1d_vertical(src, kernel_y, border);
    convolve_1d_horizontal(&tmp, kernel_x, border)
}

// ── internal helpers ──────────────────────────────────────────────────

/// Accumulate `w * src_shifted` into `dst`, where the shift is `(dy, dx)` in
/// (row, col). Interior rows use contiguous-column AXPY; border rows use the
/// border-aware scalar fallback.
fn accumulate_shifted<T: FloatScalar>(
    dst: &mut DynMatrix<T>,
    src: &DynMatrix<T>,
    w: T,
    dy: isize,
    dx: isize,
    border: BorderMode<T>,
) {
    let nrows = src.nrows();
    let ncols = src.ncols();
    let nrows_i = nrows as isize;

    // Output rows where `i + dy` stays inside [0, nrows).
    let i_lo = (-dy).max(0).min(nrows_i) as usize;
    let i_hi = (nrows_i - dy).clamp(0, nrows_i) as usize;

    for j in 0..ncols {
        let sj = j as isize + dx;
        let sj_in = sj >= 0 && (sj as usize) < ncols;

        if sj_in && i_hi > i_lo {
            let sj = sj as usize;
            // Source row range corresponding to output rows [i_lo, i_hi).
            let src_row_lo = (i_lo as isize + dy) as usize;
            let src_row_hi = (i_hi as isize + dy) as usize;
            let src_col = src.col_as_slice(sj, 0);
            let src_slice = &src_col[src_row_lo..src_row_hi];
            let dst_col = dst.col_as_mut_slice(j, 0);
            let dst_slice = &mut dst_col[i_lo..i_hi];
            simd::axpy_pos_dispatch(dst_slice, w, src_slice);
        }

        // Scalar border handling: either the top/bottom rows outside the AXPY
        // range, or the whole column when sj is out of bounds.
        let scalar_ranges: [(usize, usize); 2] = if sj_in {
            [(0, i_lo), (i_hi, nrows)]
        } else {
            [(0, nrows), (0, 0)]
        };
        for &(lo, hi) in &scalar_ranges {
            for i in lo..hi {
                let si = i as isize + dy;
                let v = fetch_border_2d(src, si, sj, border);
                dst[(i, j)] = dst[(i, j)] + w * v;
            }
        }
    }
}

/// 1D convolution along the vertical (row) axis, applied independently to
/// each column.
fn convolve_1d_vertical<T: FloatScalar>(
    src: &DynMatrix<T>,
    kernel: &[T],
    border: BorderMode<T>,
) -> DynMatrix<T> {
    let nrows = src.nrows();
    let ncols = src.ncols();
    let klen = kernel.len();
    let half = klen / 2;
    let mut dst = DynMatrix::<T>::zeros(nrows, ncols);
    if nrows == 0 || ncols == 0 {
        return dst;
    }

    for j in 0..ncols {
        // Interior AXPY: output rows where every kernel tap stays in-bounds.
        if nrows > 2 * half {
            let interior_len = nrows - 2 * half;
            let dst_col = dst.col_as_mut_slice(j, 0);
            let src_col_full = src.col_as_slice(j, 0);
            for k in 0..klen {
                let w = kernel[k];
                if w == T::zero() {
                    continue;
                }
                // Output row i (for i in [half, nrows-half)) reads source row
                // i + (k - half), i.e. source range [k, k + interior_len).
                let src_slice = &src_col_full[k..k + interior_len];
                let dst_slice = &mut dst_col[half..half + interior_len];
                simd::axpy_pos_dispatch(dst_slice, w, src_slice);
            }
        }

        // Border rows: scalar with border-aware fetch.
        let src_col = src.col_as_slice(j, 0);
        let border_top_hi = half.min(nrows);
        let border_bot_lo = nrows.saturating_sub(half).max(border_top_hi);
        for i in 0..border_top_hi {
            dst[(i, j)] = vertical_tap_sum(src_col, kernel, half, i, border);
        }
        for i in border_bot_lo..nrows {
            dst[(i, j)] = vertical_tap_sum(src_col, kernel, half, i, border);
        }
    }
    dst
}

#[inline]
fn vertical_tap_sum<T: FloatScalar>(
    src_col: &[T],
    kernel: &[T],
    half: usize,
    i: usize,
    border: BorderMode<T>,
) -> T {
    let mut sum = T::zero();
    for k in 0..kernel.len() {
        let si = i as isize + (k as isize - half as isize);
        let v = fetch_border(src_col, si, border);
        sum = sum + kernel[k] * v;
    }
    sum
}

/// 1D convolution along the horizontal (column) axis, applied independently
/// to each row. Implemented as whole-column AXPY between shifted columns —
/// contiguous memory access despite the axis name.
fn convolve_1d_horizontal<T: FloatScalar>(
    src: &DynMatrix<T>,
    kernel: &[T],
    border: BorderMode<T>,
) -> DynMatrix<T> {
    let nrows = src.nrows();
    let ncols = src.ncols();
    let klen = kernel.len();
    let half = klen / 2;
    let mut dst = DynMatrix::<T>::zeros(nrows, ncols);
    if nrows == 0 || ncols == 0 {
        return dst;
    }

    for j in 0..ncols {
        for k in 0..klen {
            let w = kernel[k];
            if w == T::zero() {
                continue;
            }
            let sj = j as isize + (k as isize - half as isize);
            if sj >= 0 && (sj as usize) < ncols {
                let src_slice = src.col_as_slice(sj as usize, 0);
                let dst_slice = dst.col_as_mut_slice(j, 0);
                simd::axpy_pos_dispatch(dst_slice, w, src_slice);
            } else {
                // Border column: apply border rule for every output row.
                for i in 0..nrows {
                    let v = fetch_border_2d(src, i as isize, sj, border);
                    dst[(i, j)] = dst[(i, j)] + w * v;
                }
            }
        }
    }
    dst
}

/// Fetch a 2D pixel with independent border handling on each axis.
#[inline]
pub(super) fn fetch_border_2d<T: Scalar>(
    src: &DynMatrix<T>,
    i: isize,
    j: isize,
    border: BorderMode<T>,
) -> T {
    let nrows = src.nrows() as isize;
    let ncols = src.ncols() as isize;
    if i >= 0 && i < nrows && j >= 0 && j < ncols {
        return src[(i as usize, j as usize)];
    }
    match border {
        BorderMode::Zero => T::zero(),
        BorderMode::Constant(c) => c,
        BorderMode::Replicate => {
            let ii = i.clamp(0, nrows - 1) as usize;
            let jj = j.clamp(0, ncols - 1) as usize;
            src[(ii, jj)]
        }
        BorderMode::Reflect => {
            let ii = reflect_index(i, nrows);
            let jj = reflect_index(j, ncols);
            src[(ii, jj)]
        }
    }
}

#[inline]
fn reflect_index(idx: isize, n: isize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut m = idx.rem_euclid(period);
    if m >= n {
        m = period - m;
    }
    m as usize
}
