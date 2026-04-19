use crate::dynmatrix::DynMatrix;
use crate::traits::{MatrixMut, MatrixRef, Scalar};

use super::border::{fetch_border, BorderMode};

/// Mirror horizontally (left-right flip).
pub fn flip_horizontal<T: Scalar>(src: &DynMatrix<T>) -> DynMatrix<T> {
    let h = src.nrows();
    let w = src.ncols();
    let mut out = DynMatrix::<T>::zeros(h, w);
    for j in 0..w {
        let src_col = src.col_as_slice(w - 1 - j, 0);
        let dst_col = out.col_as_mut_slice(j, 0);
        dst_col.copy_from_slice(src_col);
    }
    out
}

/// Mirror vertically (top-bottom flip).
pub fn flip_vertical<T: Scalar>(src: &DynMatrix<T>) -> DynMatrix<T> {
    let h = src.nrows();
    let w = src.ncols();
    let mut out = DynMatrix::<T>::zeros(h, w);
    for j in 0..w {
        let src_col = src.col_as_slice(j, 0);
        let dst_col = out.col_as_mut_slice(j, 0);
        for i in 0..h {
            dst_col[i] = src_col[h - 1 - i];
        }
    }
    out
}

/// Rotate 90° clockwise. An `H × W` image becomes `W × H`.
pub fn rotate_90<T: Scalar>(src: &DynMatrix<T>) -> DynMatrix<T> {
    // out[j, h-1-i] = src[i, j]
    let h = src.nrows();
    let w = src.ncols();
    let mut out = DynMatrix::<T>::zeros(w, h);
    for i in 0..h {
        for j in 0..w {
            out[(j, h - 1 - i)] = src[(i, j)];
        }
    }
    out
}

/// Rotate 180°. Composition of horizontal and vertical flips.
pub fn rotate_180<T: Scalar>(src: &DynMatrix<T>) -> DynMatrix<T> {
    let h = src.nrows();
    let w = src.ncols();
    let mut out = DynMatrix::<T>::zeros(h, w);
    for j in 0..w {
        let src_col = src.col_as_slice(j, 0);
        let dst_col = out.col_as_mut_slice(w - 1 - j, 0);
        for i in 0..h {
            dst_col[i] = src_col[h - 1 - i];
        }
    }
    out
}

/// Rotate 90° counter-clockwise (equivalent to 270° clockwise).
pub fn rotate_270<T: Scalar>(src: &DynMatrix<T>) -> DynMatrix<T> {
    // out[w-1-j, i] = src[i, j]
    let h = src.nrows();
    let w = src.ncols();
    let mut out = DynMatrix::<T>::zeros(w, h);
    for i in 0..h {
        for j in 0..w {
            out[(w - 1 - j, i)] = src[(i, j)];
        }
    }
    out
}

/// Pad an image by `top`, `bottom`, `left`, `right` pixels using the given
/// border mode to synthesize values outside the original extent.
///
/// Output dimensions are `(H + top + bottom) × (W + left + right)`.
pub fn pad<T: Scalar>(
    src: &DynMatrix<T>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    let h_in = src.nrows();
    let w_in = src.ncols();
    let h_out = h_in + top + bottom;
    let w_out = w_in + left + right;
    let mut out = DynMatrix::<T>::zeros(h_out, w_out);
    for j_out in 0..w_out {
        let src_j = j_out as isize - left as isize;
        let dst_col = out.col_as_mut_slice(j_out, 0);
        if src_j >= 0 && (src_j as usize) < w_in {
            let src_col = src.col_as_slice(src_j as usize, 0);
            for i_out in 0..h_out {
                let src_i = i_out as isize - top as isize;
                dst_col[i_out] = fetch_vertical(src_col, src_i, h_in, border);
            }
        } else {
            // Entire column is outside the image — fetch from a synthesized
            // column via the border rule (applied on both axes).
            for i_out in 0..h_out {
                let src_i = i_out as isize - top as isize;
                dst_col[i_out] = fetch_both_axes(src, src_i, src_j, border);
            }
        }
    }
    out
}

/// Crop a rectangular region `[top, top + height) × [left, left + width)`.
///
/// # Panics
///
/// Panics if the region extends past the image extent.
pub fn crop<T: Scalar>(
    src: &DynMatrix<T>,
    top: usize,
    left: usize,
    height: usize,
    width: usize,
) -> DynMatrix<T> {
    assert!(
        top + height <= src.nrows() && left + width <= src.ncols(),
        "crop region out of bounds"
    );
    let mut out = DynMatrix::<T>::zeros(height, width);
    for j in 0..width {
        let src_col = src.col_as_slice(left + j, top);
        let dst_col = out.col_as_mut_slice(j, 0);
        dst_col.copy_from_slice(&src_col[..height]);
    }
    out
}

/// Resize `src` to `(new_rows, new_cols)` using **nearest-neighbor**
/// interpolation. The value of each output pixel is copied from the
/// nearest input pixel without blending — essential for masks, label maps,
/// and other discrete-valued images where bilinear would corrupt values.
pub fn resize_nearest<T: Scalar>(
    src: &DynMatrix<T>,
    new_rows: usize,
    new_cols: usize,
) -> DynMatrix<T> {
    let h_in = src.nrows();
    let w_in = src.ncols();
    let mut out = DynMatrix::<T>::zeros(new_rows, new_cols);
    if h_in == 0 || w_in == 0 || new_rows == 0 || new_cols == 0 {
        return out;
    }
    // Precompute per-axis nearest-pixel mappings via integer arithmetic
    // (pixel-center convention). For output index `i_out`, we want
    //   round((i_out + 0.5) * h_in / h_out - 0.5)
    // = round(((2·i_out + 1)·h_in − h_out) / (2·h_out)),
    // clamped to [0, h_in − 1]. Done in isize to avoid std/libm dependency.
    let mut i_src = alloc::vec![0usize; new_rows];
    let mut j_src = alloc::vec![0usize; new_cols];
    let h_in_i = h_in as isize;
    let w_in_i = w_in as isize;
    let h_out_i = new_rows as isize;
    let w_out_i = new_cols as isize;
    for i_out in 0..new_rows {
        let num = (2 * i_out as isize + 1) * h_in_i - h_out_i;
        let idx = if num <= 0 {
            0
        } else {
            let rounded = (num + h_out_i) / (2 * h_out_i);
            rounded.min(h_in_i - 1) as usize
        };
        i_src[i_out] = idx;
    }
    for j_out in 0..new_cols {
        let num = (2 * j_out as isize + 1) * w_in_i - w_out_i;
        let idx = if num <= 0 {
            0
        } else {
            let rounded = (num + w_out_i) / (2 * w_out_i);
            rounded.min(w_in_i - 1) as usize
        };
        j_src[j_out] = idx;
    }
    for j_out in 0..new_cols {
        let src_col = src.col_as_slice(j_src[j_out], 0);
        let dst_col = out.col_as_mut_slice(j_out, 0);
        for i_out in 0..new_rows {
            dst_col[i_out] = src_col[i_src[i_out]];
        }
    }
    out
}

// ── internal helpers ──────────────────────────────────────────────────

#[inline]
fn fetch_vertical<T: Scalar>(col: &[T], idx: isize, n: usize, border: BorderMode<T>) -> T {
    if idx >= 0 && (idx as usize) < n {
        col[idx as usize]
    } else {
        fetch_border(col, idx, border)
    }
}

#[inline]
fn fetch_both_axes<T: Scalar>(
    src: &DynMatrix<T>,
    i: isize,
    j: isize,
    border: BorderMode<T>,
) -> T {
    let h = src.nrows() as isize;
    let w = src.ncols() as isize;
    if i >= 0 && i < h && j >= 0 && j < w {
        return src[(i as usize, j as usize)];
    }
    match border {
        BorderMode::Zero => T::zero(),
        BorderMode::Constant(c) => c,
        BorderMode::Replicate => {
            let ii = i.clamp(0, h - 1) as usize;
            let jj = j.clamp(0, w - 1) as usize;
            src[(ii, jj)]
        }
        BorderMode::Reflect => {
            let ii = reflect(i, h);
            let jj = reflect(j, w);
            src[(ii, jj)]
        }
    }
}

#[inline]
fn reflect(idx: isize, n: isize) -> usize {
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
