use alloc::vec;
use alloc::vec::Vec;

use crate::dynmatrix::DynMatrix;
use crate::traits::{FloatScalar, MatrixMut, MatrixRef};

/// Resize `src` to `(new_rows, new_cols)` using bilinear interpolation.
///
/// Uses the "pixel-center" coordinate convention (matching OpenCV's default
/// for `INTER_LINEAR` with `align_corners = false`): input-space coordinates
/// are `y_in = (i_out + 0.5) · H_in / H_out − 0.5` and analogously for `x`,
/// clamped to the valid range so edge pixels replicate.
///
/// Per-axis interpolation indices and fractional weights are precomputed
/// into tables, and the inner loop runs per output column with direct
/// column-slice access into `src` (contiguous in the column-major layout).
/// The loop is trivially auto-vectorizable — no gather intrinsics required.
///
/// Returns an empty matrix if either output dimension is zero or `src` is
/// empty.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
/// use numeris::imageproc::resize_bilinear;
///
/// // Upscale a 2×2 gradient to 4×4.
/// let img = DynMatrix::from_rows(2, 2, &[0.0_f64, 1.0, 2.0, 3.0]);
/// let up = resize_bilinear(&img, 4, 4);
/// assert_eq!(up.nrows(), 4);
/// assert_eq!(up.ncols(), 4);
/// ```
pub fn resize_bilinear<T: FloatScalar>(
    src: &DynMatrix<T>,
    new_rows: usize,
    new_cols: usize,
) -> DynMatrix<T> {
    let h_in = src.nrows();
    let w_in = src.ncols();
    let mut dst = DynMatrix::<T>::zeros(new_rows, new_cols);
    if h_in == 0 || w_in == 0 || new_rows == 0 || new_cols == 0 {
        return dst;
    }

    let half = T::from(0.5_f64).unwrap();
    let h_in_t = T::from(h_in).unwrap();
    let w_in_t = T::from(w_in).unwrap();
    let h_out_t = T::from(new_rows).unwrap();
    let w_out_t = T::from(new_cols).unwrap();
    let sy = h_in_t / h_out_t;
    let sx = w_in_t / w_out_t;
    let max_r = h_in - 1;
    let max_c = w_in - 1;

    // Precompute per-row (i0, i1, ty) and per-column (j0, j1, tx) tables.
    let mut i0s: Vec<usize> = vec![0; new_rows];
    let mut i1s: Vec<usize> = vec![0; new_rows];
    let mut tys: Vec<T> = vec![T::zero(); new_rows];
    for i_out in 0..new_rows {
        let y = (T::from(i_out).unwrap() + half) * sy - half;
        let (i0, i1, ty) = map_axis(y, max_r);
        i0s[i_out] = i0;
        i1s[i_out] = i1;
        tys[i_out] = ty;
    }
    let mut j0s: Vec<usize> = vec![0; new_cols];
    let mut j1s: Vec<usize> = vec![0; new_cols];
    let mut txs: Vec<T> = vec![T::zero(); new_cols];
    for j_out in 0..new_cols {
        let x = (T::from(j_out).unwrap() + half) * sx - half;
        let (j0, j1, tx) = map_axis(x, max_c);
        j0s[j_out] = j0;
        j1s[j_out] = j1;
        txs[j_out] = tx;
    }

    // Outer over output columns (each written contiguously in column-major
    // storage); inner over output rows. Source columns j0 and j1 are fetched
    // once as contiguous slices.
    for j_out in 0..new_cols {
        let j0 = j0s[j_out];
        let j1 = j1s[j_out];
        let tx = txs[j_out];
        let src_j0 = src.col_as_slice(j0, 0);
        let src_j1 = src.col_as_slice(j1, 0);
        let dst_col = dst.col_as_mut_slice(j_out, 0);

        // Tight inner loop — pure indexed loads + FMA. Compiler auto-vectorizes.
        for i_out in 0..new_rows {
            let i0 = i0s[i_out];
            let i1 = i1s[i_out];
            let ty = tys[i_out];
            let a = src_j0[i0];
            let b = src_j1[i0];
            let c = src_j0[i1];
            let d = src_j1[i1];
            let top = a + (b - a) * tx;
            let bot = c + (d - c) * tx;
            dst_col[i_out] = top + (bot - top) * ty;
        }
    }
    dst
}

/// Map a continuous coordinate `u` into a pair of bracketing integer indices
/// in `[0, max]` and the fractional weight on the upper side.
#[inline]
fn map_axis<T: FloatScalar>(u: T, max: usize) -> (usize, usize, T) {
    let zero = T::zero();
    let max_t = T::from(max).unwrap();
    let u_clamped = if u < zero {
        zero
    } else if u > max_t {
        max_t
    } else {
        u
    };
    let i0_t = u_clamped.floor();
    let i0 = i0_t.to_usize().unwrap_or(0).min(max);
    let i1 = (i0 + 1).min(max);
    let t = u_clamped - i0_t;
    (i0, i1, t)
}
