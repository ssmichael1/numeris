use alloc::vec;
use alloc::vec::Vec;

use crate::dynmatrix::DynMatrix;
use crate::traits::{FloatScalar, MatrixMut, MatrixRef};

use super::border::{fetch_border, BorderMode};

/// Sliding **max** filter (grayscale morphological dilation) over a square
/// `(2·radius + 1) × (2·radius + 1)` window.
///
/// Implemented with the Van Herk – Gil-Werman algorithm: two 1D separable
/// passes, each of which computes sliding max in **O(1) amortized per output
/// pixel** regardless of radius, via forward and backward cumulative-maxima
/// over blocks of size `k = 2·radius + 1`. About 3 comparisons per pixel
/// total — often 50–100× faster than a naive per-window scan at moderate
/// radii.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
/// use numeris::imageproc::{max_filter, BorderMode};
///
/// let mut img = DynMatrix::<f64>::zeros(9, 9);
/// img[(4, 4)] = 1.0;
/// let out = max_filter(&img, 1, BorderMode::Zero);
/// // The 1.0 spreads to a 3×3 block around the original location.
/// for i in 3..=5 { for j in 3..=5 { assert_eq!(out[(i, j)], 1.0); } }
/// ```
pub fn max_filter<T: FloatScalar>(
    src: &DynMatrix<T>,
    radius: usize,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    filter_1d_then_1d(src, radius, border, max_of)
}

/// Sliding **min** filter (grayscale morphological erosion) over a square
/// `(2·radius + 1) × (2·radius + 1)` window. See [`max_filter`] for the
/// algorithm — same implementation with `max` replaced by `min`.
pub fn min_filter<T: FloatScalar>(
    src: &DynMatrix<T>,
    radius: usize,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    filter_1d_then_1d(src, radius, border, min_of)
}

/// Grayscale dilation — alias for [`max_filter`].
pub fn dilate<T: FloatScalar>(
    src: &DynMatrix<T>,
    radius: usize,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    max_filter(src, radius, border)
}

/// Grayscale erosion — alias for [`min_filter`].
pub fn erode<T: FloatScalar>(
    src: &DynMatrix<T>,
    radius: usize,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    min_filter(src, radius, border)
}

/// Morphological **opening**: erosion followed by dilation with the same
/// structuring element. Removes bright features smaller than the structuring
/// element while preserving the shape of larger ones.
pub fn opening<T: FloatScalar>(
    src: &DynMatrix<T>,
    radius: usize,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    let eroded = erode(src, radius, border);
    dilate(&eroded, radius, border)
}

/// Morphological **closing**: dilation followed by erosion. Fills dark holes
/// and gaps smaller than the structuring element.
pub fn closing<T: FloatScalar>(
    src: &DynMatrix<T>,
    radius: usize,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    let dilated = dilate(src, radius, border);
    erode(&dilated, radius, border)
}

/// Morphological **gradient**: `dilate(src) − erode(src)`. Highlights
/// boundaries — the width of the response scales with the structuring
/// element radius.
pub fn morphology_gradient<T: FloatScalar>(
    src: &DynMatrix<T>,
    radius: usize,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    let d = dilate(src, radius, border);
    let e = erode(src, radius, border);
    let nrows = src.nrows();
    let ncols = src.ncols();
    let mut out = DynMatrix::<T>::zeros(nrows, ncols);
    for j in 0..ncols {
        for i in 0..nrows {
            out[(i, j)] = d[(i, j)] - e[(i, j)];
        }
    }
    out
}

/// **Top-hat** transform: `src − opening(src)`. Isolates bright features
/// smaller than the structuring element — useful for point-source extraction
/// on a slowly-varying background.
pub fn top_hat<T: FloatScalar>(
    src: &DynMatrix<T>,
    radius: usize,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    let op = opening(src, radius, border);
    let nrows = src.nrows();
    let ncols = src.ncols();
    let mut out = DynMatrix::<T>::zeros(nrows, ncols);
    for j in 0..ncols {
        for i in 0..nrows {
            out[(i, j)] = src[(i, j)] - op[(i, j)];
        }
    }
    out
}

/// **Black-hat** transform: `closing(src) − src`. Isolates dark features
/// smaller than the structuring element.
pub fn black_hat<T: FloatScalar>(
    src: &DynMatrix<T>,
    radius: usize,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    let cl = closing(src, radius, border);
    let nrows = src.nrows();
    let ncols = src.ncols();
    let mut out = DynMatrix::<T>::zeros(nrows, ncols);
    for j in 0..ncols {
        for i in 0..nrows {
            out[(i, j)] = cl[(i, j)] - src[(i, j)];
        }
    }
    out
}

// ── internal ──────────────────────────────────────────────────────────

#[inline]
fn max_of<T: FloatScalar>(a: T, b: T) -> T {
    if a >= b {
        a
    } else {
        b
    }
}

#[inline]
fn min_of<T: FloatScalar>(a: T, b: T) -> T {
    if a <= b {
        a
    } else {
        b
    }
}

/// Apply a 1D Van Herk pass vertically (each column, contiguous in
/// column-major storage) then horizontally (each row, via a temp buffer).
fn filter_1d_then_1d<T: FloatScalar>(
    src: &DynMatrix<T>,
    radius: usize,
    border: BorderMode<T>,
    combine: fn(T, T) -> T,
) -> DynMatrix<T> {
    let nrows = src.nrows();
    let ncols = src.ncols();
    let mut out = DynMatrix::<T>::zeros(nrows, ncols);
    if nrows == 0 || ncols == 0 {
        return out;
    }
    if radius == 0 {
        return src.clone();
    }

    let k = 2 * radius + 1;
    // Scratch buffers — allocated once, reused across rows/columns.
    let mut padded: Vec<T> = Vec::with_capacity(nrows.max(ncols) + 2 * radius);
    let mut g: Vec<T> = vec![T::zero(); nrows.max(ncols) + 2 * radius];
    let mut h: Vec<T> = vec![T::zero(); nrows.max(ncols) + 2 * radius];

    // Pass 1 — vertical (contiguous in column-major storage).
    let mut tmp = DynMatrix::<T>::zeros(nrows, ncols);
    for j in 0..ncols {
        let src_col = src.col_as_slice(j, 0);
        let tmp_col = tmp.col_as_mut_slice(j, 0);
        van_herk_1d(src_col, tmp_col, radius, k, border, combine, &mut padded, &mut g, &mut h);
    }

    // Pass 2 — horizontal (strided row access, copy to/from row buffer).
    let mut row_in: Vec<T> = vec![T::zero(); ncols];
    let mut row_out: Vec<T> = vec![T::zero(); ncols];
    for i in 0..nrows {
        for j in 0..ncols {
            row_in[j] = tmp[(i, j)];
        }
        van_herk_1d(&row_in, &mut row_out, radius, k, border, combine, &mut padded, &mut g, &mut h);
        for j in 0..ncols {
            out[(i, j)] = row_out[j];
        }
    }
    out
}

/// 1D Van Herk / Gil-Werman sliding min or max (selected by `combine`).
///
/// `g[i]` holds the forward running combine up to `i` within the block of
/// size `k` containing `i`. `h[i]` holds the backward running combine.
/// Output at position `p` is `combine(h[p], g[p + 2r])` — either the two
/// cover the same (single) block, or they together cover the window that
/// straddles one block boundary.
#[allow(clippy::too_many_arguments)]
fn van_herk_1d<T: FloatScalar>(
    src: &[T],
    dst: &mut [T],
    radius: usize,
    k: usize,
    border: BorderMode<T>,
    combine: fn(T, T) -> T,
    padded: &mut Vec<T>,
    g: &mut Vec<T>,
    h: &mut Vec<T>,
) {
    let n = src.len();
    if n == 0 {
        return;
    }
    let total = n + 2 * radius;

    // Materialize a border-padded buffer of length n + 2r.
    padded.clear();
    for i in 0..total {
        let idx = i as isize - radius as isize;
        padded.push(fetch_border(src, idx, border));
    }

    if g.len() < total {
        g.resize(total, T::zero());
    }
    if h.len() < total {
        h.resize(total, T::zero());
    }

    // Forward cumulative combine within blocks of size k.
    for i in 0..total {
        if i % k == 0 {
            g[i] = padded[i];
        } else {
            g[i] = combine(g[i - 1], padded[i]);
        }
    }

    // Backward cumulative combine within blocks of size k.
    // Each block ends at index (i + 1) % k == 0, or at the final index.
    for i in (0..total).rev() {
        let is_block_end = (i + 1) % k == 0 || i == total - 1;
        if is_block_end {
            h[i] = padded[i];
        } else {
            h[i] = combine(h[i + 1], padded[i]);
        }
    }

    // Emit outputs. Output p uses padded window [p, p + 2r].
    for p in 0..n {
        dst[p] = combine(h[p], g[p + 2 * radius]);
    }
}
