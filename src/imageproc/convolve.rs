use crate::dynmatrix::DynMatrix;
use crate::simd;
use crate::traits::{FloatScalar, MatrixMut, MatrixRef, Scalar};

use super::border::{border_index, fetch_border, BorderMode};

/// Approximate per-pass work (element multiply-adds: `nrows · ncols · klen`)
/// below which a separable convolution pass runs sequentially. Above it, the
/// `rayon` feature fans the output columns out across threads.
///
/// Gating on *work* rather than column count alone accounts for image height
/// and kernel length: a wide-but-short image can be cheaper than a narrow-but-
/// tall one. The value sits just past the measured crossover for a small
/// Gaussian (see `bench/results.md`: 128² loses to thread overhead, 512² wins
/// ~2.6×).
const CONV_PAR_WORK_BUDGET: usize = 500_000;

/// Minimum number of output-column chunks before parallelizing, so a tall,
/// narrow image with heavy per-column work still gets enough chunks to spread
/// across cores without splitting into uselessly tiny pieces.
const CONV_PAR_MIN_COLS: usize = 8;

/// Column-count threshold for [`crate::par::for_each_chunk_mut`] derived from the
/// work budget: parallelize once `ncols` reaches `BUDGET / (nrows · klen)`,
/// floored at [`CONV_PAR_MIN_COLS`]. Used by the test-only whole-image passes;
/// the banded production path gates on per-band work directly.
#[cfg(test)]
#[inline]
fn conv_par_col_threshold(nrows: usize, klen: usize) -> usize {
    crate::par::work_col_threshold(
        nrows.saturating_mul(klen),
        CONV_PAR_WORK_BUDGET,
        CONV_PAR_MIN_COLS,
    )
}

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

/// Default band width (output columns) for the banded separable convolution.
///
/// Each band's vertical pass writes `band + K_x − 1` columns into a scratch
/// slab that the band's horizontal pass then reads; 64 columns keeps that
/// slab cache-resident for typical image heights (a 2048-row f32 image with
/// an 11-tap kernel is ≈ 600 KB) while paying only `(K_x − 1) / 64` extra
/// vertical work for the halo. Wider kernels widen the band to bound that
/// overhead — see [`separable_band`].
const SEPARABLE_BAND_COLS: usize = 64;

/// Band width for a separable convolution of an `ncols`-wide image with an
/// `klen_x`-tap horizontal kernel: the default, widened so the halo costs at
/// most 25 % extra vertical-pass work, clamped to the image width.
#[inline]
fn separable_band(ncols: usize, klen_x: usize) -> usize {
    SEPARABLE_BAND_COLS
        .max(4 * klen_x.saturating_sub(1))
        .min(ncols)
        .max(1)
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
/// Allocates the output image on every call; see
/// [`convolve2d_separable_into`] to write into a caller-owned buffer instead
/// (this function is a thin wrapper around it, so the two are bit-identical).
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
pub fn convolve2d_separable<T: FloatScalar + crate::par::MaybeSync>(
    src: &DynMatrix<T>,
    kernel_y: &[T],
    kernel_x: &[T],
    border: BorderMode<T>,
) -> DynMatrix<T> {
    // `zeros` goes through calloc — the pages are mapped lazily and first
    // touched by the passes themselves — so pre-sizing here costs no more
    // than letting `_into` grow an empty buffer, and the `_into` path then
    // sees a matching shape and does nothing extra.
    let mut dst = DynMatrix::<T>::zeros(src.nrows(), src.ncols());
    convolve2d_separable_into(src, kernel_y, kernel_x, border, &mut dst);
    dst
}

/// [`convolve2d_separable`] into a caller-owned output buffer.
///
/// `dst` is resized to `src`'s shape if it does not already match
/// (reallocating only when its capacity is insufficient) and its prior
/// contents are discarded, so a caller that filters many frames of the same
/// size reuses one output buffer across all of them. The intermediate
/// (vertical-pass) image is never materialized: the image is processed in
/// bands of output columns, each band's vertical pass running over the band
/// plus a kernel-half-width halo into a small scratch slab that its
/// horizontal pass then consumes. That slab — `(band + K_x − 1) · nrows`
/// elements, allocated once per call, or once per rayon job (not per band)
/// under `rayon` — is the only allocation, and it stays cache-resident
/// across the horizontal pass, which is why this is markedly faster than two
/// whole-image passes on large images even before counting the output
/// allocation it saves.
///
/// The output is bit-for-bit identical to [`convolve2d_separable`] (the
/// allocating function is a thin wrapper around this one) and independent of
/// the band width and thread count: every output column is computed by the
/// same per-column pass, reading the same values in the same order.
///
/// Both 1D kernels must have odd, nonzero length.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
/// use numeris::imageproc::{convolve2d_separable_into, gaussian_kernel_1d, BorderMode};
///
/// let k = gaussian_kernel_1d::<f32>(1.0, 3.0).unwrap();
/// let mut dst = DynMatrix::<f32>::zeros(0, 0);
/// for frame in 0..3 {
///     let img = DynMatrix::<f32>::fill(64, 48, frame as f32);
///     // First call sizes the buffer; later calls reuse it.
///     convolve2d_separable_into(&img, &k, &k, BorderMode::Replicate, &mut dst);
///     assert_eq!((dst.nrows(), dst.ncols()), (64, 48));
///     assert!((dst[(10, 10)] - frame as f32).abs() < 1e-6);
/// }
/// ```
pub fn convolve2d_separable_into<T: FloatScalar + crate::par::MaybeSync>(
    src: &DynMatrix<T>,
    kernel_y: &[T],
    kernel_x: &[T],
    border: BorderMode<T>,
    dst: &mut DynMatrix<T>,
) {
    let band = separable_band(src.ncols(), kernel_x.len());
    convolve2d_separable_into_banded(src, kernel_y, kernel_x, border, band, dst);
}

/// [`convolve2d_separable_into`] with an explicit band width — the
/// implementation behind it, exposed to the tests so the band-boundary logic
/// can be exercised at widths the heuristic would never pick (1, narrower
/// than the kernel, non-dividing, wider than the image).
///
/// `band` is clamped to `[1, ncols]`. Under the `rayon` feature the bands
/// run in parallel (above the same work gate as the column-parallel passes,
/// with at least [`CONV_PAR_MIN_COLS`] bands); each band writes only its own
/// disjoint columns of `dst`, so the result does not depend on the schedule.
pub(super) fn convolve2d_separable_into_banded<T: FloatScalar + crate::par::MaybeSync>(
    src: &DynMatrix<T>,
    kernel_y: &[T],
    kernel_x: &[T],
    border: BorderMode<T>,
    band: usize,
    dst: &mut DynMatrix<T>,
) {
    debug_assert_separable_kernels(kernel_y, kernel_x);
    let nrows = src.nrows();
    let ncols = src.ncols();
    dst.resize_discard(nrows, ncols);
    if nrows == 0 || ncols == 0 {
        return;
    }
    let band = band.max(1).min(ncols);
    let klen_x = kernel_x.len();
    let half_x = klen_x / 2;
    // Scratch per band: the vertical pass over `band + 2·half_x` columns
    // (clamped to the image).
    let halo_cols = (band + 2 * half_x).min(ncols);

    // Per-band work: vertical pass over the halo, horizontal pass over the
    // band; gate exactly like the column-parallel passes.
    let per_band_work = nrows
        .saturating_mul(kernel_y.len().saturating_mul(halo_cols) + klen_x.saturating_mul(band));
    let par_threshold =
        crate::par::work_col_threshold(per_band_work, CONV_PAR_WORK_BUDGET, CONV_PAR_MIN_COLS);

    crate::par::for_each_chunk_mut_init(
        dst.as_mut_slice(),
        band * nrows,
        par_threshold,
        || alloc::vec![T::zero(); halo_cols * nrows],
        |halo, b, dst_band| {
            let band_lo = b * band;
            let band_hi = band_lo + dst_band.len() / nrows;
            // Vertical-pass columns this band's horizontal taps can read
            // (in-bounds taps only; border-mapped taps land in the same range,
            // see `horizontal_pass_col`).
            let halo_lo = band_lo.saturating_sub(half_x);
            let halo_hi = (band_hi + half_x).min(ncols);
            let halo = &mut halo[..(halo_hi - halo_lo) * nrows];
            for (c, col) in halo.chunks_exact_mut(nrows).enumerate() {
                vertical_pass_col(src, halo_lo + c, kernel_y, border, col);
            }
            for (jj, dst_col) in dst_band.chunks_exact_mut(nrows).enumerate() {
                let j = band_lo + jj;
                horizontal_pass_col(halo, halo_lo, nrows, ncols, j, kernel_x, border, dst_col);
            }
        },
    );
}

/// Reference two-pass separable convolution: whole-image vertical pass into
/// `tmp`, then whole-image horizontal pass into `dst`, both through the same
/// per-column helpers the banded path uses. The banded implementation must
/// match this bit-for-bit; it exists only to pin that down in tests.
#[cfg(test)]
pub(super) fn convolve2d_separable_two_pass<T: FloatScalar + crate::par::MaybeSync>(
    src: &DynMatrix<T>,
    kernel_y: &[T],
    kernel_x: &[T],
    border: BorderMode<T>,
) -> DynMatrix<T> {
    debug_assert_separable_kernels(kernel_y, kernel_x);
    let mut tmp = DynMatrix::<T>::zeros(src.nrows(), src.ncols());
    let mut dst = DynMatrix::<T>::zeros(src.nrows(), src.ncols());
    convolve_1d_vertical_into(src, kernel_y, border, &mut tmp);
    convolve_1d_horizontal_into(&tmp, kernel_x, border, &mut dst);
    dst
}

#[inline]
fn debug_assert_separable_kernels<T>(kernel_y: &[T], kernel_x: &[T]) {
    debug_assert!(
        !kernel_y.is_empty() && kernel_y.len() % 2 == 1,
        "kernel_y length must be odd and nonzero"
    );
    debug_assert!(
        !kernel_x.is_empty() && kernel_x.len() % 2 == 1,
        "kernel_x length must be odd and nonzero"
    );
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

/// Whole-image 1D convolution along the vertical (row) axis, applied
/// independently to each column, written into `dst` (resized to `src`'s
/// shape, contents discarded). Reference path for
/// [`convolve2d_separable_two_pass`] — production code goes through the
/// banded [`convolve2d_separable_into_banded`], which runs the same
/// [`vertical_pass_col`] per column.
///
/// Each output column depends only on the matching source column, so with the
/// `rayon` feature the columns are computed in parallel over disjoint output
/// slices (above the [`conv_par_col_threshold`] work gate) — the result is
/// identical regardless of thread count.
#[cfg(test)]
fn convolve_1d_vertical_into<T: FloatScalar + crate::par::MaybeSync>(
    src: &DynMatrix<T>,
    kernel: &[T],
    border: BorderMode<T>,
    dst: &mut DynMatrix<T>,
) {
    let nrows = src.nrows();
    let ncols = src.ncols();
    dst.resize_discard(nrows, ncols);
    if nrows == 0 || ncols == 0 {
        return;
    }

    let par_threshold = conv_par_col_threshold(nrows, kernel.len());
    crate::par::for_each_chunk_mut(dst.as_mut_slice(), nrows, par_threshold, |j, dst_col| {
        vertical_pass_col(src, j, kernel, border, dst_col);
    });
}

/// Vertical pass for a single column: `dst_col[i] = Σ_k kernel[k] ·
/// src[i + k − half, j]` with `border` resolving rows outside the image.
/// `dst_col.len() == src.nrows()`.
///
/// Shared by the banded production pass and the test-only whole-image
/// reference pass so the two are bit-identical by construction.
#[inline]
fn vertical_pass_col<T: FloatScalar>(
    src: &DynMatrix<T>,
    j: usize,
    kernel: &[T],
    border: BorderMode<T>,
    dst_col: &mut [T],
) {
    let nrows = src.nrows();
    let half = kernel.len() / 2;
    debug_assert_eq!(dst_col.len(), nrows);

    // Interior rows (every kernel tap in-bounds): single traversal with
    // register-blocked accumulators — each output element is written once
    // and never re-read, unlike a per-tap AXPY sweep.
    if nrows > 2 * half {
        let interior_len = nrows - 2 * half;
        let src_col_full = src.col_as_slice(j, 0);
        // Output row i (for i in [half, nrows-half)) reads source rows
        // [i - half, i + half], i.e. window [i - half, i - half + klen).
        simd::conv1d_dispatch(
            &mut dst_col[half..half + interior_len],
            src_col_full,
            kernel,
            1,
        );
    }

    // Border rows: scalar with border-aware fetch.
    let src_col = src.col_as_slice(j, 0);
    let border_top_hi = half.min(nrows);
    let border_bot_lo = nrows.saturating_sub(half).max(border_top_hi);
    for (i, cell) in dst_col[..border_top_hi].iter_mut().enumerate() {
        *cell = vertical_tap_sum(src_col, kernel, half, i, border);
    }
    for (off, cell) in dst_col[border_bot_lo..].iter_mut().enumerate() {
        *cell = vertical_tap_sum(src_col, kernel, half, border_bot_lo + off, border);
    }
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

/// Whole-image 1D convolution along the horizontal (column) axis, applied
/// independently to each row, written into `dst` (resized to `src`'s shape,
/// contents discarded). Implemented as a strided tap sum across neighbouring
/// columns — contiguous memory access despite the axis name. Reference path
/// for [`convolve2d_separable_two_pass`]; production code runs the same
/// [`horizontal_pass_col`] per column over a band's halo slab.
///
/// Each output column reads only (immutably) shifted source columns and writes
/// its own disjoint output column, so with the `rayon` feature the output
/// columns are computed in parallel (above the [`conv_par_col_threshold`] work gate).
#[cfg(test)]
fn convolve_1d_horizontal_into<T: FloatScalar + crate::par::MaybeSync>(
    src: &DynMatrix<T>,
    kernel: &[T],
    border: BorderMode<T>,
    dst: &mut DynMatrix<T>,
) {
    let nrows = src.nrows();
    let ncols = src.ncols();
    dst.resize_discard(nrows, ncols);
    if nrows == 0 || ncols == 0 {
        return;
    }

    let par_threshold = conv_par_col_threshold(nrows, kernel.len());
    crate::par::for_each_chunk_mut(dst.as_mut_slice(), nrows, par_threshold, |j, dst_col| {
        horizontal_pass_col(src.as_slice(), 0, nrows, ncols, j, kernel, border, dst_col);
    });
}

/// Horizontal pass for a single output column `j`.
///
/// `buf` is a column-major slab of `nrows`-element columns holding the
/// vertical-pass image; its first column is image column `buf_col0`. It must
/// contain every column this output reads: the in-bounds taps
/// `[j − half, j + half] ∩ [0, ncols)`, plus — for `Replicate` / `Reflect` —
/// the columns the out-of-bounds taps map to. Both lie within
/// `[j − half, j + half] ∩ [0, ncols)` (clamping moves an index to the nearest
/// edge, which is on the near side of `j`; reflection about an edge within
/// `half` of `j` lands within `half` of `j`), so a slab covering that window
/// always suffices. The whole image with `buf_col0 = 0` trivially does.
///
/// Shared by the banded production pass and the test-only whole-image
/// reference pass so the two are bit-identical by construction: the
/// in-bounds taps are one fused [`simd::conv1d_dispatch`] sweep, then each
/// out-of-bounds tap is added non-fused in kernel order.
#[inline]
fn horizontal_pass_col<T: FloatScalar>(
    buf: &[T],
    buf_col0: usize,
    nrows: usize,
    ncols: usize,
    j: usize,
    kernel: &[T],
    border: BorderMode<T>,
    dst_col: &mut [T],
) {
    let klen = kernel.len();
    let half = klen / 2;
    debug_assert_eq!(dst_col.len(), nrows);

    // Taps whose source column j + (k - half) is in-bounds. Nonempty for
    // every j (the center tap k = half reads column j itself).
    let k_lo = half.saturating_sub(j).min(klen);
    let k_hi = (ncols + half - j).min(klen);

    // In-bounds taps: single traversal over the rows, with the tap sum
    // held in registers. Tap k reads source column j + k - half at the
    // same row, i.e. stride `nrows` between taps in column-major storage.
    let src_from_first_tap = &buf[(j + k_lo - half - buf_col0) * nrows..];
    simd::conv1d_dispatch(dst_col, src_from_first_tap, &kernel[k_lo..k_hi], nrows);

    // Out-of-bounds taps (only within `half` of the left/right edges):
    // apply the border rule for every output row.
    for k in (0..k_lo).chain(k_hi..klen) {
        let w = kernel[k];
        if w == T::zero() {
            continue;
        }
        let sj = j as isize + (k as isize - half as isize);
        match border_index(sj, ncols as isize, border) {
            Some(c) => {
                let col = &buf[(c - buf_col0) * nrows..(c - buf_col0 + 1) * nrows];
                for (cell, &v) in dst_col.iter_mut().zip(col) {
                    *cell = *cell + w * v;
                }
            }
            None => {
                let v = match border {
                    BorderMode::Constant(c) => c,
                    _ => T::zero(),
                };
                for cell in dst_col.iter_mut() {
                    *cell = *cell + w * v;
                }
            }
        }
    }
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
    match (
        border_index(i, nrows, border),
        border_index(j, ncols, border),
    ) {
        (Some(ii), Some(jj)) => src[(ii, jj)],
        _ => match border {
            BorderMode::Constant(c) => c,
            _ => T::zero(),
        },
    }
}
