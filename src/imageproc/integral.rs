use crate::dynmatrix::DynMatrix;
use crate::traits::{FloatScalar, MatrixRef};

/// Compute the summed-area table (SAT) of `src`.
///
/// Returns a `(nrows + 1) × (ncols + 1)` matrix where
/// `sat[i+1, j+1] = Σ_{r ≤ i, c ≤ j} src[r, c]` and the first row and column
/// are zero. The zero-padded layout means rectangle sums over any region
/// `[r0, r1) × [c0, c1)` of the original image reduce to a single
/// four-term expression; see [`integral_rect_sum`].
///
/// Complexity: `O(H·W)` to construct, `O(1)` per rectangle query.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
/// use numeris::imageproc::{integral_image, integral_rect_sum};
///
/// let img = DynMatrix::from_rows(
///     3, 3,
///     &[1.0_f64, 2.0, 3.0,
///       4.0,     5.0, 6.0,
///       7.0,     8.0, 9.0],
/// );
/// let sat = integral_image(&img);
/// // Sum over the whole 3×3 image.
/// assert!((integral_rect_sum(&sat, 0, 0, 3, 3) - 45.0).abs() < 1e-12);
/// // Sum over the centre pixel only.
/// assert!((integral_rect_sum(&sat, 1, 1, 2, 2) - 5.0).abs() < 1e-12);
/// ```
pub fn integral_image<T: FloatScalar>(src: &DynMatrix<T>) -> DynMatrix<T> {
    let h = src.nrows();
    let w = src.ncols();
    let mut sat = DynMatrix::<T>::zeros(h + 1, w + 1);

    // Running row sums, then cumulative column sums (two-pass SAT).
    for i in 0..h {
        let mut row_sum = T::zero();
        for j in 0..w {
            row_sum = row_sum + *src.get(i, j);
            // sat[i+1, j+1] = sat[i, j+1] + row_sum_up_to_j
            let above = sat[(i, j + 1)];
            sat[(i + 1, j + 1)] = above + row_sum;
        }
    }
    sat
}

/// Sum over the half-open rectangle `[r0, r1) × [c0, c1)` of the original
/// image, in O(1), using the SAT produced by [`integral_image`].
///
/// Indices are into the *original* image (not the padded SAT); the `+1`
/// offset is applied internally. Out-of-range indices are clamped to the
/// image extent (so `integral_rect_sum(sat, 0, 0, nrows, ncols)` sums the
/// whole image).
///
/// # Panics
///
/// Panics (debug) if `r0 > r1` or `c0 > c1`.
#[inline]
pub fn integral_rect_sum<T: FloatScalar>(
    sat: &DynMatrix<T>,
    r0: usize,
    c0: usize,
    r1: usize,
    c1: usize,
) -> T {
    debug_assert!(r0 <= r1 && c0 <= c1, "rectangle bounds must be ordered");
    let nrows = sat.nrows().saturating_sub(1);
    let ncols = sat.ncols().saturating_sub(1);
    let r0 = r0.min(nrows);
    let c0 = c0.min(ncols);
    let r1 = r1.min(nrows);
    let c1 = c1.min(ncols);
    // Inclusion–exclusion on the SAT's one-based indices.
    sat[(r1, c1)] + sat[(r0, c0)] - sat[(r1, c0)] - sat[(r0, c1)]
}
