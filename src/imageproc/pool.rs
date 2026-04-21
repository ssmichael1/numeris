use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::dynmatrix::DynMatrix;
use crate::traits::FloatScalar;

use super::resize::resize_bilinear;

/// Block-median decimation: partition `src` into non-overlapping
/// `block_size × block_size` tiles and emit one median per tile.
///
/// The output has shape `ceil(H / block_size) × ceil(W / block_size)`. Tiles
/// at the right/bottom edge that extend past the image take the median over
/// whatever pixels they cover (no border padding).
///
/// Cost: `O(H · W)` — each input pixel is visited exactly once. Quickselect
/// per tile is `O(block_size²)`. Dramatically faster than the sliding median
/// filter (no overlap cost), and — unlike the sliding version — memory
/// traffic is purely linear.
///
/// # Use cases
///
/// - Fast preview / thumbnail with outlier rejection
/// - Coarse background estimation (pair with [`median_pool_upsampled`] to
///   upsample back to the original resolution for subtraction)
///
/// # Panics
///
/// Panics if `block_size == 0`.
pub fn median_pool<T: FloatScalar>(src: &DynMatrix<T>, block_size: usize) -> DynMatrix<T> {
    assert!(block_size > 0, "block_size must be positive");
    let h_in = src.nrows();
    let w_in = src.ncols();
    if h_in == 0 || w_in == 0 {
        return DynMatrix::<T>::zeros(0, 0);
    }
    let h_out = h_in.div_ceil(block_size);
    let w_out = w_in.div_ceil(block_size);
    let mut dst = DynMatrix::<T>::zeros(h_out, w_out);
    let mut buf: Vec<T> = Vec::with_capacity(block_size * block_size);

    for bj in 0..w_out {
        let j0 = bj * block_size;
        let j1 = (j0 + block_size).min(w_in);
        for bi in 0..h_out {
            let i0 = bi * block_size;
            let i1 = (i0 + block_size).min(h_in);
            buf.clear();
            for j in j0..j1 {
                for i in i0..i1 {
                    buf.push(src[(i, j)]);
                }
            }
            let mid = buf.len() / 2;
            buf.select_nth_unstable_by(mid, |a, b| {
                a.partial_cmp(b).unwrap_or(Ordering::Equal)
            });
            dst[(bi, bj)] = buf[mid];
        }
    }
    dst
}

/// [`median_pool`] followed by bilinear upsampling back to the original
/// image size — a fast approximation to a full sliding-median background
/// estimate.
///
/// Output shape matches `src`. Useful for spatially smooth background
/// subtraction (e.g. star-tracker flat-field estimation): the block-median
/// rejects bright point sources, and the bilinear upsample gives a smooth
/// background map without the `O(H · W · block²)` cost of a true sliding
/// median.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
/// use numeris::imageproc::median_pool_upsampled;
///
/// let img = DynMatrix::<f64>::fill(64, 64, 100.0);
/// let bg = median_pool_upsampled(&img, 8);
/// assert_eq!(bg.nrows(), 64);
/// assert_eq!(bg.ncols(), 64);
/// ```
pub fn median_pool_upsampled<T: FloatScalar>(
    src: &DynMatrix<T>,
    block_size: usize,
) -> DynMatrix<T> {
    let pooled = median_pool(src, block_size);
    resize_bilinear(&pooled, src.nrows(), src.ncols())
}
