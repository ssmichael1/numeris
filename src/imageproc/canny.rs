use alloc::vec;
use alloc::vec::Vec;

use crate::dynmatrix::DynMatrix;
use crate::traits::FloatScalar;

use super::border::BorderMode;
use super::filters::{gaussian_blur, sobel_gradients};

/// Canny edge detector — the classic five-stage pipeline:
///
/// 1. Gaussian smooth with `sigma`
/// 2. Sobel gradients → magnitude + direction
/// 3. Non-maximum suppression along the gradient direction (4-quadrant
///    quantization)
/// 4. Double threshold: pixels above `high` become **strong** edges, pixels
///    in `[low, high]` become **weak**, below `low` are suppressed
/// 5. Hysteresis: weak pixels 8-connected to any strong pixel are promoted
///    to edges; the rest are dropped
///
/// Returns a binary edge map with values in `{0, 1}`.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
/// use numeris::imageproc::{canny, BorderMode};
///
/// // A vertical step from 0 to 1.
/// let mut img = DynMatrix::<f64>::zeros(16, 16);
/// for i in 0..16 { for j in 8..16 { img[(i, j)] = 1.0; } }
/// let edges = canny(&img, 1.0, 0.05, 0.15, BorderMode::Replicate);
/// // A connected edge is found along the step (approx column 7 or 8).
/// let centre_has_edge = edges[(8, 7)] > 0.0 || edges[(8, 8)] > 0.0;
/// assert!(centre_has_edge);
/// ```
pub fn canny<T: FloatScalar>(
    src: &DynMatrix<T>,
    sigma: T,
    low_threshold: T,
    high_threshold: T,
    border: BorderMode<T>,
) -> DynMatrix<T> {
    let h = src.nrows();
    let w = src.ncols();
    let zero = T::zero();
    let one = T::one();

    if h == 0 || w == 0 {
        return DynMatrix::<T>::zeros(h, w);
    }

    // 1. Smooth.
    let smoothed = gaussian_blur(src, sigma, border);

    // 2. Gradients.
    let (gx, gy) = sobel_gradients(&smoothed, border);

    // 3. Non-maximum suppression along quantized gradient direction.
    //    The direction is quantized into 4 bins at 22.5° boundaries.
    let mut mag = DynMatrix::<T>::zeros(h, w);
    for j in 0..w {
        for i in 0..h {
            let a = gx[(i, j)];
            let b = gy[(i, j)];
            mag[(i, j)] = (a * a + b * b).sqrt();
        }
    }

    let half = T::from(0.5_f64).unwrap();
    let tan_22_5 = T::from(0.41421356237309503_f64).unwrap(); // tan(22.5°)
    let tan_67_5 = T::from(2.414213562373095_f64).unwrap(); // tan(67.5°)

    let mut nms = DynMatrix::<T>::zeros(h, w);
    for j in 1..w.saturating_sub(1) {
        for i in 1..h.saturating_sub(1) {
            let m = mag[(i, j)];
            if m == zero {
                continue;
            }
            let dx = gx[(i, j)];
            let dy = gy[(i, j)];
            // Quantize direction by ratio of |dy| / |dx|, using signs to pick
            // the quadrant of the two neighbours.
            let adx = dx.abs();
            let ady = dy.abs();

            let (n1, n2) = if ady <= tan_22_5 * adx {
                // Mostly horizontal → compare to (i, j-1), (i, j+1).
                (mag[(i, j - 1)], mag[(i, j + 1)])
            } else if ady >= tan_67_5 * adx {
                // Mostly vertical → compare to (i-1, j), (i+1, j).
                (mag[(i - 1, j)], mag[(i + 1, j)])
            } else if (dx > zero) == (dy > zero) {
                // Diagonal ↘ / ↖ → compare to (i-1, j-1), (i+1, j+1).
                (mag[(i - 1, j - 1)], mag[(i + 1, j + 1)])
            } else {
                // Diagonal ↗ / ↙ → compare to (i-1, j+1), (i+1, j-1).
                (mag[(i - 1, j + 1)], mag[(i + 1, j - 1)])
            };

            // Use ≥ with a tiny bias to keep ties resolvable.
            let eps = T::epsilon() * m * half;
            if m + eps >= n1 && m + eps >= n2 {
                nms[(i, j)] = m;
            }
        }
    }

    // 4. Double threshold → strong/weak classification.
    //    2 = strong, 1 = weak, 0 = suppressed.
    let mut cls: Vec<u8> = vec![0; h * w];
    let mut strong_seeds: Vec<(usize, usize)> = Vec::new();
    for j in 0..w {
        for i in 0..h {
            let v = nms[(i, j)];
            if v >= high_threshold {
                cls[j * h + i] = 2;
                strong_seeds.push((i, j));
            } else if v >= low_threshold {
                cls[j * h + i] = 1;
            }
        }
    }

    // 5. Hysteresis via DFS: every weak pixel 8-connected to a strong
    //    seed becomes strong.
    let mut stack = strong_seeds;
    while let Some((i, j)) = stack.pop() {
        for di in -1_isize..=1 {
            for dj in -1_isize..=1 {
                if di == 0 && dj == 0 {
                    continue;
                }
                let ni = i as isize + di;
                let nj = j as isize + dj;
                if ni < 0 || nj < 0 || ni >= h as isize || nj >= w as isize {
                    continue;
                }
                let idx = (nj as usize) * h + (ni as usize);
                if cls[idx] == 1 {
                    cls[idx] = 2;
                    stack.push((ni as usize, nj as usize));
                }
            }
        }
    }

    // Produce binary output.
    let mut out = DynMatrix::<T>::zeros(h, w);
    for j in 0..w {
        for i in 0..h {
            out[(i, j)] = if cls[j * h + i] == 2 { one } else { zero };
        }
    }
    out
}
