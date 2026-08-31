use crate::traits::Scalar;

/// How to handle pixel reads that fall outside the image extent.
///
/// Applied independently along each axis during 1D and 2D convolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BorderMode<T> {
    /// Out-of-bounds reads return zero.
    Zero,
    /// Out-of-bounds reads return a user-supplied constant value.
    Constant(T),
    /// Clamp the index to the nearest valid boundary (`aaa|abcd|ddd`).
    Replicate,
    /// Mirror the image about the boundary without duplicating the edge pixel
    /// (`cba|abcd|dcb`). Equivalent to OpenCV's `BORDER_REFLECT_101`.
    Reflect,
}

/// Fetch a value from a 1D slice using the given border mode.
///
/// `idx` may be negative or beyond `slice.len()`; this function resolves it
/// according to `border` and returns the corresponding value.
#[inline]
pub fn fetch_border<T: Scalar>(slice: &[T], idx: isize, border: BorderMode<T>) -> T {
    let n = slice.len() as isize;
    if idx >= 0 && idx < n {
        return slice[idx as usize];
    }
    match border {
        BorderMode::Zero => T::zero(),
        BorderMode::Constant(c) => c,
        BorderMode::Replicate => {
            if idx < 0 {
                slice[0]
            } else {
                slice[(n - 1) as usize]
            }
        }
        BorderMode::Reflect => {
            // Reflect without duplicating the edge: mirror about boundary.
            // Period is 2*(n-1). For n <= 1, fall back to clamping.
            if n <= 1 {
                return slice[0];
            }
            let period = 2 * (n - 1);
            let mut m = idx.rem_euclid(period);
            if m >= n {
                m = period - m;
            }
            slice[m as usize]
        }
    }
}

/// Resolve a possibly out-of-range index along one axis of length `n`.
///
/// Returns the in-range index to read for in-bounds `idx` and for the
/// index-remapping modes (`Replicate` clamps, `Reflect` mirrors about the
/// edge without duplicating it); `None` for the value-filling modes (`Zero`,
/// `Constant`), whose fill value the caller takes from `border`. This is the
/// single definition of the index rule that the 2D fetch and the strided
/// horizontal pass share.
#[inline]
pub(super) fn border_index<T>(idx: isize, n: isize, border: BorderMode<T>) -> Option<usize> {
    if idx >= 0 && idx < n {
        return Some(idx as usize);
    }
    match border {
        BorderMode::Zero | BorderMode::Constant(_) => None,
        BorderMode::Replicate => Some(idx.clamp(0, n - 1) as usize),
        BorderMode::Reflect => Some(reflect_index(idx, n)),
    }
}

/// Mirror `idx` into `[0, n)` about the edges without duplicating them
/// (period `2·(n − 1)`); for `n ≤ 1` every index maps to 0.
#[inline]
pub(super) fn reflect_index(idx: isize, n: isize) -> usize {
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
