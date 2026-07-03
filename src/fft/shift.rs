//! Spectrum-centering helpers (`fftshift` / `ifftshift`).
//!
//! No-alloc, in-place, and generic over any element type — they are pure
//! rotations of the buffer (`slice::rotate_*`, which needs no scratch).

/// Shift the zero-frequency component to the center of the spectrum.
///
/// Equivalent to NumPy's `fftshift`: rotates the buffer right by `n/2`. For odd
/// lengths the larger half moves to the front. In-place, no allocation.
#[inline]
pub fn fftshift<T>(buf: &mut [T]) {
    let k = buf.len() / 2;
    buf.rotate_right(k);
}

/// Inverse of [`fftshift`]: move the center component back to index 0.
///
/// Equivalent to NumPy's `ifftshift` (rotate left by `n/2`). For even lengths it
/// is identical to [`fftshift`]; for odd lengths it is the true inverse.
#[inline]
pub fn ifftshift<T>(buf: &mut [T]) {
    let k = buf.len() / 2;
    buf.rotate_left(k);
}
