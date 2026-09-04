//! Fixed-size, no-alloc, in-place FFT entry points (power-of-two `N ≤ 4096`).

use num_complex::Complex;

use super::radix;
use super::twiddle::TwiddleTable;
use crate::traits::FloatScalar;

/// Compile-time precondition check shared by the table-free entry points.
macro_rules! assert_valid_n {
    ($N:expr) => {
        const {
            assert!(
                $N.is_power_of_two() && $N <= 4096,
                "fft: N must be a power of two and <= 4096",
            );
        }
    };
}

/// In-place forward FFT using a precomputed [`TwiddleTable`].
///
/// No `sin`/`cos` runs in the loop — prefer this for repeated same-`N`
/// transforms. Uses the `exp(-2πi k n / N)` sign convention.
#[inline]
pub fn fft<T: FloatScalar, const N: usize>(buf: &mut [Complex<T>; N], tw: &TwiddleTable<T, N>) {
    radix::radix2_table(&mut buf[..], tw.as_slice(), false);
}

/// In-place inverse FFT using a precomputed [`TwiddleTable`], normalized by `1/N`.
#[inline]
pub fn ifft<T: FloatScalar, const N: usize>(buf: &mut [Complex<T>; N], tw: &TwiddleTable<T, N>) {
    radix::radix2_table(&mut buf[..], tw.as_slice(), true);
}

/// In-place forward FFT generating stage twiddles inline (no persistent table).
///
/// Lower persistent memory than [`fft`] (nothing held between calls); a handful
/// of `sin`/`cos` per call. Panics at compile time unless `N` is a power of two
/// and `N ≤ 4096`.
#[inline]
pub fn fft_inplace<T: FloatScalar, const N: usize>(buf: &mut [Complex<T>; N]) {
    assert_valid_n!(N);
    radix::radix2_inline(&mut buf[..], false);
}

/// In-place inverse FFT generating stage twiddles inline, normalized by `1/N`.
#[inline]
pub fn ifft_inplace<T: FloatScalar, const N: usize>(buf: &mut [Complex<T>; N]) {
    assert_valid_n!(N);
    radix::radix2_inline(&mut buf[..], true);
}
