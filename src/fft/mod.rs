//! Fast Fourier Transform.
//!
//! Pure-Rust, no-std-first FFT integrated with the crate's [`Complex`] and SIMD
//! support. **This is not FFTW.** FFTW's speed comes from runtime planning,
//! autotuned codelets, and a large C codebase that cannot run under `no_std` /
//! no-alloc — the target `numeris` was built for. The goal here is portable,
//! zero-C-dependency FFT that *also* runs on embedded, integrated with the rest
//! of the crate. On desktop, expect roughly 2–4× FFTW throughput; for the
//! audience that reaches for `numeris` (embedded DSP, no allocator) that is a
//! non-issue, because FFTW is not an option there.
//!
//! # Two tiers
//!
//! - **Fixed-size, no-alloc** ([`fft`], [`ifft`], [`fft_inplace`],
//!   [`ifft_inplace`]) — in-place transforms over `[Complex<T>; N]` for
//!   power-of-two `N ≤ 4096`. Works on the stack with no heap. [`fft`]/[`ifft`]
//!   take a precomputed [`TwiddleTable`] (no `sin`/`cos` in the hot loop);
//!   [`fft_inplace`]/[`ifft_inplace`] generate stage twiddles inline (lower
//!   persistent memory, no table to hold between calls).
//!
//! - **Runtime-sized** ([`DynFft`], requires `alloc`) — any length. Power-of-two
//!   lengths use the radix core directly; every other length (including primes)
//!   goes through Bluestein's algorithm, which reduces the DFT to power-of-two
//!   FFTs. Build the plan once, reuse across many transforms.
//!
//! - **Real input** ([`rfft`] / [`irfft`], and [`DynRealFft`] with `alloc`) —
//!   a length-`N` real signal has a Hermitian spectrum, so only the `N/2 + 1`
//!   non-redundant bins are returned. The forward transform packs the reals into
//!   `N/2` complex samples for roughly half the work of a full complex FFT.
//!
//! # Conventions
//!
//! Forward transform uses the `exp(-2πi k n / N)` sign convention; the inverse
//! is normalized by `1/N` so that `ifft(fft(x)) == x`.
//!
//! # Errors
//!
//! FFT length mismatches are programmer errors, so the transforms are infallible
//! and enforce their preconditions with compile-time (`const`) or debug asserts
//! rather than returning a `Result`. [`FftError`] is reserved for the fallible
//! runtime-sized planning surfaces (`DynFft::new`).
//!
//! # Examples
//!
//! ```
//! use numeris::fft::{fft_inplace, ifft_inplace};
//! use numeris::Complex;
//!
//! let mut buf = [
//!     Complex::new(1.0f64, 0.0),
//!     Complex::new(2.0, 0.0),
//!     Complex::new(3.0, 0.0),
//!     Complex::new(4.0, 0.0),
//! ];
//! fft_inplace(&mut buf);
//! // DC bin is the sum of the inputs.
//! assert!((buf[0].re - 10.0).abs() < 1e-12);
//! ifft_inplace(&mut buf);
//! assert!((buf[0].re - 1.0).abs() < 1e-12);
//! ```

use crate::traits::FloatScalar;

#[cfg(feature = "alloc")]
mod bluestein;
#[cfg(feature = "alloc")]
mod convolve;
#[cfg(feature = "alloc")]
mod dynfft;
#[cfg(feature = "alloc")]
mod fft2;
mod fixed;
mod radix;
mod real;
mod shift;
#[cfg(feature = "alloc")]
mod soa;
mod twiddle;

#[cfg(test)]
mod tests;

#[cfg(feature = "alloc")]
pub use convolve::{fft_convolve, fft_correlate};
#[cfg(feature = "alloc")]
pub use dynfft::DynFft;
#[cfg(feature = "alloc")]
pub use fft2::{fftshift2d, ifftshift2d, DynFft2, DynRealFft2};
pub use fixed::{fft, fft_inplace, ifft, ifft_inplace};
#[cfg(feature = "alloc")]
pub use real::DynRealFft;
pub use real::{irfft, rfft};
pub use shift::{fftshift, ifftshift};
pub use twiddle::TwiddleTable;

/// Errors from the runtime-sized FFT surfaces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FftError {
    /// A transform length of zero was requested.
    ZeroLength,
    /// A buffer length did not match the planned transform length.
    LengthMismatch,
}

impl core::fmt::Display for FftError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FftError::ZeroLength => write!(f, "transform length must be non-zero"),
            FftError::LengthMismatch => {
                write!(
                    f,
                    "buffer length does not match the planned transform length"
                )
            }
        }
    }
}

/// Cast an `f64` constant into the working float type `T`.
///
/// Uses [`num_traits::cast`] to avoid the `From`/`NumCast` ambiguity of
/// `T::from`. Constants used by the FFT (`2π`, lengths) are exactly representable
/// paths for `f32`/`f64`, so the `unwrap` never fires for the supported types.
#[inline]
pub(crate) fn cast<T: FloatScalar>(x: f64) -> T {
    num_traits::cast(x).expect("FFT constant is representable in the float type")
}
