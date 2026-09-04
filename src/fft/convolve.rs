//! FFT-based linear convolution and correlation of real signals and images
//! (requires `alloc`).
//!
//! All four functions zero-pad to the next power of two above the full output
//! size, so the transforms always take the SIMD radix path rather than
//! Bluestein, and use the real-input plans ([`DynRealFft`] / [`DynRealFft2`]) so
//! only the non-redundant half-spectrum is computed and multiplied.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use num_complex::Complex;

use super::fft2::DynRealFft2;
use super::real::DynRealFft;
use crate::dynmatrix::DynMatrix;
use crate::par::MaybeSync;
use crate::traits::FloatScalar;

/// Linear convolution `a ∗ b` via FFT.
///
/// Returns a vector of length `a.len() + b.len() − 1`. Empty if either input is
/// empty. For small kernels a direct O(n·m) convolution is faster; this pays off
/// once both operands are large.
pub fn fft_convolve<T: FloatScalar>(a: &[T], b: &[T]) -> Vec<T> {
    fft_convolve_impl(a, b, false)
}

/// Linear cross-correlation of `a` and `b` via FFT.
///
/// Defined as the convolution of `a` with the time-reversed `b`, so lag `k` of
/// the result is `Σ_n a[n] · b[n − (k − (b.len()−1))]`. Length
/// `a.len() + b.len() − 1`; empty if either input is empty.
pub fn fft_correlate<T: FloatScalar>(a: &[T], b: &[T]) -> Vec<T> {
    fft_convolve_impl(a, b, true)
}

fn fft_convolve_impl<T: FloatScalar>(a: &[T], b: &[T], reverse_b: bool) -> Vec<T> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let out_len = a.len() + b.len() - 1;
    // Pad to a power of two: a length-2ⁿ real FFT is far cheaper than Bluestein
    // at the exact length (which itself pads to ≥ 2·out_len internally).
    let n = out_len.next_power_of_two();

    let mut plan = DynRealFft::new(n);
    let mut pa = vec![T::zero(); n];
    let mut pb = vec![T::zero(); n];
    pa[..a.len()].copy_from_slice(a);
    if reverse_b {
        for (dst, &x) in pb.iter_mut().zip(b.iter().rev()) {
            *dst = x;
        }
    } else {
        pb[..b.len()].copy_from_slice(b);
    }

    let zero = Complex::new(T::zero(), T::zero());
    let mut sa = vec![zero; n / 2 + 1];
    let mut sb = vec![zero; n / 2 + 1];
    plan.forward(&pa, &mut sa);
    plan.forward(&pb, &mut sb);
    for (x, y) in sa.iter_mut().zip(&sb) {
        *x = *x * *y;
    }
    plan.inverse(&sa, &mut pa);
    pa.truncate(out_len);
    pa
}

/// Full 2D linear convolution `a ∗ b` of two real matrices via FFT.
///
/// The result is `(a.nrows() + b.nrows() − 1) × (a.ncols() + b.ncols() − 1)`
/// ("full" mode, zero boundary): element `(r, c)` is
/// `Σ a[i, j] · b[r − i, c − j]`. Returns an empty `0 × 0` matrix if either
/// input has a zero dimension.
///
/// This is `O(N log N)` in the padded size regardless of kernel size, so it wins
/// over spatial convolution (`imageproc::convolve2d`, `O(N·k²)`) once the kernel
/// is large — big-radius Gaussian/LoG kernels, template matching. For the
/// "same"-sized result centered on `a` that `imageproc` returns, crop the
/// `b.nrows()/2, b.ncols()/2` offset with `imageproc::crop` (or slice the
/// backing buffer directly).
///
/// # Examples
///
/// ```
/// # use numeris::fft::fft_convolve2d;
/// # use numeris::DynMatrix;
/// let a = DynMatrix::from_rows(2, 2, &[1.0f64, 2.0, 3.0, 4.0]);
/// let k = DynMatrix::from_rows(1, 2, &[1.0f64, 1.0]);
/// let c = fft_convolve2d(&a, &k); // 2 × 3
/// assert!((c[(0, 1)] - 3.0).abs() < 1e-12); // 1 + 2
/// ```
pub fn fft_convolve2d<T: FloatScalar + MaybeSync>(
    a: &DynMatrix<T>,
    b: &DynMatrix<T>,
) -> DynMatrix<T> {
    fft_convolve2d_impl(a, b, false)
}

/// Full 2D linear cross-correlation of `a` with `b` via FFT.
///
/// Convolution of `a` with `b` flipped along both axes; the same
/// `(ra + rb − 1) × (ca + cb − 1)` "full" output as [`fft_convolve2d`]. Lag
/// `(r, c)` of the result is `Σ a[i, j] · b[i − (r − (rb − 1)), j − (c − (cb − 1))]`,
/// so the peak of correlating an image with a template marks the template's
/// position offset by `(rb − 1, cb − 1)`. Empty if either input has a zero
/// dimension.
pub fn fft_correlate2d<T: FloatScalar + MaybeSync>(
    a: &DynMatrix<T>,
    b: &DynMatrix<T>,
) -> DynMatrix<T> {
    fft_convolve2d_impl(a, b, true)
}

fn fft_convolve2d_impl<T: FloatScalar + MaybeSync>(
    a: &DynMatrix<T>,
    b: &DynMatrix<T>,
    flip_b: bool,
) -> DynMatrix<T> {
    let (ra, ca) = (a.nrows(), a.ncols());
    let (rb, cb) = (b.nrows(), b.ncols());
    if ra == 0 || ca == 0 || rb == 0 || cb == 0 {
        return DynMatrix::zeros(0, 0);
    }
    let out_rows = ra + rb - 1;
    let out_cols = ca + cb - 1;
    let pr = out_rows.next_power_of_two();
    let pc = out_cols.next_power_of_two();

    // Zero-pad both operands into pr × pc (column-major: copy column by column).
    let mut pa = DynMatrix::<T>::zeros(pr, pc);
    for (dst, src) in pa
        .as_mut_slice()
        .chunks_exact_mut(pr)
        .zip(a.as_slice().chunks_exact(ra))
    {
        dst[..ra].copy_from_slice(src);
    }
    let mut pb = DynMatrix::<T>::zeros(pr, pc);
    if flip_b {
        for c in 0..cb {
            let src = &b.as_slice()[c * rb..(c + 1) * rb];
            let dst = &mut pb.as_mut_slice()[(cb - 1 - c) * pr..];
            for (d, &s) in dst[..rb].iter_mut().zip(src.iter().rev()) {
                *d = s;
            }
        }
    } else {
        for (dst, src) in pb
            .as_mut_slice()
            .chunks_exact_mut(pr)
            .zip(b.as_slice().chunks_exact(rb))
        {
            dst[..rb].copy_from_slice(src);
        }
    }

    let mut plan = DynRealFft2::new(pr, pc);
    let half = plan.spectrum_rows();
    let mut sa = DynMatrix::<Complex<T>>::zeros(half, pc);
    let mut sb = DynMatrix::<Complex<T>>::zeros(half, pc);
    plan.forward(&pa, &mut sa);
    plan.forward(&pb, &mut sb);
    for (x, y) in sa.as_mut_slice().iter_mut().zip(sb.as_slice()) {
        *x = *x * *y;
    }
    plan.inverse(&sa, &mut pa);

    // Crop the top-left out_rows × out_cols of the padded result.
    DynMatrix::from_fn(out_rows, out_cols, |r, c| pa[(r, c)])
}
