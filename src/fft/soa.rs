//! Structure-of-arrays (deinterleaved) power-of-two FFT for the `DynFft` tier.
//!
//! The interleaved `radix2_inline` core in [`super::radix`] is the no-alloc
//! reference. For the heap-backed [`DynFft`](super::DynFft) tier we can afford to
//! split the complex buffer into separate real/imaginary arrays; the butterfly
//! then reduces to plain elementwise ops over contiguous `[T]` slices, which the
//! SIMD kernels in [`crate::simd`] vectorize (deinterleaved re/im maps directly
//! onto SIMD lanes — see `docs/design-fft.md`).
//!
//! This module owns the orchestration (deinterleave, bit-reversal, staged
//! twiddles, block iteration); the per-lane arithmetic lives in
//! [`crate::simd::fft_butterfly_dispatch`], which falls back to a scalar loop
//! identical to the reference.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use num_complex::Complex;

use super::cast;
use crate::simd::fft_butterfly_dispatch;
use crate::traits::FloatScalar;

/// Precomputed plan + scratch for a power-of-two SoA transform of length `n`.
pub(crate) struct SoaPlan<T: FloatScalar> {
    n: usize,
    /// Start index of each stage's twiddle block within `stage_wr`/`stage_wi`.
    offsets: Vec<usize>,
    /// Per-stage contiguous twiddles `wr[k] = cos(-2πk/len)`, concatenated.
    stage_wr: Vec<T>,
    /// Per-stage contiguous twiddles `wi[k] = sin(-2πk/len)`, concatenated.
    stage_wi: Vec<T>,
    /// Deinterleaved real scratch.
    re: Vec<T>,
    /// Deinterleaved imaginary scratch.
    im: Vec<T>,
}

impl<T: FloatScalar> SoaPlan<T> {
    /// Build the plan for power-of-two length `n`.
    pub(crate) fn new(n: usize) -> Self {
        debug_assert!(n.is_power_of_two());
        let mut offsets = Vec::new();
        let mut stage_wr = Vec::new();
        let mut stage_wi = Vec::new();

        let mut len = 2;
        while len <= n {
            let half = len / 2;
            offsets.push(stage_wr.len());
            for k in 0..half {
                // Forward twiddle for this stage: exp(-2πi k / len).
                let ang = cast::<T>(-core::f64::consts::TAU * (k as f64) / (len as f64));
                stage_wr.push(ang.cos());
                stage_wi.push(ang.sin());
            }
            len <<= 1;
        }

        Self {
            n,
            offsets,
            stage_wr,
            stage_wi,
            re: vec![T::zero(); n],
            im: vec![T::zero(); n],
        }
    }

    /// Forward FFT in place over `buf` (`buf.len() == n`, a power of two).
    pub(crate) fn forward(&mut self, buf: &mut [Complex<T>]) {
        debug_assert_eq!(buf.len(), self.n);
        let Self {
            n,
            offsets,
            stage_wr,
            stage_wi,
            re,
            im,
        } = self;
        let n = *n;

        // Deinterleave into re/im, applying the bit-reversal permutation up front.
        deinterleave_bitrev(buf, re, im);

        let mut len = 2;
        let mut stage = 0;
        while len <= n {
            let half = len / 2;
            let off = offsets[stage];
            let wr = &stage_wr[off..off + half];
            let wi = &stage_wi[off..off + half];

            let mut base = 0;
            while base < n {
                let (tr, br) = re[base..base + len].split_at_mut(half);
                let (ti, bi) = im[base..base + len].split_at_mut(half);
                fft_butterfly_dispatch(tr, ti, br, bi, wr, wi);
                base += len;
            }
            len <<= 1;
            stage += 1;
        }

        for (z, (&r, &i)) in buf.iter_mut().zip(re.iter().zip(im.iter())) {
            *z = Complex::new(r, i);
        }
    }
}

/// Copy `buf` into `re`/`im` with the bit-reversal permutation applied, so the
/// subsequent decimation-in-time stages read naturally ordered data.
fn deinterleave_bitrev<T: FloatScalar>(buf: &[Complex<T>], re: &mut [T], im: &mut [T]) {
    let n = buf.len();
    // j walks the bit-reversed index sequence as i counts up.
    let mut j = 0usize;
    for (i, z) in buf.iter().enumerate() {
        re[j] = z.re;
        im[j] = z.im;
        // Advance j to the next bit-reversed value.
        if i + 1 < n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
        }
    }
}
