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
//!
//! The plan ([`SoaPlan`]: twiddles only, read-only during a transform) is
//! separate from the scratch ([`SoaScratch`]: the deinterleaved re/im buffers),
//! so one plan can drive many transforms concurrently — each worker brings its
//! own scratch. This is what lets the 2D transforms batch over rows/columns
//! under `rayon` while sharing a single set of twiddles.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use num_complex::Complex;

use super::cast;
use crate::simd::fft_butterfly_dispatch;
use crate::traits::FloatScalar;

/// Precomputed twiddles for a power-of-two SoA transform of length `n`.
///
/// Immutable after construction; pair it with a [`SoaScratch`] of the same
/// length to transform.
pub(crate) struct SoaPlan<T: FloatScalar> {
    n: usize,
    /// Start index of each stage's twiddle block within `stage_wr`/`stage_wi`.
    /// Stages of length 2 and 4 are fused into a twiddle-free pass, so the
    /// first entry belongs to the length-8 stage.
    offsets: Vec<usize>,
    /// Per-stage contiguous twiddles `wr[k] = cos(-2πk/len)`, concatenated.
    stage_wr: Vec<T>,
    /// Per-stage contiguous twiddles `wi[k] = sin(-2πk/len)`, concatenated.
    stage_wi: Vec<T>,
}

/// Deinterleaved real/imaginary work buffers for an [`SoaPlan`] of the same
/// length.
pub(crate) struct SoaScratch<T: FloatScalar> {
    re: Vec<T>,
    im: Vec<T>,
}

impl<T: FloatScalar> SoaScratch<T> {
    /// Length this scratch was built for.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.re.len()
    }
}

impl<T: FloatScalar> SoaPlan<T> {
    /// Build the plan for power-of-two length `n`.
    pub(crate) fn new(n: usize) -> Self {
        debug_assert!(n.is_power_of_two());
        let mut offsets = Vec::new();
        let mut stage_wr = Vec::new();
        let mut stage_wi = Vec::new();

        // Stages of length 2 and 4 need no stored twiddles (w ∈ {1, −i}).
        let mut len = 8;
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
        }
    }

    /// Allocate a scratch buffer sized for this plan.
    pub(crate) fn scratch(&self) -> SoaScratch<T> {
        SoaScratch {
            re: vec![T::zero(); self.n],
            im: vec![T::zero(); self.n],
        }
    }

    /// Transform `buf` in place (`buf.len() == n`). `inverse` selects the
    /// `exp(+2πi kn/N)` kernel and scales by `1/N`.
    ///
    /// The inverse reuses the forward stages via `ifft(x) = conj(fft(conj(x)))/N`,
    /// with both conjugations folded into the deinterleave / interleave copies
    /// so it costs no extra pass over the data.
    pub(crate) fn transform(
        &self,
        buf: &mut [Complex<T>],
        scratch: &mut SoaScratch<T>,
        inverse: bool,
    ) {
        debug_assert_eq!(buf.len(), self.n);
        debug_assert_eq!(scratch.len(), self.n);
        let n = self.n;
        let SoaScratch { re, im } = scratch;

        // Deinterleave into re/im, applying the bit-reversal permutation up front
        // (and the input conjugation for the inverse).
        deinterleave_bitrev(buf, re, im, inverse);

        // Fused length-2 + length-4 stages: twiddle-free, one pass.
        stages_2_4(re, im);

        let mut len = 8;
        let mut stage = 0;
        while len <= n {
            let half = len / 2;
            let off = self.offsets[stage];
            let wr = &self.stage_wr[off..off + half];
            let wi = &self.stage_wi[off..off + half];

            for (r, i) in re.chunks_exact_mut(len).zip(im.chunks_exact_mut(len)) {
                let (tr, br) = r.split_at_mut(half);
                let (ti, bi) = i.split_at_mut(half);
                fft_butterfly_dispatch(tr, ti, br, bi, wr, wi);
            }
            len <<= 1;
            stage += 1;
        }

        if inverse {
            let inv_n = T::one() / cast::<T>(n as f64);
            for (z, (&r, &i)) in buf.iter_mut().zip(re.iter().zip(im.iter())) {
                *z = Complex::new(r * inv_n, -i * inv_n);
            }
        } else {
            for (z, (&r, &i)) in buf.iter_mut().zip(re.iter().zip(im.iter())) {
                *z = Complex::new(r, i);
            }
        }
    }
}

/// Copy `buf` into `re`/`im` with the bit-reversal permutation applied, so the
/// subsequent decimation-in-time stages read naturally ordered data. With
/// `conj` the imaginary parts are negated on the way in.
fn deinterleave_bitrev<T: FloatScalar>(buf: &[Complex<T>], re: &mut [T], im: &mut [T], conj: bool) {
    let n = buf.len();
    // j walks the bit-reversed index sequence as i counts up.
    let mut j = 0usize;
    for (i, z) in buf.iter().enumerate() {
        re[j] = z.re;
        im[j] = if conj { -z.im } else { z.im };
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

/// The first two decimation-in-time stages (butterfly lengths 2 and 4) fused
/// into one pass over the arrays.
///
/// Both stages have trivial twiddles — `w = 1` for length 2, `w ∈ {1, −i}` for
/// length 4 — so they need neither a table lookup nor a complex multiply. Running
/// them through the generic per-block butterfly would cost a kernel call per
/// 1- or 2-element block (below any SIMD width, so scalar anyway); fusing them
/// turns `n` tiny calls into one tight loop the compiler can vectorize directly.
///
/// For `n == 2` only the length-2 stage applies; `n == 1` is a no-op.
fn stages_2_4<T: FloatScalar>(re: &mut [T], im: &mut [T]) {
    let n = re.len();
    if n < 4 {
        if n == 2 {
            let (a, b) = (re[0], re[1]);
            re[0] = a + b;
            re[1] = a - b;
            let (a, b) = (im[0], im[1]);
            im[0] = a + b;
            im[1] = a - b;
        }
        return;
    }
    for (r, i) in re.chunks_exact_mut(4).zip(im.chunks_exact_mut(4)) {
        // Length-2 stage on (0,1) and (2,3).
        let (a0r, a1r) = (r[0] + r[1], r[0] - r[1]);
        let (a0i, a1i) = (i[0] + i[1], i[0] - i[1]);
        let (a2r, a3r) = (r[2] + r[3], r[2] - r[3]);
        let (a2i, a3i) = (i[2] + i[3], i[2] - i[3]);
        // Length-4 stage: out0 = a0 + a2, out2 = a0 − a2 (w = 1);
        // v = a3 · (−i) = (a3.im, −a3.re); out1 = a1 + v, out3 = a1 − v.
        r[0] = a0r + a2r;
        i[0] = a0i + a2i;
        r[2] = a0r - a2r;
        i[2] = a0i - a2i;
        let (vr, vi) = (a3i, -a3r);
        r[1] = a1r + vr;
        i[1] = a1i + vi;
        r[3] = a1r - vr;
        i[3] = a1i - vi;
    }
}
