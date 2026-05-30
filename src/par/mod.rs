//! Internal parallelism dispatch.
//!
//! This module is private — it provides a single API that compiles to
//! sequential iteration by default and to [Rayon](https://docs.rs/rayon)
//! work-stealing parallelism under the `rayon` feature. It mirrors the `simd/`
//! module: algorithms call these helpers unconditionally, and the feature flag
//! lives only here, so the sequential and parallel code paths cannot drift
//! apart.
//!
//! ## Scope
//!
//! Rayon requires `std`, so the `rayon` feature implies `std` and is purely
//! additive — `no_std`/embedded builds never see it. Parallelism is applied
//! only on heap-backed, runtime-sized paths (`DynMatrix` / `imageproc` /
//! `_dyn` routines) where the work is large enough to amortize thread
//! dispatch. Small fixed-size `Matrix` operations stay sequential.
//!
//! ## Determinism
//!
//! These helpers operate on **disjoint** mutable chunks — writes never alias,
//! so the result is independent of execution order and of the thread count.
//! (Parallel *reductions*, where summation order would change the
//! floating-point result, are intentionally not provided here; add them with a
//! fixed-block decomposition so they stay bit-for-bit reproducible.)

/// Element bound required to use a type in the parallel kernels here.
///
/// With the `rayon` feature this is `Send + Sync` — element data and the shared
/// immutable inputs (e.g. the source image) are read from multiple worker
/// threads. Without the feature it is an empty bound implemented for *every*
/// type, so non-`rayon` builds carry no extra restriction at all.
///
/// Bounding an algorithm by `T: FloatScalar + MaybeSync` lets a single public
/// signature serve both configurations: the bound vanishes unless parallelism
/// is actually compiled in, and for the real element types (`f32`/`f64`) it is
/// satisfied automatically when it does apply. This avoids duplicating every
/// parallelized routine into `cfg`-gated twins.
#[cfg(feature = "rayon")]
#[doc(hidden)]
pub trait MaybeSync: Send + Sync {}
#[cfg(feature = "rayon")]
impl<T: Send + Sync> MaybeSync for T {}

#[cfg(not(feature = "rayon"))]
#[doc(hidden)]
pub trait MaybeSync {}
#[cfg(not(feature = "rayon"))]
impl<T> MaybeSync for T {}

/// Apply `f(j, chunk)` to each disjoint `chunk_len`-sized chunk of `data`.
///
/// Chunk `j` is `data[j*chunk_len .. (j+1)*chunk_len]`. `data.len()` must be a
/// multiple of `chunk_len` (true for column-major matrix storage: one chunk per
/// column). Under the `rayon` feature the chunks run in parallel when there are
/// at least `threshold` of them; otherwise — and always without the feature —
/// they run sequentially.
///
/// Chunks are non-overlapping, so the closure's writes never alias and the
/// outcome does not depend on execution order. The `rayon` variant requires
/// `Fn + Sync + Send` (the closure is shared across threads); the sequential
/// variant accepts `FnMut`, so the same algorithm body type-checks under both
/// configurations with only the public trait bound differing by `cfg`.
#[cfg(feature = "rayon")]
#[inline]
pub(crate) fn for_each_chunk_mut<T, F>(data: &mut [T], chunk_len: usize, threshold: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    use rayon::prelude::*;
    debug_assert!(chunk_len > 0);
    let nchunks = data.len() / chunk_len;
    if nchunks >= threshold {
        data.par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(j, chunk)| f(j, chunk));
    } else {
        for (j, chunk) in data.chunks_mut(chunk_len).enumerate() {
            f(j, chunk);
        }
    }
}

#[cfg(not(feature = "rayon"))]
#[inline]
pub(crate) fn for_each_chunk_mut<T, F>(data: &mut [T], chunk_len: usize, _threshold: usize, mut f: F)
where
    F: FnMut(usize, &mut [T]),
{
    debug_assert!(chunk_len > 0);
    for (j, chunk) in data.chunks_mut(chunk_len).enumerate() {
        f(j, chunk);
    }
}
