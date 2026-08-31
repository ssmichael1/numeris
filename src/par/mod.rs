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
/// parallelized routine into `cfg`-gated twins. Only `imageproc` uses it (the
/// `optim` parallel routines name `Fn + Sync + Send` directly), so it is gated
/// on that feature.
#[cfg(all(feature = "imageproc", feature = "rayon"))]
#[doc(hidden)]
pub trait MaybeSync: Send + Sync {}
#[cfg(all(feature = "imageproc", feature = "rayon"))]
impl<T: Send + Sync> MaybeSync for T {}

#[cfg(all(feature = "imageproc", not(feature = "rayon")))]
#[doc(hidden)]
pub trait MaybeSync {}
#[cfg(all(feature = "imageproc", not(feature = "rayon")))]
impl<T> MaybeSync for T {}

/// Column-count threshold for [`for_each_chunk_mut`] derived from a work budget.
///
/// Parallelizing pays only when there is enough total work to amortize thread
/// dispatch. Gating on *work* rather than raw column count lets the decision
/// account for per-column cost (image height, window size, kernel length): the
/// returned threshold is `budget / per_col_work`, floored at `min_cols` so a
/// few very heavy columns still spread across cores without splitting into
/// uselessly tiny pieces. Pass the result as `for_each_chunk_mut`'s `threshold`.
///
/// The `budget` constants are tuned on an 8-logical-core machine; under `rayon`
/// the budget is scaled by the actual thread-pool size relative to that
/// reference ([`TUNED_THREADS`]). The crossover work scales roughly linearly
/// with the thread count (more threads → more coordination overhead to amortize,
/// and more chunks needed to feed them), so a 2-core machine parallelizes
/// smaller inputs and a 64-core machine demands more work before bothering. This
/// is still a coarse machine-independent *guard*, not a per-machine optimum —
/// the cost band around the crossover is wide and forgiving. On the reference
/// 8-thread machine the scale factor is exactly 1, leaving the tuned behavior
/// unchanged.
#[cfg(feature = "imageproc")]
#[inline]
pub(crate) fn work_col_threshold(per_col_work: usize, budget: usize, min_cols: usize) -> usize {
    #[cfg(feature = "rayon")]
    let budget = {
        let threads = rayon::current_num_threads().max(1);
        budget.saturating_mul(threads) / TUNED_THREADS
    };
    (budget / per_col_work.max(1)).max(min_cols)
}

/// Logical-core count of the machine the `*_WORK_BUDGET` constants were tuned on
/// (Apple M3, 8 logical cores). Used to normalize [`work_col_threshold`]'s
/// thread-count scaling so the reference machine's behavior is unchanged.
#[cfg(all(feature = "imageproc", feature = "rayon"))]
const TUNED_THREADS: usize = 8;

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
pub(crate) fn for_each_chunk_mut<T, F>(
    data: &mut [T],
    chunk_len: usize,
    _threshold: usize,
    mut f: F,
) where
    F: FnMut(usize, &mut [T]),
{
    debug_assert!(chunk_len > 0);
    for (j, chunk) in data.chunks_mut(chunk_len).enumerate() {
        f(j, chunk);
    }
}

/// Apply `f(&mut scratch, i)` for each index `i` in `0..n`, with a
/// per-worker scratch value produced by `init`.
///
/// This is the index-driven counterpart of [`for_each_chunk_mut`] for
/// algorithms whose output is *not* a single shared buffer (e.g. a per-band
/// blur that hands each result to a caller callback instead of materializing
/// it). Under the `rayon` feature the indices run in parallel when `n` reaches
/// `threshold`, using `for_each_init` so `init` runs roughly once per worker
/// job rather than once per index — scratch buffers are therefore reused
/// across the indices a worker processes. Sequentially (below the threshold,
/// or always without the feature) `init` runs exactly once.
///
/// `f` must not depend on the scratch contents left by a previous index
/// beyond what it re-initializes itself, and must not rely on any particular
/// index order.
#[cfg(feature = "rayon")]
#[inline]
pub(crate) fn for_each_index_init<S, I, F>(n: usize, threshold: usize, init: I, f: F)
where
    S: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) + Sync + Send,
{
    use rayon::prelude::*;
    if n >= threshold {
        (0..n).into_par_iter().for_each_init(&init, |s, i| f(s, i));
    } else {
        let mut s = init();
        for i in 0..n {
            f(&mut s, i);
        }
    }
}

#[cfg(not(feature = "rayon"))]
#[inline]
pub(crate) fn for_each_index_init<S, I, F>(n: usize, _threshold: usize, init: I, mut f: F)
where
    I: FnOnce() -> S,
    F: FnMut(&mut S, usize),
{
    let mut s = init();
    for i in 0..n {
        f(&mut s, i);
    }
}
