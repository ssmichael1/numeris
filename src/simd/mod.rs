//! SIMD-accelerated kernels with compile-time (and optionally runtime) ISA dispatch.
//!
//! This module is private — it provides internal acceleration for matrix
//! and vector operations. The public API is unchanged.
//!
//! ## Dispatch strategy
//!
//! TypeId-based dispatch at monomorphization time: for `f32`/`f64`, the
//! compiler selects SIMD kernels and dead-code-eliminates the fallback.
//! For all other types (integers, complex), the scalar fallback is used.
//!
//! On x86_64, the widest available instruction set is selected at compile
//! time: AVX-512 > AVX > SSE2. Enable via `-C target-cpu=native` or
//! `-C target-feature=+avx2` etc. With the `runtime-dispatch` cargo feature the
//! compile-time tier becomes a floor that a one-time CPU probe may raise: the
//! AVX / AVX-512 kernels carry `#[target_feature]`, so a baseline binary can
//! still contain and call them once [`isa`] has confirmed support (see
//! [`x86_select!`]). aarch64 needs neither — NEON is the baseline there.
//!
//! ## Matrix multiply
//!
//! All matmul kernels use register-blocked MR×NR micro-kernels that
//! accumulate the full k-sum in SIMD registers before writing C once,
//! reducing memory traffic from O(m·n·p) to O(m·p) stores. This technique
//! is inspired by [nano-gemm](https://github.com/sarah-quinones/nano-gemm)
//! and [faer](https://github.com/sarah-quinones/faer-rs) by Sarah Quinones.
//!
//! ## Architecture support
//!
//! | Arch      | ISA       | f64 tile | f32 tile |
//! |-----------|-----------|----------|----------|
//! | `aarch64` | NEON      | 4×4      | 8×4      |
//! | `x86_64`  | SSE2      | 4×4      | 8×4      |
//! | `x86_64`  | AVX + FMA | 8×4      | 16×4     |
//! | `x86_64`  | AVX-512   | 16×4     | 32×4     |
//! | other     | scalar    | 4×4      | 4×4      |

// Each architecture file provides a full set of ISA kernels (dot, matmul, AXPY,
// …). The dispatch selects one tier per call, so in any given build some tiers
// are deliberately present but unused (SSE2 when AVX is a compile-time feature;
// AVX / AVX-512 without `runtime-dispatch` or the matching feature) — not
// removable, just inactive for this target.
#![allow(dead_code)]

// ── ISA kernel macros ──────────────────────────────────────────────────────
//
// The element-wise kernels (add/sub/scale/axpy) are algorithmically identical
// across every ISA — they differ only in the vector width and the intrinsic
// names. These macros are the single source of truth: each architecture file
// invokes them once with its own width + intrinsics, instead of hand-writing
// ~110 lines of near-identical bodies. `dot` and `matmul` stay hand-written per
// ISA because their reductions / micro-kernels genuinely diverge.
//
// Defined before the `mod` declarations below so the child module files see
// them (macro_rules textual scope flows into modules declared afterward).

/// Element-wise `add_slices` / `sub_slices` / `scale_slices` for one lane type.
///
/// `$t` is the scalar type, `$lanes` the vector width in elements, and the
/// trailing idents are this ISA's load / store / add / sub / mul / broadcast
/// intrinsics (uniform call shape: `load(ptr)`, `store(ptr, v)`, `op(v, v)`,
/// `set1(scalar)`).
#[allow(unused_macros)] // unused on non-SIMD targets (e.g. thumbv7em)
macro_rules! simd_elementwise_kernels {
    ($(@feature $feat:literal)? $t:ty, $lanes:expr, $load:ident, $store:ident, $add:ident, $sub:ident, $mul:ident, $set1:ident) => {
        /// Element-wise addition: out[i] = a[i] + b[i].
        #[inline]
        $(
        #[target_feature(enable = $feat)]
        // Register-only intrinsics (broadcasts) are safe to call inside a
        // `#[target_feature]` function, so their `unsafe` blocks — required in
        // the unattributed baseline tiers this macro also serves — are redundant
        // here.
        #[allow(unused_unsafe)]
        )?
        pub fn add_slices(a: &[$t], b: &[$t], out: &mut [$t]) {
            debug_assert_eq!(a.len(), b.len());
            debug_assert_eq!(a.len(), out.len());
            let n = a.len();
            for ((o, x), y) in out
                .chunks_exact_mut($lanes)
                .zip(a.chunks_exact($lanes))
                .zip(b.chunks_exact($lanes))
            {
                // SAFETY: `chunks_exact` yields chunks of exactly $lanes elements,
                // which is precisely the width of one vector load / store — so all
                // three accesses are in bounds by construction.
                unsafe { $store(o.as_mut_ptr(), $add($load(x.as_ptr()), $load(y.as_ptr()))) };
            }
            for i in (n - n % $lanes)..n {
                out[i] = a[i] + b[i];
            }
        }

        /// Element-wise subtraction: out[i] = a[i] - b[i].
        #[inline]
        $(
        #[target_feature(enable = $feat)]
        // Register-only intrinsics (broadcasts) are safe to call inside a
        // `#[target_feature]` function, so their `unsafe` blocks — required in
        // the unattributed baseline tiers this macro also serves — are redundant
        // here.
        #[allow(unused_unsafe)]
        )?
        pub fn sub_slices(a: &[$t], b: &[$t], out: &mut [$t]) {
            debug_assert_eq!(a.len(), b.len());
            debug_assert_eq!(a.len(), out.len());
            let n = a.len();
            for ((o, x), y) in out
                .chunks_exact_mut($lanes)
                .zip(a.chunks_exact($lanes))
                .zip(b.chunks_exact($lanes))
            {
                // SAFETY: each chunk is exactly $lanes wide — one vector load / store.
                unsafe { $store(o.as_mut_ptr(), $sub($load(x.as_ptr()), $load(y.as_ptr()))) };
            }
            for i in (n - n % $lanes)..n {
                out[i] = a[i] - b[i];
            }
        }

        /// Scalar multiplication: out[i] = a[i] * scalar.
        #[inline]
        $(
        #[target_feature(enable = $feat)]
        // Register-only intrinsics (broadcasts) are safe to call inside a
        // `#[target_feature]` function, so their `unsafe` blocks — required in
        // the unattributed baseline tiers this macro also serves — are redundant
        // here.
        #[allow(unused_unsafe)]
        )?
        pub fn scale_slices(a: &[$t], scalar: $t, out: &mut [$t]) {
            debug_assert_eq!(a.len(), out.len());
            let n = a.len();
            // SAFETY: a register broadcast of a scalar; touches no memory.
            let vs = unsafe { $set1(scalar) };
            for (o, x) in out.chunks_exact_mut($lanes).zip(a.chunks_exact($lanes)) {
                // SAFETY: each chunk is exactly $lanes wide — one vector load / store.
                unsafe { $store(o.as_mut_ptr(), $mul($load(x.as_ptr()), vs)) };
            }
            for i in (n - n % $lanes)..n {
                out[i] = a[i] * scalar;
            }
        }

        /// In-place scalar multiplication: a[i] *= scalar.
        ///
        /// Distinct from `scale_slices` with aliased arguments: a single `&mut`
        /// borrow means one provenance for both the loads and the stores, so no
        /// shared reference to the buffer exists while it is being written.
        #[inline]
        $(
        #[target_feature(enable = $feat)]
        // Register-only intrinsics (broadcasts) are safe to call inside a
        // `#[target_feature]` function, so their `unsafe` blocks — required in
        // the unattributed baseline tiers this macro also serves — are redundant
        // here.
        #[allow(unused_unsafe)]
        )?
        pub fn scale_in_place(a: &mut [$t], scalar: $t) {
            let n = a.len();
            // SAFETY: a register broadcast of a scalar; touches no memory.
            let vs = unsafe { $set1(scalar) };
            for c in a.chunks_exact_mut($lanes) {
                // SAFETY: the chunk is exactly $lanes wide, so the load and the store
                // each cover it exactly — both through the one `&mut` borrow.
                unsafe {
                    let p = c.as_mut_ptr();
                    $store(p, $mul($load(p), vs));
                }
            }
            for i in (n - n % $lanes)..n {
                a[i] *= scalar;
            }
        }
    };
}

/// AXPY kernels using a separate multiply + add/subtract (x86 SSE2, which has no FMA).
// Unused on aarch64 (which uses the fused variant below).
#[allow(unused_macros)]
macro_rules! simd_axpy_kernels_muladd {
    ($(@feature $feat:literal)? $t:ty, $lanes:expr, $load:ident, $store:ident, $add:ident, $sub:ident, $mul:ident, $set1:ident) => {
        /// AXPY: y[i] -= alpha * x[i].
        #[inline]
        $(
        #[target_feature(enable = $feat)]
        // Register-only intrinsics (broadcasts) are safe to call inside a
        // `#[target_feature]` function, so their `unsafe` blocks — required in
        // the unattributed baseline tiers this macro also serves — are redundant
        // here.
        #[allow(unused_unsafe)]
        )?
        pub fn axpy_neg(y: &mut [$t], alpha: $t, x: &[$t]) {
            debug_assert_eq!(y.len(), x.len());
            let n = y.len();
            // SAFETY: a register broadcast of a scalar; touches no memory.
            let va = unsafe { $set1(alpha) };
            for (yc, xc) in y.chunks_exact_mut($lanes).zip(x.chunks_exact($lanes)) {
                // SAFETY: each chunk is exactly $lanes wide — one vector load / store.
                unsafe {
                    let p = yc.as_mut_ptr();
                    $store(p, $sub($load(p), $mul(va, $load(xc.as_ptr()))));
                }
            }
            for i in (n - n % $lanes)..n {
                y[i] -= alpha * x[i];
            }
        }

        /// AXPY: y[i] += alpha * x[i].
        #[inline]
        $(
        #[target_feature(enable = $feat)]
        // Register-only intrinsics (broadcasts) are safe to call inside a
        // `#[target_feature]` function, so their `unsafe` blocks — required in
        // the unattributed baseline tiers this macro also serves — are redundant
        // here.
        #[allow(unused_unsafe)]
        )?
        pub fn axpy_pos(y: &mut [$t], alpha: $t, x: &[$t]) {
            debug_assert_eq!(y.len(), x.len());
            let n = y.len();
            // SAFETY: a register broadcast of a scalar; touches no memory.
            let va = unsafe { $set1(alpha) };
            for (yc, xc) in y.chunks_exact_mut($lanes).zip(x.chunks_exact($lanes)) {
                // SAFETY: each chunk is exactly $lanes wide — one vector load / store.
                unsafe {
                    let p = yc.as_mut_ptr();
                    $store(p, $add($load(p), $mul(va, $load(xc.as_ptr()))));
                }
            }
            for i in (n - n % $lanes)..n {
                y[i] += alpha * x[i];
            }
        }
    };
}

/// AXPY kernels using fused multiply-add / multiply-subtract (NEON, and the x86
/// AVX / AVX-512 tiers through accumulator-first adapters).
#[allow(unused_macros)]
macro_rules! simd_axpy_kernels_fma {
    ($(@feature $feat:literal)? $t:ty, $lanes:expr, $load:ident, $store:ident, $fma:ident, $fms:ident, $dup:ident) => {
        /// AXPY: y[i] -= alpha * x[i].
        #[inline]
        $(
        #[target_feature(enable = $feat)]
        // Register-only intrinsics (broadcasts) are safe to call inside a
        // `#[target_feature]` function, so their `unsafe` blocks — required in
        // the unattributed baseline tiers this macro also serves — are redundant
        // here.
        #[allow(unused_unsafe)]
        )?
        pub fn axpy_neg(y: &mut [$t], alpha: $t, x: &[$t]) {
            debug_assert_eq!(y.len(), x.len());
            let n = y.len();
            // SAFETY: a register broadcast of a scalar; touches no memory.
            let va = unsafe { $dup(alpha) };
            for (yc, xc) in y.chunks_exact_mut($lanes).zip(x.chunks_exact($lanes)) {
                // SAFETY: each chunk is exactly $lanes wide — one vector load / store.
                // y -= alpha * x  →  fused multiply-subtract.
                unsafe {
                    let p = yc.as_mut_ptr();
                    $store(p, $fms($load(p), va, $load(xc.as_ptr())));
                }
            }
            for i in (n - n % $lanes)..n {
                y[i] -= alpha * x[i];
            }
        }

        /// AXPY: y[i] += alpha * x[i].
        #[inline]
        $(
        #[target_feature(enable = $feat)]
        // Register-only intrinsics (broadcasts) are safe to call inside a
        // `#[target_feature]` function, so their `unsafe` blocks — required in
        // the unattributed baseline tiers this macro also serves — are redundant
        // here.
        #[allow(unused_unsafe)]
        )?
        pub fn axpy_pos(y: &mut [$t], alpha: $t, x: &[$t]) {
            debug_assert_eq!(y.len(), x.len());
            let n = y.len();
            // SAFETY: a register broadcast of a scalar; touches no memory.
            let va = unsafe { $dup(alpha) };
            for (yc, xc) in y.chunks_exact_mut($lanes).zip(x.chunks_exact($lanes)) {
                // SAFETY: each chunk is exactly $lanes wide — one vector load / store.
                unsafe {
                    let p = yc.as_mut_ptr();
                    $store(p, $fma($load(p), va, $load(xc.as_ptr())));
                }
            }
            for i in (n - n % $lanes)..n {
                y[i] += alpha * x[i];
            }
        }
    };
}

/// Strided 1D correlation kernel using NEON fused multiply-add:
/// `out[i] = Σ_k kernel[k] · src[i + k·stride]`.
///
/// The k-sum for a block of outputs is accumulated entirely in registers (four
/// vectors, matching the `dot` kernels' latency-hiding accumulator count), so
/// each output element is stored exactly once — no per-tap read-modify-write of
/// `out`. `stride` is the element distance between consecutive taps: `1` for a
/// convolution along contiguous data, the column stride for one across columns.
///
/// The source reads are strided, so unlike the element-wise kernels they cannot
/// be expressed as `chunks_exact` windows. Their bounds proof rests instead on
/// the window precondition, which is asserted once on entry (see the body).
#[allow(unused_macros)]
macro_rules! simd_conv1d_kernel_fma {
    ($(@feature $feat:literal)? $t:ty, $lanes:expr, $load:ident, $store:ident, $fma:ident, $dup:ident) => {
        /// Strided 1D correlation: out[i] = Σ_k kernel[k] · src[i + k·stride].
        ///
        /// # Panics
        ///
        /// Panics unless `stride >= 1` and `src` covers every window —
        /// `src.len() >= out.len() + (kernel.len() - 1) * stride`. This is checked
        /// once per call (not per element) because the strided loads below rely on
        /// it in release builds, not just under `debug_assertions`.
        #[inline]
        $(
        #[target_feature(enable = $feat)]
        // Register-only intrinsics (broadcasts) are safe to call inside a
        // `#[target_feature]` function, so their `unsafe` blocks — required in
        // the unattributed baseline tiers this macro also serves — are redundant
        // here.
        #[allow(unused_unsafe)]
        )?
        pub fn conv1d(out: &mut [$t], src: &[$t], kernel: &[$t], stride: usize) {
            let n = out.len();
            let klen = kernel.len();
            assert!(stride >= 1, "conv1d: stride must be >= 1");
            assert!(
                klen == 0 || src.len() >= n + (klen - 1) * stride,
                "conv1d: src too short to cover every window"
            );
            // Blocked over four vectors: for a block at output offset `i` and tap
            // `k`, the widest read is `src[i + k*stride + 4*$lanes - 1]`. Since
            // `i + 4*$lanes <= n` and `k <= klen - 1`, that index is at most
            // `n + (klen - 1)*stride - 1`, which the assert above puts inside `src`.
            let mut i = 0;
            while i + 4 * $lanes <= n {
                // SAFETY: broadcast of zero; touches no memory.
                let z = unsafe { $dup(0.0) };
                let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
                for (k, &w) in kernel.iter().enumerate() {
                    // SAFETY: in bounds by the block invariant stated above.
                    unsafe {
                        let w = $dup(w);
                        let p = src.as_ptr().add(i + k * stride);
                        a0 = $fma(a0, w, $load(p));
                        a1 = $fma(a1, w, $load(p.add($lanes)));
                        a2 = $fma(a2, w, $load(p.add(2 * $lanes)));
                        a3 = $fma(a3, w, $load(p.add(3 * $lanes)));
                    }
                }
                let block = &mut out[i..i + 4 * $lanes];
                // SAFETY: `block` is exactly 4·$lanes wide, so the four stores at
                // 0, $lanes, 2·$lanes and 3·$lanes cover it exactly.
                unsafe {
                    let q = block.as_mut_ptr();
                    $store(q, a0);
                    $store(q.add($lanes), a1);
                    $store(q.add(2 * $lanes), a2);
                    $store(q.add(3 * $lanes), a3);
                }
                i += 4 * $lanes;
            }
            while i + $lanes <= n {
                // SAFETY: broadcast of zero; touches no memory.
                let mut a0 = unsafe { $dup(0.0) };
                for (k, &w) in kernel.iter().enumerate() {
                    // SAFETY: `i + $lanes <= n` and `k <= klen - 1`, so the read ends
                    // at most at `n + (klen - 1)*stride - 1`, inside `src` by the
                    // asserted precondition.
                    unsafe { a0 = $fma(a0, $dup(w), $load(src.as_ptr().add(i + k * stride))) };
                }
                let block = &mut out[i..i + $lanes];
                // SAFETY: `block` is exactly $lanes wide — one vector store.
                unsafe { $store(block.as_mut_ptr(), a0) };
                i += $lanes;
            }
            for ii in i..n {
                let mut sum = 0.0;
                for (k, &w) in kernel.iter().enumerate() {
                    sum += w * src[ii + k * stride];
                }
                out[ii] = sum;
            }
        }
    };
}

/// Strided 1D correlation kernel using separate multiply + add
/// (x86 SSE2/AVX/AVX-512). See [`simd_conv1d_kernel_fma`] for the contract.
#[allow(unused_macros)]
macro_rules! simd_conv1d_kernel_muladd {
    ($(@feature $feat:literal)? $t:ty, $lanes:expr, $load:ident, $store:ident, $add:ident, $mul:ident, $set1:ident) => {
        /// Strided 1D correlation: out[i] = Σ_k kernel[k] · src[i + k·stride].
        ///
        /// # Panics
        ///
        /// Panics unless `stride >= 1` and `src` covers every window —
        /// `src.len() >= out.len() + (kernel.len() - 1) * stride`. This is checked
        /// once per call (not per element) because the strided loads below rely on
        /// it in release builds, not just under `debug_assertions`.
        #[inline]
        $(
        #[target_feature(enable = $feat)]
        // Register-only intrinsics (broadcasts) are safe to call inside a
        // `#[target_feature]` function, so their `unsafe` blocks — required in
        // the unattributed baseline tiers this macro also serves — are redundant
        // here.
        #[allow(unused_unsafe)]
        )?
        pub fn conv1d(out: &mut [$t], src: &[$t], kernel: &[$t], stride: usize) {
            let n = out.len();
            let klen = kernel.len();
            assert!(stride >= 1, "conv1d: stride must be >= 1");
            assert!(
                klen == 0 || src.len() >= n + (klen - 1) * stride,
                "conv1d: src too short to cover every window"
            );
            // Blocked over four vectors: for a block at output offset `i` and tap
            // `k`, the widest read is `src[i + k*stride + 4*$lanes - 1]`. Since
            // `i + 4*$lanes <= n` and `k <= klen - 1`, that index is at most
            // `n + (klen - 1)*stride - 1`, which the assert above puts inside `src`.
            let mut i = 0;
            while i + 4 * $lanes <= n {
                // SAFETY: broadcast of zero; touches no memory.
                let z = unsafe { $set1(0.0) };
                let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
                for (k, &w) in kernel.iter().enumerate() {
                    // SAFETY: in bounds by the block invariant stated above.
                    unsafe {
                        let w = $set1(w);
                        let p = src.as_ptr().add(i + k * stride);
                        a0 = $add(a0, $mul(w, $load(p)));
                        a1 = $add(a1, $mul(w, $load(p.add($lanes))));
                        a2 = $add(a2, $mul(w, $load(p.add(2 * $lanes))));
                        a3 = $add(a3, $mul(w, $load(p.add(3 * $lanes))));
                    }
                }
                let block = &mut out[i..i + 4 * $lanes];
                // SAFETY: `block` is exactly 4·$lanes wide, so the four stores at
                // 0, $lanes, 2·$lanes and 3·$lanes cover it exactly.
                unsafe {
                    let q = block.as_mut_ptr();
                    $store(q, a0);
                    $store(q.add($lanes), a1);
                    $store(q.add(2 * $lanes), a2);
                    $store(q.add(3 * $lanes), a3);
                }
                i += 4 * $lanes;
            }
            while i + $lanes <= n {
                // SAFETY: broadcast of zero; touches no memory.
                let mut a0 = unsafe { $set1(0.0) };
                for (k, &w) in kernel.iter().enumerate() {
                    // SAFETY: `i + $lanes <= n` and `k <= klen - 1`, so the read ends
                    // at most at `n + (klen - 1)*stride - 1`, inside `src` by the
                    // asserted precondition.
                    unsafe {
                        a0 = $add(a0, $mul($set1(w), $load(src.as_ptr().add(i + k * stride))));
                    }
                }
                let block = &mut out[i..i + $lanes];
                // SAFETY: `block` is exactly $lanes wide — one vector store.
                unsafe { $store(block.as_mut_ptr(), a0) };
                i += $lanes;
            }
            for ii in i..n {
                let mut sum = 0.0;
                for (k, &w) in kernel.iter().enumerate() {
                    sum += w * src[ii + k * stride];
                }
                out[ii] = sum;
            }
        }
    };
}

/// Deinterleaved (SoA) radix-2 FFT butterfly for one lane type.
///
/// Same body across every ISA — only the vector width and intrinsic names
/// differ — so validating one architecture (NEON, locally) validates the logic
/// for all of them. Computes, per lane block: `v = bot·w` (complex multiply on
/// separate re/im lanes), then `top += v`, `bot -= v`. Uses only the load /
/// store / add / sub / mul intrinsics already wired for `simd_elementwise_kernels!`.
#[allow(unused_macros)]
macro_rules! simd_fft_butterfly_kernel {
    ($(@feature $feat:literal)? $t:ty, $lanes:expr, $load:ident, $store:ident, $add:ident, $sub:ident, $mul:ident) => {
        /// SoA radix-2 butterfly (see the crate `simd::scalar::fft_butterfly` reference).
        #[inline]
        $(
        #[target_feature(enable = $feat)]
        // Register-only intrinsics (broadcasts) are safe to call inside a
        // `#[target_feature]` function, so their `unsafe` blocks — required in
        // the unattributed baseline tiers this macro also serves — are redundant
        // here.
        #[allow(unused_unsafe)]
        )?
        pub fn fft_butterfly(
            tr: &mut [$t],
            ti: &mut [$t],
            br: &mut [$t],
            bi: &mut [$t],
            wr: &[$t],
            wi: &[$t],
        ) {
            let half = tr.len();
            debug_assert_eq!(ti.len(), half);
            debug_assert_eq!(br.len(), half);
            debug_assert_eq!(bi.len(), half);
            debug_assert_eq!(wr.len(), half);
            debug_assert_eq!(wi.len(), half);
            for (((((ctr, cti), cbr), cbi), cwr), cwi) in tr
                .chunks_exact_mut($lanes)
                .zip(ti.chunks_exact_mut($lanes))
                .zip(br.chunks_exact_mut($lanes))
                .zip(bi.chunks_exact_mut($lanes))
                .zip(wr.chunks_exact($lanes))
                .zip(wi.chunks_exact($lanes))
            {
                // SAFETY: every chunk is exactly $lanes wide — one vector load /
                // store each — so all six accesses are in bounds by construction,
                // and each `&mut` chunk is loaded and stored through its own borrow.
                unsafe {
                    let vbr = $load(cbr.as_ptr());
                    let vbi = $load(cbi.as_ptr());
                    let vwr = $load(cwr.as_ptr());
                    let vwi = $load(cwi.as_ptr());
                    // v = bot * w  (complex)
                    let vr = $sub($mul(vbr, vwr), $mul(vbi, vwi));
                    let vi = $add($mul(vbr, vwi), $mul(vbi, vwr));
                    let vtr = $load(ctr.as_ptr());
                    let vti = $load(cti.as_ptr());
                    $store(ctr.as_mut_ptr(), $add(vtr, vr));
                    $store(cti.as_mut_ptr(), $add(vti, vi));
                    $store(cbr.as_mut_ptr(), $sub(vtr, vr));
                    $store(cbi.as_mut_ptr(), $sub(vti, vi));
                }
            }
            for k in (half - half % $lanes)..half {
                let vr = br[k] * wr[k] - bi[k] * wi[k];
                let vi = br[k] * wi[k] + bi[k] * wr[k];
                let trk = tr[k];
                let tik = ti[k];
                tr[k] = trk + vr;
                ti[k] = tik + vi;
                br[k] = trk - vr;
                bi[k] = tik - vi;
            }
        }
    };
}

/// Deinterleaved (SoA) radix-4 FFT butterfly for one lane type.
///
/// One radix-4 stage does the work of two radix-2 stages in a single sweep
/// (half the loads/stores) with three complex twiddle multiplies per four
/// elements instead of four. For a block of four quarter-slices `a, b, c, d`
/// (each a finished length-`q` sub-transform, in the bit-reversed layout the
/// radix-2 stages use) and the stage twiddles `w1 = w^k`, `w2 = w^2k`,
/// `w3 = w^3k` with `w = exp(-2πi/4q)`:
///
/// ```text
/// t1 = w2·b,  t2 = w1·c,  t3 = w3·d
/// a' = a + t1,  b' = a − t1,  u = t2 + t3,  v = t2 − t3
/// a ← a' + u,   c ← a' − u,   b ← b' − i·v,   d ← b' + i·v
/// ```
///
/// which is exactly the pair of radix-2 stages (`len = 2q` then `4q`) fused,
/// with `w_{2q}^k = w2` and the second stage's `w_{4q}^{k+q} = −i·w1`. Same body
/// across every ISA, like `simd_fft_butterfly_kernel!`.
#[allow(unused_macros)]
macro_rules! simd_fft_butterfly4_kernel {
    ($(@feature $feat:literal)? $t:ty, $lanes:expr, $load:ident, $store:ident, $add:ident, $sub:ident, $mul:ident) => {
        /// SoA radix-4 butterfly (see the crate `simd::scalar::fft_butterfly4` reference).
        #[inline]
        #[allow(clippy::too_many_arguments)]
        $(
        #[target_feature(enable = $feat)]
        // Register-only intrinsics (broadcasts) are safe to call inside a
        // `#[target_feature]` function, so their `unsafe` blocks — required in
        // the unattributed baseline tiers this macro also serves — are redundant
        // here.
        #[allow(unused_unsafe)]
        )?
        pub fn fft_butterfly4(
            ar: &mut [$t],
            ai: &mut [$t],
            br: &mut [$t],
            bi: &mut [$t],
            cr: &mut [$t],
            ci: &mut [$t],
            dr: &mut [$t],
            di: &mut [$t],
            w1r: &[$t],
            w1i: &[$t],
            w2r: &[$t],
            w2i: &[$t],
            w3r: &[$t],
            w3i: &[$t],
        ) {
            let q = ar.len();
            for s in [
                &*ai, &*br, &*bi, &*cr, &*ci, &*dr, &*di, w1r, w1i, w2r, w2i, w3r, w3i,
            ] {
                debug_assert_eq!(s.len(), q);
            }
            let it = ar
                .chunks_exact_mut($lanes)
                .zip(ai.chunks_exact_mut($lanes))
                .zip(br.chunks_exact_mut($lanes))
                .zip(bi.chunks_exact_mut($lanes))
                .zip(cr.chunks_exact_mut($lanes))
                .zip(ci.chunks_exact_mut($lanes))
                .zip(dr.chunks_exact_mut($lanes))
                .zip(di.chunks_exact_mut($lanes))
                .zip(w1r.chunks_exact($lanes))
                .zip(w1i.chunks_exact($lanes))
                .zip(w2r.chunks_exact($lanes))
                .zip(w2i.chunks_exact($lanes))
                .zip(w3r.chunks_exact($lanes))
                .zip(w3i.chunks_exact($lanes));
            for (
                (
                    (
                        (
                            (((((((((car, cai), cbr), cbi), ccr), cci), cdr), cdi), cw1r), cw1i),
                            cw2r,
                        ),
                        cw2i,
                    ),
                    cw3r,
                ),
                cw3i,
            ) in it
            {
                // SAFETY: every chunk is exactly $lanes wide — one vector load /
                // store each — so all fourteen accesses are in bounds by
                // construction, and each `&mut` chunk is loaded and stored
                // through its own borrow.
                unsafe {
                    let (vbr, vbi) = ($load(cbr.as_ptr()), $load(cbi.as_ptr()));
                    let (vcr, vci) = ($load(ccr.as_ptr()), $load(cci.as_ptr()));
                    let (vdr, vdi) = ($load(cdr.as_ptr()), $load(cdi.as_ptr()));
                    let (v1r, v1i) = ($load(cw1r.as_ptr()), $load(cw1i.as_ptr()));
                    let (v2r, v2i) = ($load(cw2r.as_ptr()), $load(cw2i.as_ptr()));
                    let (v3r, v3i) = ($load(cw3r.as_ptr()), $load(cw3i.as_ptr()));
                    // t1 = w2·b, t2 = w1·c, t3 = w3·d
                    let t1r = $sub($mul(vbr, v2r), $mul(vbi, v2i));
                    let t1i = $add($mul(vbr, v2i), $mul(vbi, v2r));
                    let t2r = $sub($mul(vcr, v1r), $mul(vci, v1i));
                    let t2i = $add($mul(vcr, v1i), $mul(vci, v1r));
                    let t3r = $sub($mul(vdr, v3r), $mul(vdi, v3i));
                    let t3i = $add($mul(vdr, v3i), $mul(vdi, v3r));
                    let (var, vai) = ($load(car.as_ptr()), $load(cai.as_ptr()));
                    // a' = a + t1, b' = a − t1, u = t2 + t3, v = t2 − t3
                    let (apr, api) = ($add(var, t1r), $add(vai, t1i));
                    let (bpr, bpi) = ($sub(var, t1r), $sub(vai, t1i));
                    let (ur, ui) = ($add(t2r, t3r), $add(t2i, t3i));
                    let (vr, vi) = ($sub(t2r, t3r), $sub(t2i, t3i));
                    // a = a' + u, c = a' − u, b = b' − i·v, d = b' + i·v
                    // (−i·v = (v.im, −v.re), so b' − i·v = (b'.re + v.im, b'.im − v.re)).
                    $store(car.as_mut_ptr(), $add(apr, ur));
                    $store(cai.as_mut_ptr(), $add(api, ui));
                    $store(ccr.as_mut_ptr(), $sub(apr, ur));
                    $store(cci.as_mut_ptr(), $sub(api, ui));
                    $store(cbr.as_mut_ptr(), $add(bpr, vi));
                    $store(cbi.as_mut_ptr(), $sub(bpi, vr));
                    $store(cdr.as_mut_ptr(), $sub(bpr, vi));
                    $store(cdi.as_mut_ptr(), $add(bpi, vr));
                }
            }
            for k in (q - q % $lanes)..q {
                let t1r = br[k] * w2r[k] - bi[k] * w2i[k];
                let t1i = br[k] * w2i[k] + bi[k] * w2r[k];
                let t2r = cr[k] * w1r[k] - ci[k] * w1i[k];
                let t2i = cr[k] * w1i[k] + ci[k] * w1r[k];
                let t3r = dr[k] * w3r[k] - di[k] * w3i[k];
                let t3i = dr[k] * w3i[k] + di[k] * w3r[k];
                let (apr, api) = (ar[k] + t1r, ai[k] + t1i);
                let (bpr, bpi) = (ar[k] - t1r, ai[k] - t1i);
                let (ur, ui) = (t2r + t3r, t2i + t3i);
                let (vr, vi) = (t2r - t3r, t2i - t3i);
                ar[k] = apr + ur;
                ai[k] = api + ui;
                cr[k] = apr - ur;
                ci[k] = api - ui;
                br[k] = bpr + vi;
                bi[k] = bpi - vr;
                dr[k] = bpr - vi;
                di[k] = bpi + vr;
            }
        }
    };
}

pub(crate) mod scalar;

#[cfg(target_arch = "aarch64")]
pub(crate) mod f32_neon;
#[cfg(target_arch = "aarch64")]
pub(crate) mod f64_neon;

#[cfg(target_arch = "x86_64")]
pub(crate) mod f32_sse2;
#[cfg(target_arch = "x86_64")]
pub(crate) mod f64_sse2;

// The AVX / AVX-512 tiers compile on every x86_64 target: each kernel carries
// its own `#[target_feature]`, and `isa()` below decides whether it may be
// called. In a build without the feature enabled at compile time and without
// `runtime-dispatch`, they are never referenced and are dropped by the linker.
#[cfg(target_arch = "x86_64")]
pub(crate) mod f32_avx;
#[cfg(target_arch = "x86_64")]
pub(crate) mod f64_avx;

#[cfg(target_arch = "x86_64")]
pub(crate) mod f32_avx512;
#[cfg(target_arch = "x86_64")]
pub(crate) mod f64_avx512;

use core::any::TypeId;
use core::marker::PhantomData;

use crate::traits::Scalar;

/// Zero-sized proof that the type parameters `T` and `U` are the same type.
///
/// The ISA kernels are written against concrete `f32` / `f64` slices while this
/// dispatch layer is generic over `T: Scalar`, so bridging the two needs a
/// reinterpreting cast — and the only thing that makes such a cast sound is a
/// `TypeId` comparison establishing `T == U`.
///
/// Rather than re-deriving that argument at every cast, where a copy-paste slip
/// (testing `f64`, casting to `f32`) would compile cleanly and silently
/// reinterpret memory, the comparison is performed once — in [`TypeEq::new`],
/// the sole constructor — and yields this witness. Every cast then flows through
/// a method on the witness, so the `U` that was tested is necessarily the `U`
/// that is cast to: the mismatch is unrepresentable. All of the module's
/// reinterpreting `unsafe` is confined to the four methods below.
struct TypeEq<T: Copy + 'static, U: Copy + 'static>(PhantomData<fn() -> (T, U)>);

impl<T: Copy + 'static, U: Copy + 'static> Clone for TypeEq<T, U> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Copy + 'static, U: Copy + 'static> Copy for TypeEq<T, U> {}

impl<T: Copy + 'static, U: Copy + 'static> TypeEq<T, U> {
    /// Returns a witness iff `T` and `U` really are the same type.
    #[inline(always)]
    fn new() -> Option<Self> {
        (TypeId::of::<T>() == TypeId::of::<U>()).then_some(Self(PhantomData))
    }

    /// Reinterprets a `T` slice as the equivalent `U` slice.
    #[inline(always)]
    fn slice(self, s: &[T]) -> &[U] {
        // SAFETY: holding `self` proves `TypeId::of::<T>() == TypeId::of::<U>()`,
        // and `TypeId` equality of two `'static` types means they are the same
        // type — so `[T]` and `[U]` have identical size, alignment and layout.
        // The cast preserves the pointer's provenance and the shared borrow of
        // `s`, whose lifetime is tied to the returned reference.
        unsafe { &*(s as *const [T] as *const [U]) }
    }

    /// Reinterprets a mutable `T` slice as the equivalent `U` slice.
    #[inline(always)]
    fn slice_mut(self, s: &mut [T]) -> &mut [U] {
        // SAFETY: as in `slice`, `T` and `U` are the same type, so the layouts
        // match exactly. The cast consumes the exclusive borrow of `s` and ties
        // it to the returned reference, so no aliasing is introduced.
        unsafe { &mut *(s as *mut [T] as *mut [U]) }
    }

    /// Reinterprets a `T` value as the equivalent `U` value.
    #[inline(always)]
    fn value(self, v: T) -> U {
        // SAFETY: `T` and `U` are the same type, so the read is correctly sized
        // and aligned, and reads an initialized value. Both are `Copy`, so
        // producing a second copy duplicates no ownership.
        unsafe { *(&v as *const T as *const U) }
    }

    /// Reinterprets a `U` value — a kernel's return — back as a `T` value.
    #[inline(always)]
    fn value_back(self, v: U) -> T {
        // SAFETY: the mirror of `value`; same type, both `Copy`.
        unsafe { *(&v as *const U as *const T) }
    }
}

// ── x86_64 tier selection ──────────────────────────────────────────────────

/// The x86_64 SIMD tier a dispatch call may use.
///
/// Ordered by width, so `>=` comparisons read as "at least this wide".
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum Isa {
    /// 128-bit — the x86_64 baseline, always available.
    Sse2,
    /// 256-bit (`avx` + `fma`; the tier fuses every multiply-add).
    Avx,
    /// 512-bit (`avx512f`).
    Avx512,
}

/// Widest x86_64 tier this build may call.
///
/// The compile-time target features set a floor: a tier that is enabled for the
/// whole compilation unit is returned as a constant, so the `match` in
/// [`x86_select!`] folds away and the generated code is identical to a build
/// without runtime dispatch. Under the `runtime-dispatch` feature the floor may
/// only be *raised*, by [`runtime_isa`], never lowered.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn isa() -> Isa {
    #[cfg(target_feature = "avx512f")]
    {
        Isa::Avx512
    }
    #[cfg(not(target_feature = "avx512f"))]
    {
        #[cfg(all(target_feature = "avx", target_feature = "fma"))]
        let floor = Isa::Avx;
        #[cfg(not(all(target_feature = "avx", target_feature = "fma")))]
        let floor = Isa::Sse2;
        #[cfg(feature = "runtime-dispatch")]
        {
            runtime_isa().max(floor)
        }
        #[cfg(not(feature = "runtime-dispatch"))]
        {
            floor
        }
    }
}

/// Widest tier the running CPU (and OS) support, probed once and cached.
///
/// `std::is_x86_feature_detected!` already caches its own probe, but each call
/// is a load plus a bit test per feature; caching the resolved tier in one byte
/// keeps the per-dispatch cost to a single relaxed load and a compare. The OS
/// state-save check (`xgetbv`) is part of the std probe, so a CPU whose kernel
/// has not enabled AVX / ZMM state reports the lower tier.
#[cfg(all(target_arch = "x86_64", feature = "runtime-dispatch"))]
#[inline(always)]
fn runtime_isa() -> Isa {
    use core::sync::atomic::{AtomicU8, Ordering};
    // 0 = not yet probed; otherwise `Isa as u8 + 1`.
    static CACHE: AtomicU8 = AtomicU8::new(0);

    #[cold]
    fn probe() -> Isa {
        let isa = if std::is_x86_feature_detected!("avx512f") {
            Isa::Avx512
        } else if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
            Isa::Avx
        } else {
            Isa::Sse2
        };
        CACHE.store(isa as u8 + 1, Ordering::Relaxed);
        isa
    }

    match CACHE.load(Ordering::Relaxed) {
        0 => probe(),
        1 => Isa::Sse2,
        2 => Isa::Avx,
        _ => Isa::Avx512,
    }
}

/// Call `$kernel` from the widest x86_64 tier that [`isa`] reports.
///
/// The first token names the element type, which selects the module family
/// (`f64` → `f64_sse2` / `f64_avx` / `f64_avx512`). Expands to an expression, so
/// it also carries `dot`'s return value.
///
/// The AVX and AVX-512 arms are `unsafe` blocks: those kernels are
/// `#[target_feature]` functions, and calling one is unsafe unless the *caller*
/// carries the same attribute — a crate-wide `-C target-feature` flag does not
/// count, so the block is needed even in a build where the tier is the
/// compile-time floor. This is the crate's only dispatch-site `unsafe`.
#[cfg(target_arch = "x86_64")]
macro_rules! x86_select {
    (f64, $kernel:ident ( $($arg:expr),* $(,)? )) => {
        x86_select!(@tiers f64_sse2, f64_avx, f64_avx512, $kernel($($arg),*))
    };
    (f32, $kernel:ident ( $($arg:expr),* $(,)? )) => {
        x86_select!(@tiers f32_sse2, f32_avx, f32_avx512, $kernel($($arg),*))
    };
    (@tiers $sse2:ident, $avx:ident, $avx512:ident, $kernel:ident ( $($arg:expr),* )) => {
        match isa() {
            // SAFETY: `isa()` returns `Avx512` only when AVX-512F is a
            // compile-time target feature of this build, or `runtime_isa` has
            // confirmed via `is_x86_feature_detected!("avx512f")` that the
            // running CPU and OS support it — the precondition for calling a
            // `#[target_feature(enable = "avx512f")]` kernel.
            Isa::Avx512 => unsafe { $avx512::$kernel($($arg),*) },
            // SAFETY: as for `Avx512` — `isa()` returns `Avx` only when AVX and FMA
            // are compile-time target features or the runtime probe confirmed both.
            Isa::Avx => unsafe { $avx::$kernel($($arg),*) },
            Isa::Sse2 => $sse2::$kernel($($arg),*),
        }
    };
}

/// Dispatch dot product to SIMD or scalar fallback.
#[inline]
pub(crate) fn dot_dispatch<T: Scalar>(a: &[T], b: &[T]) -> T {
    #[cfg(target_arch = "aarch64")]
    {
        if let Some(w) = TypeEq::<T, f64>::new() {
            return w.value_back(f64_neon::dot(w.slice(a), w.slice(b)));
        }
        if let Some(w) = TypeEq::<T, f32>::new() {
            return w.value_back(f32_neon::dot(w.slice(a), w.slice(b)));
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(w) = TypeEq::<T, f64>::new() {
            let (a, b) = (w.slice(a), w.slice(b));
            let result = x86_select!(f64, dot(a, b));
            return w.value_back(result);
        }
        if let Some(w) = TypeEq::<T, f32>::new() {
            let (a, b) = (w.slice(a), w.slice(b));
            let result = x86_select!(f32, dot(a, b));
            return w.value_back(result);
        }
    }
    scalar::dot(a, b)
}

/// Dispatch conjugated dot product `Σ conj(a[i]) · b[i]` to SIMD or scalar fallback.
///
/// For real floats `conj` is the identity, so `f32`/`f64` forward to the SIMD
/// [`dot_dispatch`]; all other element types (e.g. complex) use a scalar
/// conjugating loop. Slices shorter than `DOTC_SIMD_CUTOFF` also take the
/// scalar loop: the out-of-line SIMD kernel call costs more than an inlined
/// few-iteration loop (measured: 4×4 QR regressed ~33% without the cutoff).
#[inline]
pub(crate) fn dotc_dispatch<T: crate::traits::LinalgScalar>(a: &[T], b: &[T]) -> T {
    // A plain type test, not a cast — no `TypeEq` witness needed.
    const DOTC_SIMD_CUTOFF: usize = 8;
    if a.len() >= DOTC_SIMD_CUTOFF
        && (TypeId::of::<T>() == TypeId::of::<f64>() || TypeId::of::<T>() == TypeId::of::<f32>())
    {
        return dot_dispatch(a, b);
    }
    let mut acc = T::zero();
    for i in 0..a.len() {
        acc = acc + a[i].conj() * b[i];
    }
    acc
}

/// Dispatch matrix multiply to SIMD or scalar fallback.
///
/// `c` must be zero-initialized. Computes `C += A * B` in-place.
#[inline]
pub(crate) fn matmul_dispatch<T: Scalar>(
    a: &[T],
    b: &[T],
    c: &mut [T],
    m: usize,
    n: usize,
    p: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if let Some(w) = TypeEq::<T, f64>::new() {
            f64_neon::matmul(w.slice(a), w.slice(b), w.slice_mut(c), m, n, p);
            return;
        }
        if let Some(w) = TypeEq::<T, f32>::new() {
            f32_neon::matmul(w.slice(a), w.slice(b), w.slice_mut(c), m, n, p);
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(w) = TypeEq::<T, f64>::new() {
            let (a, b, c) = (w.slice(a), w.slice(b), w.slice_mut(c));
            x86_select!(f64, matmul(a, b, c, m, n, p));
            return;
        }
        if let Some(w) = TypeEq::<T, f32>::new() {
            let (a, b, c) = (w.slice(a), w.slice(b), w.slice_mut(c));
            x86_select!(f32, matmul(a, b, c, m, n, p));
            return;
        }
    }
    scalar::matmul(a, b, c, m, n, p);
}

/// Dispatch a SIMD element-wise kernel: select the widest available ISA
/// (AVX-512 > AVX > SSE2 on x86_64, NEON on aarch64) for `f32`/`f64`, otherwise
/// fall back to the scalar kernel. One arm per argument shape; `dot`/`matmul`
/// keep the bespoke dispatch above because their reductions genuinely diverge.
macro_rules! simd_dispatch {
    // out[i] = a[i] (op) b[i]  — add_slices / sub_slices
    (binop $(#[$attr:meta])* $name:ident => $kernel:ident) => {
        $(#[$attr])*
        #[inline]
        pub(crate) fn $name<T: Scalar>(a: &[T], b: &[T], out: &mut [T]) {
            #[cfg(target_arch = "aarch64")]
            {
                if let Some(w) = TypeEq::<T, f64>::new() {
                    f64_neon::$kernel(w.slice(a), w.slice(b), w.slice_mut(out));
                    return;
                }
                if let Some(w) = TypeEq::<T, f32>::new() {
                    f32_neon::$kernel(w.slice(a), w.slice(b), w.slice_mut(out));
                    return;
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if let Some(w) = TypeEq::<T, f64>::new() {
                    let (a, b, out) = (w.slice(a), w.slice(b), w.slice_mut(out));
                    x86_select!(f64, $kernel(a, b, out));
                    return;
                }
                if let Some(w) = TypeEq::<T, f32>::new() {
                    let (a, b, out) = (w.slice(a), w.slice(b), w.slice_mut(out));
                    x86_select!(f32, $kernel(a, b, out));
                    return;
                }
            }
            scalar::$kernel(a, b, out);
        }
    };
    // out[i] = a[i] * scalar  — scale_slices
    (scale $(#[$attr:meta])* $name:ident => $kernel:ident) => {
        $(#[$attr])*
        #[inline]
        pub(crate) fn $name<T: Scalar>(a: &[T], scalar: T, out: &mut [T]) {
            #[cfg(target_arch = "aarch64")]
            {
                if let Some(w) = TypeEq::<T, f64>::new() {
                    f64_neon::$kernel(w.slice(a), w.value(scalar), w.slice_mut(out));
                    return;
                }
                if let Some(w) = TypeEq::<T, f32>::new() {
                    f32_neon::$kernel(w.slice(a), w.value(scalar), w.slice_mut(out));
                    return;
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if let Some(w) = TypeEq::<T, f64>::new() {
                    let (a, s, out) = (w.slice(a), w.value(scalar), w.slice_mut(out));
                    x86_select!(f64, $kernel(a, s, out));
                    return;
                }
                if let Some(w) = TypeEq::<T, f32>::new() {
                    let (a, s, out) = (w.slice(a), w.value(scalar), w.slice_mut(out));
                    x86_select!(f32, $kernel(a, s, out));
                    return;
                }
            }
            scalar::$kernel(a, scalar, out);
        }
    };
    // a[i] *= scalar  — scale_in_place
    (scale_ip $(#[$attr:meta])* $name:ident => $kernel:ident => $fallback:ident) => {
        $(#[$attr])*
        #[inline]
        pub(crate) fn $name<T: Scalar>(a: &mut [T], scalar: T) {
            #[cfg(target_arch = "aarch64")]
            {
                if let Some(w) = TypeEq::<T, f64>::new() {
                    f64_neon::$kernel(w.slice_mut(a), w.value(scalar));
                    return;
                }
                if let Some(w) = TypeEq::<T, f32>::new() {
                    f32_neon::$kernel(w.slice_mut(a), w.value(scalar));
                    return;
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if let Some(w) = TypeEq::<T, f64>::new() {
                    let (s, a) = (w.value(scalar), w.slice_mut(a));
                    x86_select!(f64, $kernel(a, s));
                    return;
                }
                if let Some(w) = TypeEq::<T, f32>::new() {
                    let (s, a) = (w.value(scalar), w.slice_mut(a));
                    x86_select!(f32, $kernel(a, s));
                    return;
                }
            }
            scalar::$fallback(a, scalar);
        }
    };
    // y[i] (op)= alpha * x[i]  — axpy_neg / axpy_pos
    (axpy $(#[$attr:meta])* $name:ident => $kernel:ident) => {
        $(#[$attr])*
        #[inline]
        pub(crate) fn $name<T: Scalar>(y: &mut [T], alpha: T, x: &[T]) {
            // Short slices: the out-of-line SIMD call costs more than the loop.
            if y.len() < 8 {
                scalar::$kernel(y, alpha, x);
                return;
            }
            #[cfg(target_arch = "aarch64")]
            {
                if let Some(w) = TypeEq::<T, f64>::new() {
                    f64_neon::$kernel(w.slice_mut(y), w.value(alpha), w.slice(x));
                    return;
                }
                if let Some(w) = TypeEq::<T, f32>::new() {
                    f32_neon::$kernel(w.slice_mut(y), w.value(alpha), w.slice(x));
                    return;
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if let Some(w) = TypeEq::<T, f64>::new() {
                    let (y, al, x) = (w.slice_mut(y), w.value(alpha), w.slice(x));
                    x86_select!(f64, $kernel(y, al, x));
                    return;
                }
                if let Some(w) = TypeEq::<T, f32>::new() {
                    let (y, al, x) = (w.slice_mut(y), w.value(alpha), w.slice(x));
                    x86_select!(f32, $kernel(y, al, x));
                    return;
                }
            }
            scalar::$kernel(y, alpha, x);
        }
    };
}

simd_dispatch!(binop
    /// Dispatch element-wise addition to SIMD or scalar fallback.
    add_slices_dispatch => add_slices);
simd_dispatch!(binop
    /// Dispatch element-wise subtraction to SIMD or scalar fallback.
    sub_slices_dispatch => sub_slices);
simd_dispatch!(scale
    /// Dispatch scalar multiplication to SIMD or scalar fallback.
    scale_slices_dispatch => scale_slices);
simd_dispatch!(scale_ip
    /// Dispatch in-place scalar multiplication (`a[i] *= scalar`) to SIMD or
    /// scalar fallback.
    ///
    /// Uses dedicated in-place kernels rather than aliasing the input and output
    /// of [`scale_slices_dispatch`]: handing the same buffer to a `&[T]` and a
    /// `&mut [T]` parameter simultaneously violates Rust's aliasing rules even
    /// though the element-wise kernel never reads ahead of its writes.
    scale_in_place_dispatch => scale_in_place => scale_assign_slices);
simd_dispatch!(axpy
    /// Dispatch AXPY: y[i] -= alpha * x[i].
    ///
    /// For short slices (< 8 elements) falls back to the scalar kernel to avoid
    /// SIMD dispatch / register-setup overhead, which dominates at small sizes.
    axpy_neg_dispatch => axpy_neg);
simd_dispatch!(axpy
    /// Dispatch AXPY: y[i] += alpha * x[i].
    ///
    /// For short slices (< 8 elements) falls back to the scalar kernel to avoid
    /// SIMD dispatch / register-setup overhead, which dominates at small sizes.
    axpy_pos_dispatch => axpy_pos);

/// Dispatch strided 1D correlation `out[i] = Σ_k kernel[k] · src[i + k·stride]`
/// to SIMD or scalar fallback.
///
/// `stride` is the element distance between consecutive kernel taps: `1`
/// convolves along contiguous data (e.g. down a matrix column), the column
/// stride (`nrows`) convolves across columns at a fixed row. `src` must cover
/// every window: `src.len() >= out.len() + (kernel.len() - 1) · stride`.
#[inline]
pub(crate) fn conv1d_dispatch<T: Scalar>(out: &mut [T], src: &[T], kernel: &[T], stride: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        if let Some(w) = TypeEq::<T, f64>::new() {
            f64_neon::conv1d(w.slice_mut(out), w.slice(src), w.slice(kernel), stride);
            return;
        }
        if let Some(w) = TypeEq::<T, f32>::new() {
            f32_neon::conv1d(w.slice_mut(out), w.slice(src), w.slice(kernel), stride);
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(w) = TypeEq::<T, f64>::new() {
            let (out, src, kernel) = (w.slice_mut(out), w.slice(src), w.slice(kernel));
            x86_select!(f64, conv1d(out, src, kernel, stride));
            return;
        }
        if let Some(w) = TypeEq::<T, f32>::new() {
            let (out, src, kernel) = (w.slice_mut(out), w.slice(src), w.slice(kernel));
            x86_select!(f32, conv1d(out, src, kernel, stride));
            return;
        }
    }
    scalar::conv1d(out, src, kernel, stride);
}

/// Dispatch a deinterleaved (SoA) radix-2 FFT butterfly to SIMD or scalar.
///
/// Operates on the top halves (`tr`/`ti`), bottom halves (`br`/`bi`), and stage
/// twiddles (`wr`/`wi`), all of equal length. For `f32`/`f64` the SIMD kernels
/// are selected at compile time (widest ISA available); all other types use the
/// scalar reference in [`scalar::fft_butterfly`]. Used by the `fft` module's
/// heap-backed `DynFft` tier.
#[inline]
#[allow(dead_code)]
pub(crate) fn fft_butterfly_dispatch<T: Scalar>(
    tr: &mut [T],
    ti: &mut [T],
    br: &mut [T],
    bi: &mut [T],
    wr: &[T],
    wi: &[T],
) {
    // Reinterpret the six `T` slices as the concrete float type through the
    // `TypeEq` witness `$w`, then call `$module::fft_butterfly`. Unused on
    // architectures without a SIMD kernel (e.g. the thumbv7em no_std target).
    #[allow(unused_macros)]
    macro_rules! simd_call {
        ($w:ident, $module:ident) => {{
            let (tr, ti) = ($w.slice_mut(tr), $w.slice_mut(ti));
            let (br, bi) = ($w.slice_mut(br), $w.slice_mut(bi));
            let (wr, wi) = ($w.slice(wr), $w.slice(wi));
            $module::fft_butterfly(tr, ti, br, bi, wr, wi);
            return;
        }};
        ($w:ident, x86 $ty:ident) => {{
            let (tr, ti) = ($w.slice_mut(tr), $w.slice_mut(ti));
            let (br, bi) = ($w.slice_mut(br), $w.slice_mut(bi));
            let (wr, wi) = ($w.slice(wr), $w.slice(wi));
            x86_select!($ty, fft_butterfly(tr, ti, br, bi, wr, wi));
            return;
        }};
    }

    #[cfg(target_arch = "aarch64")]
    {
        if let Some(w) = TypeEq::<T, f64>::new() {
            simd_call!(w, f64_neon);
        }
        if let Some(w) = TypeEq::<T, f32>::new() {
            simd_call!(w, f32_neon);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(w) = TypeEq::<T, f64>::new() {
            simd_call!(w, x86 f64);
        }
        if let Some(w) = TypeEq::<T, f32>::new() {
            simd_call!(w, x86 f32);
        }
    }
    scalar::fft_butterfly(tr, ti, br, bi, wr, wi);
}

/// Dispatch a deinterleaved (SoA) radix-4 FFT butterfly to SIMD or scalar.
///
/// Operates on four quarter-slices `a`/`b`/`c`/`d` (re/im each) and three
/// twiddle sets `w1`/`w2`/`w3` (re/im each), all of equal length — see
/// [`scalar::fft_butterfly4`] for the arithmetic. Used by the `fft` module's
/// heap-backed `DynFft` tier.
#[inline]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fft_butterfly4_dispatch<T: Scalar>(
    ar: &mut [T],
    ai: &mut [T],
    br: &mut [T],
    bi: &mut [T],
    cr: &mut [T],
    ci: &mut [T],
    dr: &mut [T],
    di: &mut [T],
    w1r: &[T],
    w1i: &[T],
    w2r: &[T],
    w2i: &[T],
    w3r: &[T],
    w3i: &[T],
) {
    // Reinterpret the fourteen `T` slices as the concrete float type through
    // the `TypeEq` witness `$w`, then call `$module::fft_butterfly4`. Unused on
    // architectures without a SIMD kernel (e.g. the thumbv7em no_std target).
    #[allow(unused_macros)]
    macro_rules! simd_call {
        ($w:ident, $module:ident) => {{
            let (ar, ai) = ($w.slice_mut(ar), $w.slice_mut(ai));
            let (br, bi) = ($w.slice_mut(br), $w.slice_mut(bi));
            let (cr, ci) = ($w.slice_mut(cr), $w.slice_mut(ci));
            let (dr, di) = ($w.slice_mut(dr), $w.slice_mut(di));
            let (w1r, w1i) = ($w.slice(w1r), $w.slice(w1i));
            let (w2r, w2i) = ($w.slice(w2r), $w.slice(w2i));
            let (w3r, w3i) = ($w.slice(w3r), $w.slice(w3i));
            $module::fft_butterfly4(ar, ai, br, bi, cr, ci, dr, di, w1r, w1i, w2r, w2i, w3r, w3i);
            return;
        }};
        ($w:ident, x86 $ty:ident) => {{
            let (ar, ai) = ($w.slice_mut(ar), $w.slice_mut(ai));
            let (br, bi) = ($w.slice_mut(br), $w.slice_mut(bi));
            let (cr, ci) = ($w.slice_mut(cr), $w.slice_mut(ci));
            let (dr, di) = ($w.slice_mut(dr), $w.slice_mut(di));
            let (w1r, w1i) = ($w.slice(w1r), $w.slice(w1i));
            let (w2r, w2i) = ($w.slice(w2r), $w.slice(w2i));
            let (w3r, w3i) = ($w.slice(w3r), $w.slice(w3i));
            x86_select!(
                $ty,
                fft_butterfly4(ar, ai, br, bi, cr, ci, dr, di, w1r, w1i, w2r, w2i, w3r, w3i)
            );
            return;
        }};
    }

    #[cfg(target_arch = "aarch64")]
    {
        if let Some(w) = TypeEq::<T, f64>::new() {
            simd_call!(w, f64_neon);
        }
        if let Some(w) = TypeEq::<T, f32>::new() {
            simd_call!(w, f32_neon);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(w) = TypeEq::<T, f64>::new() {
            simd_call!(w, x86 f64);
        }
        if let Some(w) = TypeEq::<T, f32>::new() {
            simd_call!(w, x86 f32);
        }
    }
    scalar::fft_butterfly4(ar, ai, br, bi, cr, ci, dr, di, w1r, w1i, w2r, w2i, w3r, w3i);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    // ── FFT butterflies: SIMD dispatch vs scalar reference ─────────────

    #[test]
    fn fft_butterfly4_dispatch_matches_scalar() {
        for q in [1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33] {
            let mk = |seed: usize| -> Vec<f64> {
                (0..q)
                    .map(|k| (((k * 7 + seed * 13) % 23) as f64) * 0.25 - 2.0)
                    .collect()
            };
            let mut a = [mk(1), mk(2), mk(3), mk(4), mk(5), mk(6), mk(7), mk(8)];
            let w: Vec<Vec<f64>> = (0..6)
                .map(|s| {
                    (0..q)
                        .map(|k| ((k * 3 + s * 5) as f64 * 0.37).cos())
                        .collect()
                })
                .collect();
            let mut r = a.clone();
            let [ar, ai, br, bi, cr, ci, dr, di] = &mut a;
            fft_butterfly4_dispatch(
                ar, ai, br, bi, cr, ci, dr, di, &w[0], &w[1], &w[2], &w[3], &w[4], &w[5],
            );
            let [ar, ai, br, bi, cr, ci, dr, di] = &mut r;
            scalar::fft_butterfly4(
                ar, ai, br, bi, cr, ci, dr, di, &w[0], &w[1], &w[2], &w[3], &w[4], &w[5],
            );
            for (x, y) in a.iter().zip(&r) {
                for (p, s) in x.iter().zip(y) {
                    assert!((p - s).abs() < 1e-12, "radix-4 SIMD vs scalar at q={q}");
                }
            }
        }
    }

    #[test]
    fn fft_butterfly_dispatch_matches_scalar() {
        for h in [1usize, 2, 3, 5, 8, 9, 16, 17, 33] {
            let mk = |seed: usize| -> Vec<f64> {
                (0..h)
                    .map(|k| (((k * 5 + seed * 11) % 19) as f64) * 0.5 - 4.0)
                    .collect()
            };
            let mut a = [mk(1), mk(2), mk(3), mk(4)];
            let wr: Vec<f64> = (0..h).map(|k| (k as f64 * 0.3).cos()).collect();
            let wi: Vec<f64> = (0..h).map(|k| (k as f64 * 0.3).sin()).collect();
            let mut r = a.clone();
            let [tr, ti, br, bi] = &mut a;
            fft_butterfly_dispatch(tr, ti, br, bi, &wr, &wi);
            let [tr, ti, br, bi] = &mut r;
            scalar::fft_butterfly(tr, ti, br, bi, &wr, &wi);
            for (x, y) in a.iter().zip(&r) {
                for (p, s) in x.iter().zip(y) {
                    assert!((p - s).abs() < 1e-12, "radix-2 SIMD vs scalar at h={h}");
                }
            }
        }
    }

    // ── Dot product boundary tests ─────────────────────────────────

    #[test]
    fn dot_f64_boundary_lengths() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let a: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
            let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64 * 0.5).collect();
            let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let result = dot_dispatch(&a, &b);
            assert!(
                (result - expected).abs() < 1e-10,
                "dot f64 n={n}: got {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn dot_f32_boundary_lengths() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let a: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
            let b: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.5).collect();
            let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let result = dot_dispatch(&a, &b);
            assert!(
                (result - expected).abs() < 1e-4,
                "dot f32 n={n}: got {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn dot_integer_fallback() {
        let a = vec![1_i32, 2, 3, 4, 5];
        let b = vec![6_i32, 7, 8, 9, 10];
        let result = dot_dispatch(&a, &b);
        assert_eq!(result, 6 + 2 * 7 + 3 * 8 + 4 * 9 + 5 * 10);
    }

    // ── Matmul boundary tests ──────────────────────────────────────

    #[test]
    fn matmul_f64_boundary_sizes() {
        for size in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let n = size;
            let a: Vec<f64> = (0..n * n).map(|i| (i + 1) as f64).collect();
            let b: Vec<f64> = (0..n * n).map(|i| (i + 1) as f64 * 0.1).collect();
            let mut c = vec![0.0_f64; n * n];
            let mut c_ref = vec![0.0_f64; n * n];

            matmul_dispatch(&a, &b, &mut c, n, n, n);
            scalar::matmul(&a, &b, &mut c_ref, n, n, n);

            for i in 0..n * n {
                assert!(
                    (c[i] - c_ref[i]).abs() < 1e-8,
                    "matmul f64 n={n} idx={i}: got {}, expected {}",
                    c[i],
                    c_ref[i]
                );
            }
        }
    }

    #[test]
    fn matmul_f32_boundary_sizes() {
        for size in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let n = size;
            let a: Vec<f32> = (0..n * n).map(|i| (i + 1) as f32).collect();
            let b: Vec<f32> = (0..n * n).map(|i| (i + 1) as f32 * 0.1).collect();
            let mut c = vec![0.0_f32; n * n];
            let mut c_ref = vec![0.0_f32; n * n];

            matmul_dispatch(&a, &b, &mut c, n, n, n);
            scalar::matmul(&a, &b, &mut c_ref, n, n, n);

            for i in 0..n * n {
                assert!(
                    (c[i] - c_ref[i]).abs() < 1e-2,
                    "matmul f32 n={n} idx={i}: got {}, expected {}",
                    c[i],
                    c_ref[i]
                );
            }
        }
    }

    #[test]
    fn matmul_non_square_f64() {
        // (3×5) * (5×7) → (3×7)
        let m = 3;
        let n = 5;
        let p = 7;
        let a: Vec<f64> = (0..m * n).map(|i| (i + 1) as f64).collect();
        let b: Vec<f64> = (0..n * p).map(|i| (i + 1) as f64 * 0.1).collect();
        let mut c = vec![0.0_f64; m * p];
        let mut c_ref = vec![0.0_f64; m * p];

        matmul_dispatch(&a, &b, &mut c, m, n, p);
        scalar::matmul(&a, &b, &mut c_ref, m, n, p);

        for i in 0..m * p {
            assert!(
                (c[i] - c_ref[i]).abs() < 1e-10,
                "matmul non-square idx={i}: got {}, expected {}",
                c[i],
                c_ref[i]
            );
        }
    }

    #[test]
    fn matmul_integer_fallback() {
        // Column-major 2×2: A=[[1,2],[3,4]] stored as [1,3,2,4]
        // B=[[5,6],[7,8]] stored as [5,7,6,8]
        // C=A*B=[[19,22],[43,50]] stored as [19,43,22,50]
        let a = vec![1_i32, 3, 2, 4];
        let b = vec![5_i32, 7, 6, 8];
        let mut c = vec![0_i32; 4];
        matmul_dispatch(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, vec![19, 43, 22, 50]);
    }

    // ── Element-wise ops boundary tests ────────────────────────────

    #[test]
    fn add_slices_f64_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let a: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let b: Vec<f64> = (0..n).map(|i| (i * 10) as f64).collect();
            let mut out = vec![0.0_f64; n];

            add_slices_dispatch(&a, &b, &mut out);

            for i in 0..n {
                assert_eq!(out[i], a[i] + b[i], "add f64 n={n} idx={i}");
            }
        }
    }

    #[test]
    fn sub_slices_f64_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let a: Vec<f64> = (0..n).map(|i| (i * 10) as f64).collect();
            let b: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let mut out = vec![0.0_f64; n];

            sub_slices_dispatch(&a, &b, &mut out);

            for i in 0..n {
                assert_eq!(out[i], a[i] - b[i], "sub f64 n={n} idx={i}");
            }
        }
    }

    #[test]
    fn scale_slices_f64_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let a: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
            let mut out = vec![0.0_f64; n];

            scale_slices_dispatch(&a, 3.0, &mut out);

            for i in 0..n {
                assert_eq!(out[i], a[i] * 3.0, "scale f64 n={n} idx={i}");
            }
        }
    }

    #[test]
    fn add_slices_f32_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..n).map(|i| (i * 10) as f32).collect();
            let mut out = vec![0.0_f32; n];

            add_slices_dispatch(&a, &b, &mut out);

            for i in 0..n {
                assert_eq!(out[i], a[i] + b[i], "add f32 n={n} idx={i}");
            }
        }
    }

    #[test]
    fn sub_slices_f32_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let a: Vec<f32> = (0..n).map(|i| (i * 10) as f32).collect();
            let b: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let mut out = vec![0.0_f32; n];

            sub_slices_dispatch(&a, &b, &mut out);

            for i in 0..n {
                assert_eq!(out[i], a[i] - b[i], "sub f32 n={n} idx={i}");
            }
        }
    }

    #[test]
    fn scale_slices_f32_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let a: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
            let mut out = vec![0.0_f32; n];

            scale_slices_dispatch(&a, 3.0_f32, &mut out);

            for i in 0..n {
                assert_eq!(out[i], a[i] * 3.0, "scale f32 n={n} idx={i}");
            }
        }
    }

    #[test]
    fn scale_in_place_f64_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let mut a: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
            let expected: Vec<f64> = a.iter().map(|x| x * 3.0).collect();

            scale_in_place_dispatch(&mut a, 3.0);

            for i in 0..n {
                assert_eq!(a[i], expected[i], "scale_in_place f64 n={n} idx={i}");
            }
        }
    }

    #[test]
    fn scale_in_place_f32_boundary() {
        for n in [
            0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            let mut a: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
            let expected: Vec<f32> = a.iter().map(|x| x * 3.0).collect();

            scale_in_place_dispatch(&mut a, 3.0_f32);

            for i in 0..n {
                assert_eq!(a[i], expected[i], "scale_in_place f32 n={n} idx={i}");
            }
        }
    }

    #[test]
    fn scale_in_place_integer_fallback() {
        let mut a = vec![1_i32, 2, 3, 4, 5];
        scale_in_place_dispatch(&mut a, 3);
        assert_eq!(a, vec![3, 6, 9, 12, 15]);
    }

    // ── AXPY boundary tests ───────────────────────────────────────────

    #[test]
    fn axpy_neg_f64_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
            let alpha = 2.5_f64;
            let mut y: Vec<f64> = (0..n).map(|i| (i * 10) as f64).collect();
            let expected: Vec<f64> = y
                .iter()
                .zip(x.iter())
                .map(|(yi, xi)| yi - alpha * xi)
                .collect();

            axpy_neg_dispatch(&mut y, alpha, &x);

            for i in 0..n {
                assert!(
                    (y[i] - expected[i]).abs() < 1e-10,
                    "axpy f64 n={n} idx={i}: got {}, expected {}",
                    y[i],
                    expected[i]
                );
            }
        }
    }

    #[test]
    fn axpy_neg_f32_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let x: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
            let alpha = 2.5_f32;
            let mut y: Vec<f32> = (0..n).map(|i| (i * 10) as f32).collect();
            let expected: Vec<f32> = y
                .iter()
                .zip(x.iter())
                .map(|(yi, xi)| yi - alpha * xi)
                .collect();

            axpy_neg_dispatch(&mut y, alpha, &x);

            for i in 0..n {
                assert!(
                    (y[i] - expected[i]).abs() < 1e-4,
                    "axpy f32 n={n} idx={i}: got {}, expected {}",
                    y[i],
                    expected[i]
                );
            }
        }
    }

    #[test]
    fn axpy_neg_integer_fallback() {
        let x = vec![1_i32, 2, 3, 4, 5];
        let mut y = vec![10_i32, 20, 30, 40, 50];
        axpy_neg_dispatch(&mut y, 3, &x);
        assert_eq!(y, vec![7, 14, 21, 28, 35]);
    }

    // ── AXPY positive boundary tests ─────────────────────────────────

    #[test]
    fn axpy_pos_f64_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
            let alpha = 2.5_f64;
            let mut y: Vec<f64> = (0..n).map(|i| (i * 10) as f64).collect();
            let expected: Vec<f64> = y
                .iter()
                .zip(x.iter())
                .map(|(yi, xi)| yi + alpha * xi)
                .collect();

            axpy_pos_dispatch(&mut y, alpha, &x);

            for i in 0..n {
                assert!(
                    (y[i] - expected[i]).abs() < 1e-10,
                    "axpy_pos f64 n={n} idx={i}: got {}, expected {}",
                    y[i],
                    expected[i]
                );
            }
        }
    }

    #[test]
    fn axpy_pos_f32_boundary() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
            let x: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
            let alpha = 2.5_f32;
            let mut y: Vec<f32> = (0..n).map(|i| (i * 10) as f32).collect();
            let expected: Vec<f32> = y
                .iter()
                .zip(x.iter())
                .map(|(yi, xi)| yi + alpha * xi)
                .collect();

            axpy_pos_dispatch(&mut y, alpha, &x);

            for i in 0..n {
                assert!(
                    (y[i] - expected[i]).abs() < 1e-4,
                    "axpy_pos f32 n={n} idx={i}: got {}, expected {}",
                    y[i],
                    expected[i]
                );
            }
        }
    }

    #[test]
    fn axpy_pos_integer_fallback() {
        let x = vec![1_i32, 2, 3, 4, 5];
        let mut y = vec![10_i32, 20, 30, 40, 50];
        axpy_pos_dispatch(&mut y, 3, &x);
        assert_eq!(y, vec![13, 26, 39, 52, 65]);
    }

    // ── conv1d boundary tests ─────────────────────────────────────────

    #[test]
    fn conv1d_f64_boundary_lengths() {
        // Output lengths spanning the scalar tail, single-vector, and
        // 4-vector block paths for every ISA width; strides 1 and 3.
        let kernel = [0.25_f64, 0.5, 0.25, 0.125, 0.375];
        for stride in [1_usize, 3] {
            for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 65] {
                let src_len = n + (kernel.len() - 1) * stride;
                let src: Vec<f64> = (0..src_len).map(|i| (i as f64) * 0.5 - 3.0).collect();
                let mut out = vec![0.0_f64; n];
                let mut out_ref = vec![0.0_f64; n];

                conv1d_dispatch(&mut out, &src, &kernel, stride);
                scalar::conv1d(&mut out_ref, &src, &kernel, stride);

                for i in 0..n {
                    assert!(
                        (out[i] - out_ref[i]).abs() < 1e-12,
                        "conv1d f64 n={n} stride={stride} idx={i}: got {}, expected {}",
                        out[i],
                        out_ref[i]
                    );
                }
            }
        }
    }

    #[test]
    fn conv1d_f32_boundary_lengths() {
        let kernel = [0.25_f32, 0.5, 0.25];
        for stride in [1_usize, 4] {
            for n in [0, 1, 3, 4, 5, 15, 16, 17, 63, 64, 65, 129] {
                let src_len = n + (kernel.len() - 1) * stride;
                let src: Vec<f32> = (0..src_len).map(|i| (i as f32) * 0.5 - 3.0).collect();
                let mut out = vec![0.0_f32; n];
                let mut out_ref = vec![0.0_f32; n];

                conv1d_dispatch(&mut out, &src, &kernel, stride);
                scalar::conv1d(&mut out_ref, &src, &kernel, stride);

                for i in 0..n {
                    assert!(
                        (out[i] - out_ref[i]).abs() < 1e-4,
                        "conv1d f32 n={n} stride={stride} idx={i}: got {}, expected {}",
                        out[i],
                        out_ref[i]
                    );
                }
            }
        }
    }

    #[test]
    fn conv1d_integer_fallback() {
        let src = vec![1_i32, 2, 3, 4, 5, 6];
        let kernel = vec![1_i32, -2, 1];
        let mut out = vec![0_i32; 4];
        conv1d_dispatch(&mut out, &src, &kernel, 1);
        // out[i] = src[i] - 2*src[i+1] + src[i+2] = 0 for a linear ramp.
        assert_eq!(out, vec![0, 0, 0, 0]);
    }

    // ── x86_64 tiers: every tier this CPU supports vs the scalar reference ──
    //
    // The `*_dispatch` tests above exercise whichever tier `isa()` picks; these
    // call each tier's kernels directly so that a CPU with AVX-512 checks all
    // three, and so that runtime detection is compared against `std`'s.

    #[cfg(all(target_arch = "x86_64", feature = "runtime-dispatch"))]
    mod x86_tiers {
        use super::super::{f32_avx, f32_avx512, f32_sse2, f64_avx, f64_avx512, f64_sse2};
        use super::super::{isa, scalar, Isa};

        #[test]
        fn isa_agrees_with_std_detection() {
            let expected = if std::is_x86_feature_detected!("avx512f") {
                Isa::Avx512
            } else if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
                Isa::Avx
            } else {
                Isa::Sse2
            };
            assert_eq!(isa(), expected);
            // Second call takes the cached path.
            assert_eq!(isa(), expected);
            // The compile-time floor is never lowered by the probe.
            if cfg!(target_feature = "avx512f") {
                assert_eq!(isa(), Isa::Avx512);
            } else if cfg!(all(target_feature = "avx", target_feature = "fma")) {
                assert!(isa() >= Isa::Avx);
            }
        }

        /// Deterministic pseudo-random values in [-1, 1).
        fn seq<T: From<f32>>(n: usize, seed: u32) -> Vec<T> {
            (0..n)
                .map(|i| {
                    let x = (i as u32)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(seed.wrapping_mul(40_503))
                        >> 16;
                    T::from((x % 2000) as f32 / 1000.0 - 1.0)
                })
                .collect()
        }

        fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
            let scale = 1.0 + got.abs().max(want.abs());
            assert!(
                (got - want).abs() <= tol * scale,
                "{what}: got {got}, want {want}"
            );
        }

        fn assert_all_close<T: Copy + Into<f64>>(got: &[T], want: &[T], tol: f64, what: &str) {
            assert_eq!(got.len(), want.len(), "{what}: length");
            for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
                assert_close(g.into(), w.into(), tol, &format!("{what}[{i}]"));
            }
        }

        /// Every kernel of one tier module against `scalar`, for one element type.
        macro_rules! battery {
            ($m:ident, $t:ty, $tol:expr) => {{
                let tol: f64 = $tol;
                let name = stringify!($m);
                let alpha: $t = <$t>::from(0.75f32);
                let lens = [
                    0usize, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 129,
                ];
                for &n in &lens {
                    let a = seq::<$t>(n, 1);
                    let b = seq::<$t>(n, 2);
                    let what = format!("{name} n={n}");

                    let want: f64 = scalar::dot(&a, &b).into();
                    let got: f64 = $m::dot(&a, &b).into();
                    assert_close(got, want, tol, &format!("{what} dot"));

                    let (mut want, mut got) =
                        (vec![<$t>::from(0.0f32); n], vec![<$t>::from(0.0f32); n]);
                    scalar::add_slices(&a, &b, &mut want);
                    $m::add_slices(&a, &b, &mut got);
                    assert_all_close(&got, &want, tol, &format!("{what} add"));
                    scalar::sub_slices(&a, &b, &mut want);
                    $m::sub_slices(&a, &b, &mut got);
                    assert_all_close(&got, &want, tol, &format!("{what} sub"));
                    scalar::scale_slices(&a, alpha, &mut want);
                    $m::scale_slices(&a, alpha, &mut got);
                    assert_all_close(&got, &want, tol, &format!("{what} scale"));

                    let (mut want, mut got) = (a.clone(), a.clone());
                    scalar::scale_assign_slices(&mut want, alpha);
                    $m::scale_in_place(&mut got, alpha);
                    assert_all_close(&got, &want, tol, &format!("{what} scale_in_place"));

                    let (mut want, mut got) = (a.clone(), a.clone());
                    scalar::axpy_neg(&mut want, alpha, &b);
                    $m::axpy_neg(&mut got, alpha, &b);
                    assert_all_close(&got, &want, tol, &format!("{what} axpy_neg"));
                    let (mut want, mut got) = (a.clone(), a.clone());
                    scalar::axpy_pos(&mut want, alpha, &b);
                    $m::axpy_pos(&mut got, alpha, &b);
                    assert_all_close(&got, &want, tol, &format!("{what} axpy_pos"));
                }

                for &(m, n, p) in &[
                    (1usize, 1usize, 1usize),
                    (2, 3, 4),
                    (4, 4, 4),
                    (5, 7, 3),
                    (8, 8, 8),
                    (9, 5, 6),
                    (16, 4, 4),
                    (17, 9, 5),
                    (33, 7, 4),
                    (12, 300, 5),
                ] {
                    let a = seq::<$t>(m * n, 3);
                    let b = seq::<$t>(n * p, 4);
                    let (mut want, mut got) = (
                        vec![<$t>::from(0.0f32); m * p],
                        vec![<$t>::from(0.0f32); m * p],
                    );
                    scalar::matmul(&a, &b, &mut want, m, n, p);
                    $m::matmul(&a, &b, &mut got, m, n, p);
                    assert_all_close(&got, &want, tol, &format!("{name} matmul {m}x{n}x{p}"));
                }

                for &(n, k, stride) in &[
                    (1usize, 1usize, 1usize),
                    (20, 3, 1),
                    (17, 5, 4),
                    (64, 7, 1),
                    (33, 4, 9),
                ] {
                    let src = seq::<$t>(n + (k - 1) * stride, 5);
                    let kernel = seq::<$t>(k, 6);
                    let (mut want, mut got) =
                        (vec![<$t>::from(0.0f32); n], vec![<$t>::from(0.0f32); n]);
                    scalar::conv1d(&mut want, &src, &kernel, stride);
                    $m::conv1d(&mut got, &src, &kernel, stride);
                    assert_all_close(
                        &got,
                        &want,
                        tol,
                        &format!("{name} conv1d n={n} k={k} stride={stride}"),
                    );
                }

                for &h in &[1usize, 2, 3, 5, 8, 9, 16, 17, 33] {
                    let mut want: Vec<Vec<$t>> = (0..4).map(|s| seq::<$t>(h, 10 + s)).collect();
                    let mut got = want.clone();
                    let (wr, wi) = (seq::<$t>(h, 20), seq::<$t>(h, 21));
                    let [tr, ti, br, bi] = &mut want[..] else {
                        unreachable!()
                    };
                    scalar::fft_butterfly(tr, ti, br, bi, &wr, &wi);
                    let [tr, ti, br, bi] = &mut got[..] else {
                        unreachable!()
                    };
                    $m::fft_butterfly(tr, ti, br, bi, &wr, &wi);
                    for (g, w) in got.iter().zip(&want) {
                        assert_all_close(g, w, tol, &format!("{name} fft_butterfly h={h}"));
                    }
                }

                for &q in &[1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33] {
                    let mut want: Vec<Vec<$t>> = (0..8).map(|s| seq::<$t>(q, 30 + s)).collect();
                    let mut got = want.clone();
                    let w: Vec<Vec<$t>> = (0..6).map(|s| seq::<$t>(q, 40 + s)).collect();
                    let [ar, ai, br, bi, cr, ci, dr, di] = &mut want[..] else {
                        unreachable!()
                    };
                    scalar::fft_butterfly4(
                        ar, ai, br, bi, cr, ci, dr, di, &w[0], &w[1], &w[2], &w[3], &w[4], &w[5],
                    );
                    let [ar, ai, br, bi, cr, ci, dr, di] = &mut got[..] else {
                        unreachable!()
                    };
                    $m::fft_butterfly4(
                        ar, ai, br, bi, cr, ci, dr, di, &w[0], &w[1], &w[2], &w[3], &w[4], &w[5],
                    );
                    for (g, w) in got.iter().zip(&want) {
                        assert_all_close(g, w, tol, &format!("{name} fft_butterfly4 q={q}"));
                    }
                }
            }};
        }

        macro_rules! tier_test {
            ($name:ident, $f64m:ident, $f32m:ident, $tier:expr) => {
                #[test]
                // The `unsafe` below is redundant for the SSE2 tier, whose
                // kernels are unattributed baseline functions.
                #[allow(unused_unsafe)]
                fn $name() {
                    if isa() < $tier {
                        eprintln!(
                            "skipping {}: this CPU has no {:?}",
                            stringify!($name),
                            $tier
                        );
                        return;
                    }
                    // SAFETY: `isa() >= $tier` was just established, and `isa()`
                    // reports a tier only when it is a compile-time target feature
                    // or `is_x86_feature_detected!` confirmed the CPU supports it —
                    // the precondition for calling the tier's `#[target_feature]`
                    // kernels from this baseline-compiled test.
                    unsafe {
                        battery!($f64m, f64, 1e-12);
                        battery!($f32m, f32, 1e-4);
                    }
                }
            };
        }

        tier_test!(sse2_tier_matches_scalar, f64_sse2, f32_sse2, Isa::Sse2);
        tier_test!(avx_tier_matches_scalar, f64_avx, f32_avx, Isa::Avx);
        tier_test!(
            avx512_tier_matches_scalar,
            f64_avx512,
            f32_avx512,
            Isa::Avx512
        );
    }
}
