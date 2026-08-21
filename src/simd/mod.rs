//! SIMD-accelerated kernels with compile-time architecture dispatch.
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
//! `-C target-feature=+avx2` etc.
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
//! | `x86_64`  | AVX       | 8×4      | 16×4     |
//! | `x86_64`  | AVX-512   | 16×4     | 32×4     |
//! | other     | scalar    | 4×4      | 4×4      |

// Each architecture file provides a full set of ISA kernels (dot, matmul, AXPY,
// …). The compile-time dispatch selects the widest available ISA, so the
// lower-ISA kernels (e.g. SSE2 when AVX is enabled) are deliberately present but
// unused in that build — not removable, just inactive for this target.
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
    ($t:ty, $lanes:expr, $load:ident, $store:ident, $add:ident, $sub:ident, $mul:ident, $set1:ident) => {
        /// Element-wise addition: out[i] = a[i] + b[i].
        #[inline]
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

/// AXPY kernels using a separate multiply + add/subtract (x86 SSE2/AVX/AVX-512).
// Unused on aarch64 (which uses the fused variant below); the reverse holds on x86.
#[allow(unused_macros)]
macro_rules! simd_axpy_kernels_muladd {
    ($t:ty, $lanes:expr, $load:ident, $store:ident, $add:ident, $sub:ident, $mul:ident, $set1:ident) => {
        /// AXPY: y[i] -= alpha * x[i].
        #[inline]
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

/// AXPY kernels using NEON fused multiply-add / multiply-subtract.
#[allow(unused_macros)]
macro_rules! simd_axpy_kernels_fma {
    ($t:ty, $lanes:expr, $load:ident, $store:ident, $fma:ident, $fms:ident, $dup:ident) => {
        /// AXPY: y[i] -= alpha * x[i].
        #[inline]
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
    ($t:ty, $lanes:expr, $load:ident, $store:ident, $fma:ident, $dup:ident) => {
        /// Strided 1D correlation: out[i] = Σ_k kernel[k] · src[i + k·stride].
        ///
        /// # Panics
        ///
        /// Panics unless `stride >= 1` and `src` covers every window —
        /// `src.len() >= out.len() + (kernel.len() - 1) * stride`. This is checked
        /// once per call (not per element) because the strided loads below rely on
        /// it in release builds, not just under `debug_assertions`.
        #[inline]
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
    ($t:ty, $lanes:expr, $load:ident, $store:ident, $add:ident, $mul:ident, $set1:ident) => {
        /// Strided 1D correlation: out[i] = Σ_k kernel[k] · src[i + k·stride].
        ///
        /// # Panics
        ///
        /// Panics unless `stride >= 1` and `src` covers every window —
        /// `src.len() >= out.len() + (kernel.len() - 1) * stride`. This is checked
        /// once per call (not per element) because the strided loads below rely on
        /// it in release builds, not just under `debug_assertions`.
        #[inline]
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

pub(crate) mod scalar;

#[cfg(target_arch = "aarch64")]
pub(crate) mod f32_neon;
#[cfg(target_arch = "aarch64")]
pub(crate) mod f64_neon;

#[cfg(target_arch = "x86_64")]
pub(crate) mod f32_sse2;
#[cfg(target_arch = "x86_64")]
pub(crate) mod f64_sse2;

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
pub(crate) mod f32_avx;
#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
pub(crate) mod f64_avx;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub(crate) mod f32_avx512;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
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
            #[cfg(target_feature = "avx512f")]
            let result = f64_avx512::dot(a, b);
            #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
            let result = f64_avx::dot(a, b);
            #[cfg(not(target_feature = "avx"))]
            let result = f64_sse2::dot(a, b);
            return w.value_back(result);
        }
        if let Some(w) = TypeEq::<T, f32>::new() {
            let (a, b) = (w.slice(a), w.slice(b));
            #[cfg(target_feature = "avx512f")]
            let result = f32_avx512::dot(a, b);
            #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
            let result = f32_avx::dot(a, b);
            #[cfg(not(target_feature = "avx"))]
            let result = f32_sse2::dot(a, b);
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
            #[cfg(target_feature = "avx512f")]
            f64_avx512::matmul(a, b, c, m, n, p);
            #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
            f64_avx::matmul(a, b, c, m, n, p);
            #[cfg(not(target_feature = "avx"))]
            f64_sse2::matmul(a, b, c, m, n, p);
            return;
        }
        if let Some(w) = TypeEq::<T, f32>::new() {
            let (a, b, c) = (w.slice(a), w.slice(b), w.slice_mut(c));
            #[cfg(target_feature = "avx512f")]
            f32_avx512::matmul(a, b, c, m, n, p);
            #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
            f32_avx::matmul(a, b, c, m, n, p);
            #[cfg(not(target_feature = "avx"))]
            f32_sse2::matmul(a, b, c, m, n, p);
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
                    #[cfg(target_feature = "avx512f")]
                    f64_avx512::$kernel(a, b, out);
                    #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
                    f64_avx::$kernel(a, b, out);
                    #[cfg(not(target_feature = "avx"))]
                    f64_sse2::$kernel(a, b, out);
                    return;
                }
                if let Some(w) = TypeEq::<T, f32>::new() {
                    let (a, b, out) = (w.slice(a), w.slice(b), w.slice_mut(out));
                    #[cfg(target_feature = "avx512f")]
                    f32_avx512::$kernel(a, b, out);
                    #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
                    f32_avx::$kernel(a, b, out);
                    #[cfg(not(target_feature = "avx"))]
                    f32_sse2::$kernel(a, b, out);
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
                    #[cfg(target_feature = "avx512f")]
                    f64_avx512::$kernel(a, s, out);
                    #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
                    f64_avx::$kernel(a, s, out);
                    #[cfg(not(target_feature = "avx"))]
                    f64_sse2::$kernel(a, s, out);
                    return;
                }
                if let Some(w) = TypeEq::<T, f32>::new() {
                    let (a, s, out) = (w.slice(a), w.value(scalar), w.slice_mut(out));
                    #[cfg(target_feature = "avx512f")]
                    f32_avx512::$kernel(a, s, out);
                    #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
                    f32_avx::$kernel(a, s, out);
                    #[cfg(not(target_feature = "avx"))]
                    f32_sse2::$kernel(a, s, out);
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
                    #[cfg(target_feature = "avx512f")]
                    f64_avx512::$kernel(a, s);
                    #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
                    f64_avx::$kernel(a, s);
                    #[cfg(not(target_feature = "avx"))]
                    f64_sse2::$kernel(a, s);
                    return;
                }
                if let Some(w) = TypeEq::<T, f32>::new() {
                    let (s, a) = (w.value(scalar), w.slice_mut(a));
                    #[cfg(target_feature = "avx512f")]
                    f32_avx512::$kernel(a, s);
                    #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
                    f32_avx::$kernel(a, s);
                    #[cfg(not(target_feature = "avx"))]
                    f32_sse2::$kernel(a, s);
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
                    #[cfg(target_feature = "avx512f")]
                    f64_avx512::$kernel(y, al, x);
                    #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
                    f64_avx::$kernel(y, al, x);
                    #[cfg(not(target_feature = "avx"))]
                    f64_sse2::$kernel(y, al, x);
                    return;
                }
                if let Some(w) = TypeEq::<T, f32>::new() {
                    let (y, al, x) = (w.slice_mut(y), w.value(alpha), w.slice(x));
                    #[cfg(target_feature = "avx512f")]
                    f32_avx512::$kernel(y, al, x);
                    #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
                    f32_avx::$kernel(y, al, x);
                    #[cfg(not(target_feature = "avx"))]
                    f32_sse2::$kernel(y, al, x);
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
            #[cfg(target_feature = "avx512f")]
            f64_avx512::conv1d(out, src, kernel, stride);
            #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
            f64_avx::conv1d(out, src, kernel, stride);
            #[cfg(not(target_feature = "avx"))]
            f64_sse2::conv1d(out, src, kernel, stride);
            return;
        }
        if let Some(w) = TypeEq::<T, f32>::new() {
            let (out, src, kernel) = (w.slice_mut(out), w.slice(src), w.slice(kernel));
            #[cfg(target_feature = "avx512f")]
            f32_avx512::conv1d(out, src, kernel, stride);
            #[cfg(all(target_feature = "avx", not(target_feature = "avx512f")))]
            f32_avx::conv1d(out, src, kernel, stride);
            #[cfg(not(target_feature = "avx"))]
            f32_sse2::conv1d(out, src, kernel, stride);
            return;
        }
    }
    scalar::conv1d(out, src, kernel, stride);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

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
}
