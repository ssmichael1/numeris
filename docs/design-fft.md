# FFT module — design notes

Status: **1D + 2D implemented** (fixed/dyn complex, Bluestein, rfft/irfft, convolve, shift;
`DynFft2`/`DynRealFft2` + `fftshift2d`/`ifftshift2d`). Feature flag: `fft`. The 2D landing
took recommendation (1) — gather/scatter on the strided row axis, reusing the per-column
`DynFft` directly. Still open as pure optimizations: transpose-based passes (both axes
contiguous) and `rayon` per-row/column parallelism (needs a scratch-external transform API,
since `DynFft::forward` currently takes `&mut self`); and `fft_convolve2d` for imageproc's
large-kernel path.

## Multi-dimensional (2D) FFT — design

### The algorithm is separable (row–column)

An N×M 2D DFT factors into 1D transforms: FFT every row, then FFT every column
(order doesn't matter). No new kernel is needed — 2D is entirely built on the 1D
[`DynFft`] we already have. Cost is `O(N·M·log(N·M))`; two reusable 1D plans
(one length-M for rows, one length-N for columns) cover any dimensions, including
non-power-of-two (Bluestein handles those transparently).

### The layout question is the whole design

`DynMatrix` is **column-major**: element `(row, col)` lives at `col*nrows + row`,
so each *column* is contiguous. That means:

- **Column transforms** read/write contiguous memory → cache- and SIMD-friendly,
  can be sliced straight out of the backing `Vec` and handed to `DynFft`.
- **Row transforms** are strided (stride = `nrows`) → cache-hostile, and can't be
  passed to `DynFft` (which wants a contiguous `&mut [Complex<T>]`).

Two ways to handle the strided pass:

1. **Gather/scatter** each row into a length-M scratch buffer, transform, scatter
   back. Simple; one extra copy per row.
2. **Transpose between passes** (`out = T(F_col(T(F_col(A)))))`): transform the
   contiguous dimension, transpose, transform the (now contiguous) other
   dimension, transpose back. Both FFT passes are contiguous → best for SIMD and
   the eventual `rayon` per-column parallelism, at the cost of two transposes.
   This is what FFTW/pocketfft do for large arrays.

**Recommendation:** start with (1) gather/scatter (correct, simple, reuses the
per-column `DynFft` directly); offer (2) as an optimization once the SIMD
butterflies land, since transpose-based makes *both* passes hit the vectorized
contiguous path. Both parallelize cleanly over independent rows/columns via the
existing `par` module (each row/column FFT is disjoint output — same pattern as
`imageproc`).

### API sketch

```rust
#[cfg(feature = "alloc")]
pub struct DynFft2<T: FloatScalar> {
    rows: usize,
    cols: usize,
    row_plan: DynFft<T>,   // length = cols
    col_plan: DynFft<T>,   // length = rows
    scratch: Vec<Complex<T>>,
}
impl<T: FloatScalar> DynFft2<T> {
    pub fn new(rows: usize, cols: usize) -> Self;
    pub fn forward(&mut self, data: &mut DynMatrix<Complex<T>>); // in place
    pub fn inverse(&mut self, data: &mut DynMatrix<Complex<T>>); // 1/(rows*cols)
}
```

`DynMatrix<Complex<T>>` already exists (`DynMatrixz64`/`z32` aliases), and its
column-major backing `Vec` is exactly what the column plan slices into.

### Real 2D input (`rfft2`)

For a real image, transform the first axis with `rfft` (yields `N/2+1` complex
rows), then a full complex FFT along the other axis → an `(N/2+1)×M` complex
result, Hermitian along the reduced axis. ~2× saving, same as 1D. This is the
form `imageproc` would want.

### `fftshift2d` / quadrant swap

2D `fftshift` swaps diagonal quadrants — equivalent to applying the 1D
[`fftshift`] along each axis (rows then columns). Can reuse the 1D rotation per
row/column, or do a dedicated quadrant swap.

### Rayon — parallelize the batch axis, not a single transform

**A single 1D FFT should not use rayon.** Cooley–Tukey is `log N` stages with a hard
barrier between each (stage `s+1` reads stage `s`'s output). Within a stage the butterflies
are independent, but the work is only `O(N)` per stage, so you'd pay `log N` rayon joins
(~1–10 µs each) against microsecond-scale stages — sync dominates until `N` is in the
hundreds of thousands to millions, well above the numeris size range. It also cuts against
the repo's own rule (`par` only parallelizes *disjoint-output* work, "never order-sensitive
reductions") — a single transform's stages are exactly an order-sensitive chain. RustFFT
likewise does not multithread a single transform.

**Batched / 2D FFTs are the right place — and it's embarrassingly parallel.** The 2D
row–column algorithm is nothing but two batches of independent 1D transforms (all rows,
then all columns); each row/column FFT is disjoint output, the exact
`par::for_each_chunk_mut` pattern `imageproc` already uses for its per-column kernels. So:

- `DynFft2` parallelizes over rows in the first pass and over columns in the second, gated
  on per-pass work via the shared `par::work_col_threshold` helper — same as the imageproc
  filters. Near-linear speedup on large images, zero cross-thread coordination.
- The same applies to any batched 1D API (a spectrogram's windows, batched `rfft`).
- The plan itself (`DynFft`/twiddles) is read-only during a transform, so it can be shared
  across threads (`&`); only the per-row/column scratch is thread-local.

`rayon` implies `std`, so this only ever touches the alloc/`DynFft` tier; the no-alloc
fixed tier is unaffected. Net: **no rayon in the 1D core; add it with the 2D / batched
work** (task #8), following the existing `par` gating.

### Why this matters for `imageproc`

The payoff is **FFT-based 2D convolution/correlation**: `imageproc` currently
does spatial convolution (`O(N·M·k²)` for a k×k kernel), which is the right
choice for small kernels but loses badly for large ones. An `fft_convolve2d`
(multiply in the frequency domain) is `O(N·M·log(N·M))` regardless of kernel
size — the standard win for large-kernel Gaussian blur, template matching, and
LoG/DoG at big radii. That's the natural first consumer of `DynFft2`.

### Scope call

2D is **deferred to a follow-up** — it's purely additive (a new `fft2` submodule
built on the finished 1D core), and the 1D landing is already large. Tracked
here so the layout decision (column-major → column-first, gather/scatter rows,
transpose-optimize later) is recorded before it's built.

---

## 0. Locked scope (first cut)

Decisions made 2026-07-03 — the first cut is the **full-scope** version:

| Decision | Choice | Consequence |
|---|---|---|
| Algorithm coverage | **Power-of-two + arbitrary N** | Fixed tier is power-of-two only; `DynFft` covers arbitrary/prime N via **Bluestein** (chirp-z → power-of-two FFT). Mixed-radix (3,5) is deferred as a pure optimization — Bluestein already covers all N by reusing the tested radix core. |
| Real transform | **rfft/irfft included** | `real.rs` ships in the first cut. |
| Fixed-size cap | **up to 4096** | Const-generic tier supports power-of-two `N` from 2..=4096. Larger → `DynFft`. Document the cap; keep `.text` bounded. |

### SIMD decisions (refined 2026-07-03)

| Decision | Choice | Consequence |
|---|---|---|
| SIMD placement | **Alloc/DynFft tier only** | A *fused* SIMD butterfly wants deinterleaved re/im buffers (`2N` scratch); on the no-alloc fixed tier's stack that undercuts the low-memory point of that tier. So the **fixed tier stays scalar** (its audience is embedded: small N, code-size-sensitive) and SIMD lives in the `DynFft` tier where scratch is already heap-allocated. Folds SIMD work into the `DynFft` step. |
| Fixed-tier radix-4 | **Deferred into the DynFft SIMD kernels** | Radix-4 mainly pays off *with* fused SIMD; combined radix-2/4 is standard inside a SIMD butterfly kernel. The scalar fixed tier stays the proven radix-2 (simpler, smaller `.text`). |
| x86 kernels | **Macro-shared with NEON, CI-validated** | One `simd_fft_butterfly_kernel!($t,$lanes,$load,$store,$add,$sub,$mul)` macro body, invoked by each ISA file (neon / sse2 / avx / avx512) — identical algorithm, only intrinsic names differ. Mirrors the existing `simd_elementwise_kernels!` macro. |

**Implementation status (done):** `SoaPlan` (deinterleaved re/im, staged twiddles) drives
`DynFft`'s power-of-two path through `simd::fft_butterfly_dispatch`. Scalar reference in
`simd::scalar::fft_butterfly`. Validated by running (not just compiling) the kernels:
NEON native on the arm64 dev machine, and **x86 SSE2 + AVX2 executed via Rosetta 2** — all
pass `dynfft_simd_path_matches_scalar_reference` at 1e-11. AVX-512 compiles for all
target-features but can't run under Rosetta, so it alone is CI-confirmed. Inverse FFT reuses
the accelerated forward via the conjugate trick, so it's SIMD-accelerated too.

This is a large first landing. §9 rollout order below is re-sequenced accordingly — still
build in verifiable increments internally, but everything lands together (or as a stacked
PR series) rather than incrementally shipping to users.

## 1. Positioning — do not try to beat FFTW

FFTW is not the competitive set for `numeris`. Its speed comes from things that are
structurally incompatible with a no-std / no-alloc / pure-Rust library:

- Runtime **planning + codelet autotuning** (measures the machine, picks a plan) — needs
  `std`, an allocator, a timer, a large codelet database.
- Hand-tuned SIMD **codelets** per radix per ISA.
- Split-radix, Rader, Bluestein, prime-factor for arbitrary sizes.

None of that runs on a Cortex-M. **FFTW isn't even available on the target `numeris` was
built for.** So the real competitive set is:

| Competitor | no_std? | no-alloc? | Pure Rust? | Notes |
|---|---|---|---|---|
| FFTW | ✗ | ✗ | ✗ (C) | Desktop king. Irrelevant on embedded. |
| RustFFT | ✗ (needs `Vec`) | ✗ | ✓ | Planner-based, SIMD; within ~2× of FFTW. |
| microfft | ✓ | ✓ | ✓ | Fixed power-of-two sizes only; minimal. |
| CMSIS-DSP | ✓ | ✓ | ✗ (C) | ARM-only, tables baked in. |
| **numeris fft** | ✓ | ✓ (fixed tier) | ✓ | Integrated with `Complex`, `simd::`, `DynMatrix`. |

**Goal:** be the cleanest pure-Rust FFT that also runs on embedded, integrated with the
rest of the crate. On desktop, land within ~2–4× of FFTW (radix-2/4 + SIMD). That is
entirely acceptable for the audience — they pick `numeris` for portability and zero C
deps, not peak throughput.

Explicitly a **non-goal:** matching FFTW/RustFFT desktop throughput. Say so in the module
rustdoc so expectations are set.

## 2. Two-tier structure (mirrors `estimate` / `optim` / `dynmatrix`)

### Tier 1 — fixed-size, no-alloc, stack (the reason the module exists)

Works on Cortex-M, no heap. In-place, power-of-two `N`.

```rust
// Twiddles precomputed once by the caller; hot loop does no sin/cos.
pub struct TwiddleTable<T: FloatScalar, const N: usize> { /* [Complex<T>; N/2] */ }
impl<T: FloatScalar, const N: usize> TwiddleTable<T, N> {
    pub fn new() -> Self;                 // computes sin/cos once
}

pub fn fft<T: FloatScalar, const N: usize>(buf: &mut [Complex<T>; N], tw: &TwiddleTable<T, N>);
pub fn ifft<T: FloatScalar, const N: usize>(buf: &mut [Complex<T>; N], tw: &TwiddleTable<T, N>);

// Convenience: builds twiddles on entry (fine on desktop, allocates a stack table).
pub fn fft_inplace<T: FloatScalar, const N: usize>(buf: &mut [Complex<T>; N]);
```

- Iterative **Cooley–Tukey**, radix-2 first, then **radix-4** for the ~2× win.
- Bit-reversal permutation + butterfly passes.
- `N` must be a power of two — enforce with a `const { assert!(N.is_power_of_two()) }`
  compile-time check (or a debug assert + documented precondition).
- Twiddles passed in via `TwiddleTable` so embedded users build the table once at init and
  never call `sin`/`cos` in the loop.

### Tier 2 — `alloc` planner (arbitrary sizes, SIMD payoff)

Gated on `alloc` (consistent with `estimate`/`imageproc`). Bluestein needs scratch, which
is the natural reason this tier requires alloc.

```rust
pub struct DynFft<T: FloatScalar> {
    len: usize,
    twiddles: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
    plan: Plan,   // radix chain, or Bluestein/Rader fallback
}
impl<T: FloatScalar> DynFft<T> {
    pub fn new(len: usize) -> Self;              // builds the plan + twiddle cache once
    pub fn forward(&mut self, buf: &mut [Complex<T>]);
    pub fn inverse(&mut self, buf: &mut [Complex<T>]);
}
```

- Mixed-radix Cooley–Tukey (radix 2, 3, 4, 5) for composite `N`.
- **Bluestein** (chirp-z) for prime / awkward `N`; optionally **Rader** for prime `N` later.
- Twiddle + scratch cached in the struct → amortized across repeated transforms (the
  RustFFT planner pattern, minus autotuning).

## 3. Companions (used more than raw complex FFT in practice)

- `rfft` / `irfft` — real-input transform at ~half the cost. For embedded DSP this is the
  *most-used* entry point. Pack real input as N/2 complex + one untangle pass.
- `fftshift` / `ifftshift`.
- FFT-based `convolve` / `correlate` that plugs into `imageproc` (large-kernel path).
- Possibly `czt` (chirp-z) exposed publicly once Bluestein exists — cheap add-on.

## 4. The real trap: monomorphization

Every distinct `N` in the fixed-size tier is a fresh code copy. Radix-2 butterflies fully
unrolled per-`N` bloat `.text` fast — bad on flash-constrained embedded.

- Keep the **core butterfly loop size-generic over a runtime length**; only the entry
  point + `TwiddleTable` are const-generic.
- Cap the supported fixed sizes to a sensible set (≤ 4096) in docs.
- Do **not** const-unroll the whole transform.

## 5. SIMD reuse — needs new kernels, not the existing ones

Complex butterflies do **not** map cleanly onto the current `simd::dot`/`axpy` dispatch.
Those operate on interleaved contiguous slices; butterflies want **deinterleaved** real/imag
lanes (`f64x2` / `f32x4` each holding separate re/im, i.e. structure-of-arrays).

Plan: add a small set of complex-butterfly kernels under `src/simd/` following the existing
`TypeId` compile-time dispatch (scalar fallback for integers/unknown arch, NEON/SSE2 baseline,
AVX/AVX-512 opt-in). Butterfly radix-2/4 kernels:

- `fft_bfly2_dispatch`, `fft_bfly4_dispatch` over `&mut [Complex<T>]` split into re/im.
- Reuse the same feature-gating story as `simd/mod.rs` (always-on, no cargo flag).

## 5b. Public API spec (signed off 2026-07-03)

All four flagged decisions confirmed as recommended: (1) infallible transforms with
`assert`/`const` panics, `FftError` kept for `DynFft::new(0)` + optional `try_*`;
(2) caller-supplied output slice for fixed-tier real FFT; (3) `TwiddleTable` stores full
`[Complex<T>; N]`, `fft_inplace` is the table-free low-memory path; (4) deinterleaved
`re`/`im` scratch for the SIMD butterflies.

### Feature wiring

`Cargo.toml`:
```toml
fft = ["dep:num-complex"]   # complex output is intrinsic; reuse the existing optional dep
```
- Add `"fft"` to the `all = [...]` list.
- Fixed tier is no-alloc. `DynFft` / `bluestein` / `convolve` are additionally gated
  **inside the module** on `feature = "alloc"` (mirrors how `estimate` uses `alloc`).

`src/lib.rs`:
```rust
#[cfg(feature = "fft")]
pub mod fft;
```
Module rustdoc opens with the explicit "this is not FFTW; the target is embedded/no-alloc
portability, expect ~2–4× FFTW on desktop" note.

### Error handling — recommendation: mostly infallible

FFT length errors are **programmer errors**, not runtime conditions, and the hot path is
called repeatedly after a one-time plan. So, deviating slightly from the per-module-error
convention:

- **Fixed tier:** infallible. Power-of-two and `N ≤ 4096` enforced at compile time via
  `const { assert!(N.is_power_of_two() && N <= 4096) }`.
- **Dyn tier:** `DynFft::new(len)` panics on `len == 0` (documented precondition);
  `forward`/`inverse` panic via `assert_eq!` on buffer-length mismatch. No `Result` on the
  per-call hot path.

A `FftError` enum is still defined for the few genuinely fallible/planning surfaces and for
API-symmetry with the rest of the crate:
```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FftError {
    ZeroLength,        // DynFft::new(0)
    LengthMismatch,    // buffer length != planned length
}
// + core::fmt::Display
```
**Decision needed:** infallible-with-panics (recommended) vs. `Result`-returning transforms.
The recommendation is panics for the transforms; keep `FftError` for a fallible
`DynFft::try_new` / `try_forward` pair if you want a non-panicking option too.

### Tier 1 — fixed-size, no-alloc (power-of-two, N ≤ 4096)

```rust
/// Precomputed twiddle factors for repeated same-N transforms.
/// Stores [Complex<T>; N] (uses the first N/2). Full-N storage avoids the
/// unstable `generic_const_exprs` that [Complex<T>; N/2] would require — the
/// crate deliberately never enables that feature.
pub struct TwiddleTable<T: FloatScalar, const N: usize> { /* factors: [Complex<T>; N] */ }

impl<T: FloatScalar, const N: usize> TwiddleTable<T, N> {
    pub fn new() -> Self;   // computes sin/cos once; const asserts N pow2 & <= 4096
}

/// In-place forward FFT using a precomputed table (no sin/cos in the loop — embedded path).
pub fn fft<T: FloatScalar, const N: usize>(buf: &mut [Complex<T>; N], tw: &TwiddleTable<T, N>);
/// In-place inverse FFT (normalized by 1/N).
pub fn ifft<T: FloatScalar, const N: usize>(buf: &mut [Complex<T>; N], tw: &TwiddleTable<T, N>);

/// Convenience: generates stage twiddles inline (incremental rotation, ~log2(N) sin/cos,
/// no persistent table). Lower memory than TwiddleTable; slightly less accurate for large N.
pub fn fft_inplace<T: FloatScalar, const N: usize>(buf: &mut [Complex<T>; N]);
pub fn ifft_inplace<T: FloatScalar, const N: usize>(buf: &mut [Complex<T>; N]);
```
Memory note: `TwiddleTable<f64, 4096>` = 64 KB stack (full-N storage). Fine on capable
targets; memory-constrained callers use `fft_inplace` (no table) or small `N`. Documented.

### Tier 2 — DynFft planner (requires `alloc`; arbitrary N incl. primes)

```rust
#[cfg(feature = "alloc")]
pub struct DynFft<T: FloatScalar> { /* len, plan, twiddles: Vec, scratch: Vec */ }

#[cfg(feature = "alloc")]
impl<T: FloatScalar> DynFft<T> {
    /// Build plan + twiddle/scratch cache once. Mixed-radix (2,3,5); Bluestein for the
    /// remaining prime/awkward factors. Panics if `len == 0`.
    pub fn new(len: usize) -> Self;
    pub fn len(&self) -> usize;

    /// In-place forward transform. Panics unless `buf.len() == self.len()`.
    pub fn forward(&mut self, buf: &mut [Complex<T>]);
    /// In-place inverse transform (normalized by 1/len).
    pub fn inverse(&mut self, buf: &mut [Complex<T>]);
}
```
`&mut self` because forward/inverse reuse the internal `scratch` buffer (Bluestein / mixed
-radix need workspace). This is the RustFFT-planner shape minus autotuning.

### Real-input transforms (`real.rs`)

```rust
// Fixed tier: real input of length N (power-of-two) -> N/2+1 complex bins.
// Output length is N/2+1; caller passes a matching output slice to dodge generic_const_exprs.
pub fn rfft<T: FloatScalar, const N: usize>(input: &[T; N], output: &mut [Complex<T>]);   // output.len() == N/2 + 1
pub fn irfft<T: FloatScalar, const N: usize>(input: &[Complex<T>], output: &mut [T; N]);  // input.len()  == N/2 + 1

// Dyn tier (alloc):
#[cfg(feature = "alloc")]
pub struct DynRealFft<T: FloatScalar> { /* ... */ }
#[cfg(feature = "alloc")]
impl<T: FloatScalar> DynRealFft<T> {
    pub fn new(len: usize) -> Self;                              // len = real signal length
    pub fn forward(&mut self, input: &[T], out: &mut [Complex<T>]);  // out.len() == len/2 + 1
    pub fn inverse(&mut self, input: &[Complex<T>], out: &mut [T]);  // out.len() == len
}
```
**Decision needed:** for the fixed-tier real transform, pass the output slice (chosen above,
avoids `generic_const_exprs`) vs. return a fixed array (needs `[_; N/2+1]` → unstable).
Recommendation: caller-supplied output slice with an `assert_eq!` on length.

### Convolution / shift helpers

```rust
#[cfg(feature = "alloc")]
pub fn fft_convolve<T: FloatScalar>(a: &[T], b: &[T]) -> Vec<T>;      // linear convolution
#[cfg(feature = "alloc")]
pub fn fft_correlate<T: FloatScalar>(a: &[T], b: &[T]) -> Vec<T>;

// No-alloc, in-place, any length:
pub fn fftshift<T: Copy>(buf: &mut [T]);
pub fn ifftshift<T: Copy>(buf: &mut [T]);
```

### Private SIMD dispatch (internal, `src/simd/`)

Not public API; listed so the signatures are agreed before kernels are written. Structure-of
-arrays (deinterleaved re/im) to match SIMD lanes:
```rust
// TypeId compile-time dispatch, scalar fallback == reference impl.
pub(crate) fn fft_bfly2_dispatch<T: FloatScalar>(re: &mut [T], im: &mut [T], tw: &[Complex<T>], stride: usize);
pub(crate) fn fft_bfly4_dispatch<T: FloatScalar>(re: &mut [T], im: &mut [T], tw: &[Complex<T>], stride: usize);
```
**Decision needed:** carry the working buffer as deinterleaved `re: &mut [T], im: &mut [T]`
(SIMD-friendly, one deinterleave/reinterleave at the boundary) vs. keep `&mut [Complex<T>]`
(interleaved, no repack but worse SIMD). Recommendation: deinterleave into re/im scratch for
the SIMD path; scalar path can operate on either. This is the single biggest perf lever.

## 6. Proposed file layout

```
src/fft/                 # (requires `fft` feature)
├── mod.rs               # FftError, re-exports, module rustdoc (incl. "not FFTW" note)
├── twiddle.rs           # TwiddleTable<T, N>, bit-reversal tables
├── radix.rs             # radix-2 / radix-4 butterfly passes (size-generic core)
├── fixed.rs             # fft/ifft/fft_inplace (const-generic entry points, no-alloc)
├── real.rs              # rfft / irfft (real-input packing + untangle)
├── dynfft.rs            # DynFft planner: mixed-radix + Bluestein (requires `alloc`)
├── bluestein.rs         # chirp-z transform for arbitrary/prime N (requires `alloc`)
├── convolve.rs          # fft_convolve / fft_correlate (requires `alloc`)
├── shift.rs             # fftshift / ifftshift
└── tests.rs             # comprehensive tests (vs known DFT, round-trip, Parseval)

src/simd/
├── fft_neon.rs          # aarch64 complex butterfly kernels
├── fft_sse2.rs          # x86_64 complex butterfly kernels
└── ... (avx / avx512 opt-in, mirroring existing files)
```

## 7. Cargo feature

```toml
fft = ["dep:num-complex"]          # fixed tier is no-alloc; complex support required
# DynFft / bluestein / convolve are additionally gated on `alloc` inside the module.
```

- `complex` is effectively a prerequisite — FFT output is complex. Reuse the existing
  `num-complex` optional dep (already wired for `std`/`libm`).
- Add to the `all` feature list.

## 8. Testing strategy

- **Correctness:** compare against a naive O(N²) DFT for small N (exact reference).
- **Round-trip:** `ifft(fft(x)) ≈ x` to tight tolerance.
- **Parseval / Plancherel:** energy conservation.
- **Real transform:** `rfft` vs full complex `fft` of real-valued input.
- **Arbitrary sizes:** `DynFft` for primes (Bluestein path) and mixed composites.
- **Proptest** (repo already uses proptest): round-trip + linearity + shift invariants,
  matching the pattern in the current `test/proptest-invariants` branch.

## 9. Build order (internal increments; all land together per locked scope)

Full scope was chosen, so nothing ships to users incrementally — but build and verify in
this order so correctness is proven before perf and before the harder paths:

1. `twiddle.rs` + `radix.rs` + `fixed.rs` — **scalar** radix-2/4 power-of-two, const-generic
   up to 4096. Get it correct vs naive DFT + round-trip first (SIMD bugs are easier to
   isolate against a trusted scalar baseline).
2. `simd/fft_neon.rs` + `simd/fft_sse2.rs` — deinterleaved re/im butterfly kernels behind
   `TypeId` dispatch; scalar fallback stays as the reference. AVX/AVX-512 opt-in mirror.
   Assert SIMD path == scalar path in tests.
3. `real.rs` — `rfft`/`irfft` (real-input packing + untangle), built on the complex core.
4. `dynfft.rs` — mixed-radix (2,3,5) planner + twiddle/scratch cache (alloc tier).
5. `bluestein.rs` — chirp-z for prime / awkward N; wire into `DynFft` as the fallback plan.
6. `convolve.rs`, `shift.rs`, docs page + `Module Plan` checkbox flip.

Each step gated behind the correctness tests from §8 before moving on.

## 10. Pre-push doc surfaces to update when landing (per CLAUDE.md checklist)

- `src/lib.rs` rustdoc (module bullet, features table, re-exports)
- `README.md` (features list, table, `<details>` block, Module plan checkbox)
- `CHANGELOG.md` new entry + `Cargo.toml` version bump
- `docs/fft.md` + `mkdocs.yml` nav
- `CLAUDE.md` Module Plan / Cargo Features / File Layout
