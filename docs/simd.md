# SIMD & Runtime Dispatch

numeris accelerates `f32` / `f64` hot paths — dot products, matrix multiply, AXPY, element-wise
ops, strided 1-D convolution, and the FFT butterflies — with hand-written `core::arch` intrinsics.
This page explains which instruction sets are used, how the crate decides which one to run, and
what the optional **`runtime-dispatch`** feature changes.

The short version:

| | Default build | With `runtime-dispatch` |
|---|---|---|
| **How the tier is chosen** | At compile time, from `-C target-feature` / `-C target-cpu` | At compile time as a *floor*, raised once at runtime by a CPU probe |
| **Best for** | A binary built on the machine that runs it | One binary distributed to many machines (CLI, Python wheel) |
| **Cost** | None | One relaxed atomic load + compare per kernel call |
| **Platforms affected** | all | x86_64 only (aarch64 / no-std unchanged) |
| **Requires** | — | `std` |

## Tiers

SIMD is always on — there is no feature flag for it. Integer and complex element types fall back
to scalar loops at zero cost (the `TypeId` dispatch is a compile-time constant and dead branches
are eliminated at monomorphization).

| Architecture | Tier | Vector width | f64 matmul tile | f32 matmul tile | Fused multiply-add |
|---|---|---|---|---|---|
| aarch64 | NEON | 128-bit | 8×4 | 8×4 | yes |
| x86_64 | SSE2 | 128-bit | 4×4 | 8×4 | no (SSE2 has no FMA) |
| x86_64 | AVX + FMA | 256-bit | 8×4 | 16×4 | yes |
| x86_64 | AVX-512F | 512-bit | 16×4 | 32×4 | yes |
| other | scalar | — | 4×4 | 4×4 | — |

NEON and SSE2 are the architectural baselines of their targets and are always available. The AVX
and AVX-512 tiers must be *selected*, either at compile time or at runtime.

!!! note "Why the AVX tier requires FMA"
    Every multiply-add in the AVX and AVX-512 kernels is fused (`vfmadd`): one instruction and one
    rounding instead of two. Rust never contracts a separate multiply and add into an FMA on its
    own — even under `target-cpu=native` — because it changes rounding, so fusion is explicit in
    the kernels, and the tier that contains it must require the `fma` feature. Every AVX2 CPU
    (Intel Haswell / AMD Zen 1 and later) has FMA. The AVX-only Sandy Bridge and Ivy Bridge parts
    (2011–2013) take the SSE2 tier instead. AVX-512F implies FMA.

## Compile-time selection (default)

Without `runtime-dispatch`, the tier is fixed when the crate is compiled. Whatever `-C
target-feature` (or `-C target-cpu`) enables for the whole build is what the dispatch uses:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release                   # widest the build host has
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release           # AVX tier (x86-64-v3)
RUSTFLAGS="-C target-feature=+avx2,+fma,+avx512f" cargo build --release  # AVX-512 tier
```

With no flags, an x86_64 build uses SSE2 everywhere. This is the right default for code built on
the machine that runs it, and it is the only mode available on no-std targets.

!!! warning "Don't commit `target-cpu=native`"
    A blanket `target-cpu=native` in a repo's `.cargo/config.toml` is non-portable, and LLVM's
    `native` detection guesses features from the CPU model rather than reading them — some
    virtualized CI runners then `SIGILL` on AVX-512 instructions the host reports but cannot run.
    Opt in per shell, or use a portable level like `x86-64-v3`.

## Runtime dispatch

Compile-time selection forces a distributed binary into a bad choice: SSE2 everywhere, or a
`SIGILL` on any machine narrower than the build flags. The `runtime-dispatch` feature removes it:

```toml
[dependencies]
numeris = { version = "0.7", features = ["runtime-dispatch"] }   # implies std
```

What changes:

1. **All x86_64 tiers are compiled into the binary.** Each AVX / AVX-512 kernel carries its own
   `#[target_feature(enable = ...)]`, so the compiler emits the wide instructions *inside* those
   functions regardless of the crate-wide target. Tiers the build never selects are dead code and
   are dropped by the linker.
2. **A one-time probe picks the tier.** On the first kernel call, `std::is_x86_feature_detected!`
   checks `avx512f`, then `avx` + `fma`. The macro includes the OS state-save check (`xgetbv`), so
   a kernel that has not enabled ZMM state correctly reports the lower tier. The result is cached
   in a single byte; every later call is one relaxed load and a compare.
3. **The compile-time features remain a floor.** The probe can only raise the tier, never lower
   it. Build with `+avx512f` and the selector is a compile-time constant — the dispatch `match`
   folds away and the code is identical to a build without the feature.

The feature is purely additive: no signatures change, and nothing happens on aarch64 (NEON is the
baseline; there are no wider kernels) or in no-std builds (the feature needs `std`).

!!! warning "Results can differ between machines"
    Without runtime dispatch, a given build produces bit-identical results wherever it runs. With
    it, the same binary may take a different tier on different CPUs, and the tiers do not round
    identically: SSE2 multiplies then adds, while AVX and AVX-512 fuse the two into one rounding,
    and each tier reduces a dot product in a different order. The differences are at the level of
    floating-point round-off (the crate's own tests compare tiers against a scalar reference with
    tolerances of about `1e-12` for `f64` and `1e-4` for `f32`), but they are real. If you compare
    outputs across machines, or store expected values from one machine and check them on another,
    compare with a tolerance rather than for equality — or build without the feature and pin the
    tier with `-C target-feature` so every machine runs the same kernels.

### Cost

The per-call overhead is one relaxed atomic load of a cached byte and a predicted branch. This
has not been benchmarked. It is expected to be lost in the noise on anything larger than the
smallest fixed-size operations, and on those (a 4×4 or 6×6 product, 80–200 ns) to be small
compared with the ±10 % swings that *code alignment* alone produces in the crate's fixed-size
benchmarks — which is also why measuring it cleanly is hard. If you depend on those small
operations in a hot loop, measure on your own workload; see [Performance](performance.md) for
the alignment caveat.

### Which tier am I getting?

There is no public accessor by design (the tier is an implementation detail). To check a machine,
run the crate's own tier tests, which print which tiers they skipped:

```bash
cargo test --features runtime-dispatch --lib simd::tests::x86_tiers -- --nocapture
```

Each tier the CPU supports is exercised directly against the scalar reference; the probe is also
compared against `std`'s detection.

## Tier notes

**SSE2.** The x86_64 baseline. No FMA, 128-bit registers. What a default x86_64 build uses, and
what the AVX-only Sandy / Ivy Bridge generation falls back to under runtime dispatch.

**AVX + FMA.** Every CPU from Intel Haswell (2013) and AMD Zen 1 (2017) onward; the `x86-64-v3`
level. The main beneficiary of runtime dispatch — most x86_64 machines in service land here.

**AVX-512F.** The probe checks only `avx512f`; the kernels use no DQ / BW / VL extensions.
Available on Intel server parts since Skylake-SP, some Intel laptop parts (Ice Lake through Raptor
Lake), and AMD Zen 4 / Zen 5. Most recent Intel *desktop* chips have it fused off, and Rosetta on
Apple silicon does not expose it. Two things to know:

- *Frequency licensing.* Skylake-X and Cascade Lake lower their clock under sustained 512-bit use,
  so short AVX-512 bursts inside otherwise scalar code can lose to AVX2 on those parts. Ice Lake
  and later, and Zen 4 / Zen 5, have largely removed the penalty (Zen 4 executes 512-bit ops on
  double-pumped 256-bit datapaths — a modest gain over AVX2, never a loss).
- *Where it helps.* Runtime-sized work: `DynMatrix` products, `imageproc` convolution, the FFT
  butterflies. For 4×4 and 6×6 fixed-size matrices the 16×4 tile does not fit, so the tier falls
  straight to its narrower micro-kernels and gains little over AVX.

If a machine's AVX-512 turns out to be a net loss for a workload, build without
`runtime-dispatch` and set the tier explicitly with `-C target-feature`.

## How it is built (design notes)

The mechanism rests on a few properties of `#[target_feature]` that are easy to get wrong:

- **The attribute goes on the kernels, not the dispatcher.** An intrinsic called from a function
  that lacks its feature cannot be inlined, so the whole kernel body must sit inside an attributed
  function. The kernels were already self-contained `fn`s, so the change was one attribute per
  kernel (the shared kernel macros take a `@feature "avx,fma"` argument and emit it).
- **Safe kernels need Rust 1.86; AVX-512 intrinsics need 1.89.** Before 1.86 a `#[target_feature]`
  function had to be `unsafe fn`, which would have forced a `# Safety` section onto every kernel.
  Hence the crate's MSRV of **1.89** (see [Design](design.md#avoiding-unstable-features)).
- **Calling an attributed function is `unsafe` unless the *caller's own attribute* covers the
  feature.** A crate-wide `-C target-feature` flag does not count. So the dispatch macro's AVX /
  AVX-512 arms are `unsafe` blocks in every configuration, justified by the selector — the crate's
  single dispatch-site `unsafe`, alongside the audited kernel `unsafe` (see the
  [`unsafe` discipline](design.md#simd-dispatch)).
- **Inside an attributed function, register-only intrinsics are safe.** Broadcasts and horizontal
  sums no longer need `unsafe` in the AVX tiers, while the unattributed SSE2 / NEON tiers still
  need it; the shared macros allow `unused_unsafe` alongside the attribute for that reason.
- **`#[inline(always)]` and `#[target_feature]` are mutually exclusive.** The matmul micro-kernel
  helpers stay unattributed `#[inline(always)] unsafe fn`s and inline into the attributed caller,
  picking up its features.

## Testing

- `simd::tests` compares every `*_dispatch` entry point against the scalar reference at boundary
  lengths — whichever tier the build selects.
- `simd::tests::x86_tiers` (x86_64, `runtime-dispatch`) calls each tier's kernels *directly*,
  guarded by the probe, so a machine with AVX-512 checks all three tiers, and asserts that the cached
  probe agrees with `std::is_x86_feature_detected!`.
- CI runs the x86_64 matrix under `x86-64-v3` (AVX tier as the compile-time floor), a dedicated
  `runtime-dispatch` job on the baseline target (SSE2 floor, AVX reached only through the probe),
  NEON on the aarch64 runner, and an MSRV job on 1.89. AVX-512 executes only when a runner happens
  to have it; the tier test's skip lines in the log say whether it did.

## See also

- [Performance](performance.md) — benchmark numbers, matmul micro-kernel design, `rayon` parallelism
- [Design › SIMD Dispatch](design.md#simd-dispatch) — the `TypeId` dispatch and the `unsafe` argument
- [No-std / Embedded](no-std.md) — targets without SIMD fall back to scalar loops
