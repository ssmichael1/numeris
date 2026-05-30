# Performance

numeris achieves competitive performance via SIMD intrinsics, register-blocked micro-kernels, and direct formulas for small sizes.

## SIMD Architecture

SIMD is **always-on** for `f32` and `f64` — no feature flag, no runtime detection. Dispatch selects the widest available ISA at compile time via `#[cfg(target_feature)]`. Integer and complex types fall back to scalar loops via `TypeId` dispatch, with zero runtime overhead (dead-code eliminated at monomorphization).

| Architecture | ISA | f64 tile (MR×NR) | f32 tile (MR×NR) |
|---|---|---|---|
| aarch64 | NEON (128-bit) | 8×4 | 8×4 |
| x86_64 | SSE2 (128-bit) | 4×4 | 8×4 |
| x86_64 | AVX (256-bit) | 8×4 | 16×4 |
| x86_64 | AVX-512 (512-bit) | 16×4 | 32×4 |
| other | scalar fallback | 4×4 | 4×4 |

AVX and AVX-512 require compile-time opt-in:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
# or explicitly:
RUSTFLAGS="-C target-feature=+avx2,+avx512f" cargo build --release
```

SSE2 (x86_64) and NEON (aarch64) are always-on baselines.

## Matrix Multiply Micro-Kernels

The matmul micro-kernels are inspired by [nano-gemm](https://github.com/sarah-quinones/nano-gemm) and [faer](https://github.com/sarah-quinones/faer-rs) (Sarah Quinones). Each kernel:

1. Tiles the output matrix into MR×NR panels
2. Loads A and B into SIMD registers
3. Accumulates the full k-sum into MR×NR SIMD accumulators (never reading/writing C in the inner loop)
4. Writes the tile to C exactly once

This reduces memory traffic by O(n) compared to a naive C-read-per-k implementation. The k-loop is cache-blocked at KC=256 elements for L1 cache locality.

```
For each (block_i, block_j) tile of C:
  accum[0..MR*NR] = 0
  for k = 0..K:
    load A[block_i..block_i+MR, k] into MR SIMD vectors
    load B[k, block_j..block_j+NR] into NR scalars
    for row in 0..MR: accum[row*NR..] += A_row * B_col
  store accum → C[block_i..block_i+MR, block_j..block_j+NR]
```

## Optimizations Applied

- **Direct formulas** for N ≤ 4: `inverse()` and `det()` use adjugate-based closed-form expressions, bypassing LU decomposition entirely.
- **Unrolled LU/Cholesky** for N ≤ 6: direct `data[col][row]` array indexing eliminates trait dispatch in inner loops.
- **AXPY SIMD**: LU elimination, QR Householder, SVD bidiagonalization, and Hessenberg reduction all use `axpy_neg_dispatch` on contiguous column slices.
- **4-accumulator dot product**: reduces dependency chains in the dot product inner loop.

## Benchmark Results

Platform: Apple Silicon M-series (aarch64 NEON), Rust stable.

Compared against nalgebra 0.34 and faer 0.24. All benchmarks run with `cargo bench`.

| Benchmark | numeris | nalgebra | faer | Winner |
|---|---|---|---|---|
| matmul 4×4 | **4.9 ns** | 4.9 ns | 58 ns | ~tie |
| matmul 6×6 | **13.4 ns** | 20.0 ns | 87 ns | numeris |
| matmul 50×50 (dyn) | **5.76 µs** | 6.63 µs | 6.3 µs | numeris |
| matmul 200×200 (dyn) | 369 µs | 361 µs | **193 µs** | faer |
| dot 100 (dyn) | **11.6 ns** | 14.5 ns | — | numeris |
| LU 4×4 | 33.2 ns | **28.2 ns** | 203 ns | nalgebra |
| LU 6×6 | 84.7 ns | **82.1 ns** | 292 ns | nalgebra |
| LU 50×50 (dyn) | 8.4 µs | **7.5 µs** | 7.7 µs | nalgebra |
| Cholesky 4×4 | 25.2 ns | **11.8 ns** | 139 ns | nalgebra |
| Cholesky 6×6 | 70.7 ns | **39.6 ns** | 186 ns | nalgebra |
| QR 4×4 | **46.4 ns** | 90.6 ns | 303 ns | numeris |
| QR 6×6 | **85.5 ns** | 207.9 ns | 445 ns | numeris |
| SVD 4×4 | **299 ns** | 461 ns | 1278 ns | numeris |
| SVD 6×6 | 1171 ns | **925 ns** | 1858 ns | nalgebra |
| Inverse 4×4 | 27.6 ns | **23.3 ns** | — | nalgebra |
| Inverse 6×6 | 163 ns | **127 ns** | — | nalgebra |
| Eigen sym 4×4 | **165 ns** | 201 ns | 578 ns | numeris |
| Eigen sym 6×6 | **287 ns** | 528 ns | 1088 ns | numeris |

### Summary

- **numeris wins**: matmul 6×6 (1.5×), QR (2×), SVD 4×4 (1.5×), symmetric eigendecomposition (1.2–1.8×), dot product, matmul 50×50
- **nalgebra wins**: Cholesky, LU, inverse, SVD 6×6 — Cholesky gap is a measurement artifact (see below)
- **faer wins**: large dynamic matmul (200×200) — A+B packing, cache-aware blocking
- faer has high overhead at small sizes due to dynamic dispatch / runtime machinery
- matmul 4×4: dead heat with nalgebra (4.9 ns each)

!!! note "Cholesky benchmark artifact"
    The ~2× Cholesky gap is a **measurement artifact**, not a real performance difference. Micro-benchmarking shows raw computation is within 4% of nalgebra. The gap comes from Criterion's `black_box` reading `Result<CholeskyDecomposition, LinalgError>` byte-by-byte (48 `ldrb` instructions) vs nalgebra's `Option<Cholesky>` using word-sized `ldr` (17 instructions).

## Remaining Opportunities

- **Large matmul**: A-panel packing + larger tile sizes could close the remaining ~2× gap with faer
- **SVD 6×6**: 27% behind nalgebra — likely dominated by Givens rotations in bidiagonal QR
- **LU**: small margin behind nalgebra — possibly similar `Result` vs `Option` artifact

## No-std Performance

On embedded targets with no hardware FPU, float operations fall back to the `libm` software implementation. Performance in this mode is entirely determined by the target's ALU throughput — SIMD code paths are not compiled for targets without SIMD (the `TypeId` dispatch compiles down to scalar loops).

## Parallelism (`rayon`)

SIMD parallelizes *within* a core (vector lanes); the optional **`rayon`** feature parallelizes *across* cores (threads). The two are orthogonal and compose — each worker thread still runs the SIMD kernels. Parallelism is **opt-in** because [rayon](https://docs.rs/rayon) requires `std` and a thread pool, which the no-std / embedded baseline cannot assume:

```toml
numeris = { version = "0.5", features = ["rayon"] }   # implies std
```

The feature is **purely additive**: builds without it are byte-for-byte unchanged, and the signatures are unconstrained (the `Send + Sync` element requirement is carried by a marker bound that is empty unless `rayon` is enabled, and is satisfied automatically by `f32` / `f64`).

### What is parallelized

Only heap-backed, runtime-sized paths with **independent, disjoint output columns** — never small fixed-size `Matrix` ops (thread dispatch would dwarf the work) and never order-sensitive reductions (which would sacrifice reproducibility):

| Area | Routines | Axis |
|---|---|---|
| `optim` | `finite_difference_jacobian_dyn`, `finite_difference_gradient_dyn` | columns = independent function evaluations |
| `imageproc` convolution | `gaussian_blur`, `box_blur`, `unsharp_mask`, `laplacian_of_gaussian`, `canny`, Harris / Shi-Tomasi corners, DoG, Gaussian pyramid | output columns |
| `imageproc` rank/median | `rank_filter`, `percentile_filter`, `median_filter` | output columns (quickselect per pixel) |
| `imageproc` geometric/stats | `resize_bilinear`, `local_mean` / `local_variance` / `local_stddev`, `adaptive_threshold` | output columns |
| `imageproc` morphology | `dilate` / `erode`, `opening` / `closing`, `max`/`min_filter`, gradient, top-hat, black-hat | output columns (horizontal pass via transpose) |

### Determinism and gating

- **Deterministic results.** Each worker writes a disjoint slice of the output, so the result is identical regardless of thread count. (Parallel *reductions*, where summation order would change the floating-point answer, are intentionally not used.)
- **Work-aware gating.** Each routine decides whether to fan out based on *total work* (e.g. `nrows · ncols · kernel_size`), not raw column count — so a small image, or a cheap operation on a medium image, stays sequential and never pays thread-dispatch overhead. The crossover that matters is the cost of the work, not its shape.

### Measured speedups

Apple M3 — 8 cores (4 performance + 4 efficiency), aarch64, `f64`. Rayon uses its default thread pool (all 8 logical cores), so the heterogeneous P+E cores cap practical scaling below 8×. Parallel vs. the same build with `rayon` off:

| Workload | Size | Speedup |
|---|---|---|
| Separable Gaussian blur | 512² | ~2.6× |
| Median filter (5×5) | 256² | ~3.4× |
| Median filter (3×3) | 256² | ~3.7× |
| Morphological dilation (r=3) | 1024² | ~4.1× |
| Finite-difference Jacobian | n=256, expensive `f` | ~4× |

Wins scale with total work; below each routine's gate the parallel build matches the sequential one (no regression). See `bench/` for the harnesses — each is feature-toggled (`--no-default-features` flips `rayon` off) so Criterion can diff the two baselines.
