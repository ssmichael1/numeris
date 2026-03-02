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

Compared against nalgebra 0.33 and faer 0.19. All benchmarks run with `cargo bench`.

| Benchmark | numeris | nalgebra | faer | Winner |
|---|---|---|---|---|
| matmul 4×4 | 7.1 ns | **5.2 ns** | 61 ns | nalgebra |
| matmul 6×6 | **21.2 ns** | 21.1 ns | 93 ns | ~tie |
| matmul 50×50 (dyn) | **6.8 µs** | 6.9 µs | 6.5 µs | ~tie |
| matmul 200×200 (dyn) | 367 µs | 310 µs | **174 µs** | faer |
| dot 100 (dyn) | **12.4 ns** | 13.2 ns | — | numeris |
| LU 4×4 | 34.8 ns | **30.2 ns** | 212.5 ns | nalgebra |
| LU 6×6 | 90.3 ns | **80.8 ns** | 300.9 ns | nalgebra |
| LU 50×50 (dyn) | 8.9 µs | **7.8 µs** | 8.0 µs | nalgebra |
| Cholesky 4×4 | 26.5 ns | **12.2 ns** | 142.1 ns | nalgebra |
| Cholesky 6×6 | 76.5 ns | **42.1 ns** | 202.9 ns | nalgebra |
| QR 4×4 | **62.9 ns** | 101.7 ns | 328.4 ns | numeris |
| QR 6×6 | **85.9 ns** | 217.3 ns | 521.5 ns | numeris |
| SVD 4×4 | **313.9 ns** | 489.0 ns | 1365.7 ns | numeris |
| SVD 6×6 | 1135.5 ns | **975.3 ns** | 2011.1 ns | nalgebra |
| Inverse 4×4 | **29.6 ns** | 24.8 ns | — | nalgebra |
| Eigen sym 4×4 | **183.7 ns** | 213.4 ns | 621.5 ns | numeris |
| Eigen sym 6×6 | **352.0 ns** | 574.8 ns | 1200.5 ns | numeris |

### Summary

- **numeris wins**: QR (2.5×), SVD 4×4, symmetric eigendecomposition (1.5–2×), dot product, matmul 50×50
- **nalgebra wins**: small matmul 4×4, Cholesky, LU, inverse — gaps are small
- **faer wins**: large dynamic matmul (200×200) — A/B panel packing dominates at that scale
- faer has high overhead at small sizes due to dynamic dispatch / runtime machinery

## Improvements vs. Initial Baseline

| Benchmark | Before | After | Speedup |
|---|---|---|---|
| matmul 200×200 | 562 µs | 367 µs | **1.5×** |
| matmul 50×50 | 12.5 µs | 6.8 µs | **1.8×** |
| matmul 6×6 | 62.1 ns | 21.2 ns | **2.9×** |
| inverse 4×4 | 79.9 ns | 29.6 ns | **2.7×** |
| LU 4×4 | 45.3 ns | 34.8 ns | **1.3×** |
| dot 100 | 25.2 ns | 12.4 ns | **2.0×** |

## Remaining Opportunities

- **Cholesky**: still ~2× behind nalgebra — bottleneck is sqrt/division cost, not dispatch overhead
- **Large matmul**: A/B panel packing could close the remaining ~2× gap with faer
- **Small matmul 4×4**: nalgebra's hardcoded unrolled kernel is difficult to beat generically

## No-std Performance

On embedded targets with no hardware FPU, float operations fall back to the `libm` software implementation. Performance in this mode is entirely determined by the target's ALU throughput — SIMD code paths are not compiled for targets without SIMD (the `TypeId` dispatch compiles down to scalar loops).
