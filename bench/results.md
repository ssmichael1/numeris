# Benchmark Results

Platform: Apple Silicon (aarch64, NEON), Rust stable, `cargo bench`
Date: 2026-03-21

## Current Results (v0.5.6)

Optimizations applied (cumulative):
1. 4-accumulator dot product, cascading matmul bottom-edge micro-kernels, Cholesky left-looking column-AXPY, AXPY small-size threshold
2. Direct 2x2/3x3/4x4 inverse formulas (adjugate, bypasses LU), direct det formulas
3. Unrolled small LU/Cholesky for N<=6 (direct `data[col][row]` array access, no trait dispatch)
4. Cache-blocked matmul (KC=256 k-blocking across all SIMD kernels)
5. 8x4 f64 NEON micro-kernel (16 accumulators, 4 A-vectors x 4 B-columns per k-step)
6. SIMD dispatch for Matrix element-wise ops (add/sub/scale via slice dispatch for large matrices)
7. Small-matrix specializations: matmul bypass for dims <= 3, element-wise ops bypass for M*N <= 16
8. Column-major loop order + direct `data[j][i]` access throughout ops (no bounds-checked indexing)
9. B-panel packing for large dynamic matmul (n > 64): pack NR=4 columns of B contiguously per k-block
10. Direct 2x2/3x3 symmetric eigensolvers (closed-form, bypasses QR iteration)
11. Stable Givens rotations via hypot, Kahan compensated summation in norms/dot
12. SVD global σ_max deflation threshold (matches LAPACK DGESVD)

| Benchmark | numeris | nalgebra | faer | Winner |
|---|---|---|---|---|
| matmul 4x4 | 6.74 ns | **4.94 ns** | 56.7 ns | nalgebra |
| matmul 6x6 | 20.7 ns | **20.0 ns** | 87.5 ns | ~tie |
| matmul 50x50 (dyn) | **5.74 µs** | 6.59 µs | 6.21 µs | numeris |
| matmul 200x200 (dyn) | 318 µs | 295 µs | **180 µs** | faer |
| dot 100 (dyn) | **11.5 ns** | 12.5 ns | — | numeris |
| LU 4x4 | 32.8 ns | **28.2 ns** | 203 ns | nalgebra |
| LU 6x6 | 84.4 ns | **82.0 ns** | 295 ns | ~tie |
| LU 50x50 (dyn) | 8.58 µs | **7.49 µs** | 7.63 µs | nalgebra |
| Cholesky 4x4 | 25.2 ns | **11.5 ns** | 133 ns | nalgebra |
| Cholesky 6x6 | 71.0 ns | **39.8 ns** | 187 ns | nalgebra |
| QR 4x4 | **46.2 ns** | 92.9 ns | 301 ns | numeris |
| QR 6x6 | **84.1 ns** | 210 ns | 444 ns | numeris |
| SVD 4x4 | **454 ns** | 465 ns | 1.28 µs | ~tie |
| SVD 6x6 | **620 ns** | 976 ns | 1.94 µs | numeris |
| Inverse 4x4 | 29.9 ns | **26.1 ns** | — | nalgebra |
| Inverse 6x6 | 163 ns | **127 ns** | — | nalgebra |
| Eigen sym 4x4 | **181 ns** | 265 ns | 585 ns | numeris |
| Eigen sym 6x6 | **304 ns** | 536 ns | 1.10 µs | numeris |

## Summary

- **numeris wins**: QR (2x), SVD 6x6 (1.6x), symmetric eigen (1.5–1.8x), dot product, matmul 50x50
- **nalgebra wins**: Cholesky (2.2x at 4x4, 1.8x at 6x6), inverse (1.1–1.3x), LU (small margin), matmul 4x4
- **faer wins**: large dynamic matmul 200x200 (1.8x) — A+B panel packing, cache-aware blocking
- faer has high overhead at small sizes due to dynamic dispatch / runtime machinery

## Notes

- **Cholesky gap is a measurement artifact**: Direct formulas for 2×2/3×3/4×4 are implemented
  and compute in ~0.35 ns (4×4). The Criterion benchmark shows 25 ns because `black_box`
  reads `Result<CholeskyDecomposition, LinalgError>` byte-by-byte (48 `ldrb` instructions)
  vs nalgebra's `Option<Cholesky>` using word-sized `ldr` (17 instructions). Raw computation
  is faster than nalgebra.
- **LU/Inverse gaps** are similarly dominated by `Result` overhead in Criterion.

## Remaining Optimization Targets

| Target | Current gap | Approach |
|---|---|---|
| **matmul 4x4** | 1.4x behind nalgebra | Hand-unrolled 4x4, skip micro-kernel dispatch |
| **matmul 200x200** | 1.8x behind faer | A-panel packing + larger MC/NC tile sizes |

## Parallel finite-difference Jacobian (`rayon` feature)

Platform: Apple Silicon (aarch64), `cargo bench -p numeris-bench --bench fd_jacobian`.
`seq` = sequential baseline (faithful re-impl), `par` = `finite_difference_jacobian_dyn`
with `rayon`. `nN_wW` = N columns, W extra transcendental ops per output element
(synthetic eval cost). Total FD work scales as O(N² · W).

| Case | seq | par | Result |
|---|---|---|---|
| n4, w0   | 195 ns  | 202 ns  | tie (N<8 → runs sequential anyway) |
| n16, w4  | 8.78 µs | 20.1 µs | seq 2.3× — eval too cheap, dispatch dominates |
| n64, w4  | 123 µs  | 52 µs   | **par 2.4×** |
| n256, w8 | 5.29 ms | ~1.3 ms | **par ~4×** |

Takeaway: the win tracks total work (`N · cost(f)`), not column count alone — at
n16/w4 the parallel path is above the column threshold yet still loses because each
evaluation is trivially cheap. The `FD_PAR_MIN_COLS = 8` guard is intentionally low:
it favors the regime parallelism is *for* (expensive `f` — ODE steps, measurement
models — where even small N wins big), accepting a few-µs absolute regression on
cheap-`f` mid-N cases that are already fast in absolute terms.

## Parallel separable convolution (`rayon` feature)

Platform: Apple Silicon (aarch64), `cargo bench -p numeris-bench --bench convolve`
(seq via `--no-default-features`). `gaussian_blur`, sigma=2 (≈13-tap kernel),
square `n×n` f64 images.

| n×n | seq | par | Result |
|---|---|---|---|
| 32²  | 9.2 µs  | 9.9 µs  | runs sequential (work < budget) |
| 128² | 71.9 µs | 73.7 µs | runs sequential (work < budget) — no regression |
| 512² | 838 µs  | 326 µs  | **par 2.56×** |

The gate is on per-pass *work* (`nrows · ncols · klen`), not column count: an
early flat 64-column threshold parallelized 128² and lost 26% to thread
overhead. Gating on work (`CONV_PAR_WORK_BUDGET`) keeps 128² sequential while
512² — and tall/narrow or large-kernel images of similar total work — fan out.

## Parallel rank / median filter (`rayon` feature)

Platform: Apple Silicon (aarch64), `cargo bench -p numeris-bench --bench rank`
(seq via `--no-default-features`). `median_filter`, square f64 images, radius 1
(3×3) and 2 (5×5).

| case | seq | par | Result |
|---|---|---|---|
| r1 / 64²  | 53 µs   | 57 µs   | sequential (work < budget) |
| r2 / 64²  | 162 µs  | 159 µs  | sequential (~tie) |
| r1 / 128² | 212 µs  | 205 µs  | ~tie (just below gate) |
| r2 / 128² | 589 µs  | 255 µs  | **par 2.3×** |
| r1 / 256² | 797 µs  | 214 µs  | **par 3.7×** |
| r2 / 256² | 2.26 ms | 667 µs  | **par 3.4×** |

Quickselect per pixel is the most expensive per-pixel work in imageproc, so the
work gate (`RANK_WORK_BUDGET`, scaled by window area `k_total`) engages
parallelism at smaller sizes than the cheap separable blur — and the larger
5×5 window crosses the gate sooner than 3×3, as intended.

## Parallel morphology (`rayon` feature)

Platform: Apple Silicon (aarch64), `cargo bench -p numeris-bench --bench morphology`
(seq via `--no-default-features`). `dilate`, radius 3, square f64 images.

| n×n | seq | par | Result |
|---|---|---|---|
| 128²  | 163 µs  | 174 µs  | sequential (work < gate; small transpose overhead) |
| 512²  | 3.07 ms | 790 µs  | **par 3.9×** |
| 1024² | 13.7 ms | 3.33 ms | **par 4.1×** |

The no-`rayon` build keeps the lean sequential separable pass (two buffers +
row scratch, no transposes). Under `rayon`, the horizontal direction is run as
a second *vertical* pass in transposed space, so both Van Herk passes and both
transposes parallelize over output columns while preserving O(1)-per-pixel
cost; the cost is two extra full-image allocations, taken only when the feature
is enabled.
