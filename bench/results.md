# Benchmark Results

Platform: Apple Silicon (aarch64, NEON), Rust stable, `cargo bench`
Date: 2026-03-10

## Current Results

Optimizations applied (cumulative):
1. 4-accumulator dot product, cascading matmul bottom-edge micro-kernels, Cholesky left-looking column-AXPY, AXPY small-size threshold
2. Direct 2x2/3x3/4x4 inverse formulas (adjugate, bypasses LU), direct det formulas
3. Unrolled small LU/Cholesky for N<=6 (direct `data[col][row]` array access, no trait dispatch)
4. Cache-blocked matmul (KC=256 k-blocking across all SIMD kernels)
5. 8x4 f64 NEON micro-kernel (16 accumulators, 4 A-vectors x 4 B-columns per k-step)
6. SIMD dispatch for Matrix element-wise ops (add/sub/scale via slice dispatch for large matrices)
7. SIMD vecmul via axpy_pos_dispatch (column-oriented AXPY)
8. Small-matrix specializations: matmul bypass for dims <= 3, vecmul bypass for dims <= 6, element-wise ops bypass for M*N <= 16
9. Column-major loop order + direct `data[j][i]` access throughout ops (no bounds-checked indexing)
10. B-panel packing for large dynamic matmul (n > 64): pack NR=4 columns of B contiguously per k-block
11. Direct 2x2/3x3 symmetric eigensolvers (closed-form, bypasses QR iteration)
12. Stable Givens rotations via hypot, Kahan compensated summation in norms/dot
13. SVD global σ_max deflation threshold (matches LAPACK DGESVD)

| Benchmark | numeris | nalgebra | faer | Winner |
|---|---|---|---|---|
| matmul 4x4 | 6.7 ns | **4.9 ns** | 56 ns | nalgebra |
| matmul 6x6 | 20.5 ns | **19.8 ns** | 86 ns | ~tie |
| matmul 50x50 (dyn) | **5.69 µs** | 6.64 µs | 6.10 µs | numeris |
| matmul 200x200 (dyn) | 306 µs | 293 µs | **179 µs** | faer |
| dot 100 (dyn) | **11.8 ns** | 12.3 ns | — | numeris |
| LU 4x4 | 32.7 ns | **28.2 ns** | 206 ns | nalgebra |
| LU 6x6 | 83.6 ns | **82.0 ns** | 303 ns | ~tie |
| LU 50x50 (dyn) | 8.49 µs | **7.44 µs** | 7.61 µs | nalgebra |
| Cholesky 4x4 | 25.1 ns | **11.5 ns** | 130 ns | nalgebra |
| Cholesky 6x6 | 70.3 ns | **39.3 ns** | 184 ns | nalgebra |
| QR 4x4 | **46.0 ns** | 90.6 ns | 299 ns | numeris |
| QR 6x6 | **83.4 ns** | 208 ns | 439 ns | numeris |
| SVD 4x4 | **449 ns** | 459 ns | 1.26 µs | ~tie |
| SVD 6x6 | **607 ns** | 915 ns | 1.85 µs | numeris |
| Inverse 4x4 | 27.6 ns | **23.1 ns** | — | nalgebra |
| Inverse 6x6 | 168 ns | **128 ns** | — | nalgebra |
| Eigen sym 4x4 | **178 ns** | 202 ns | 607 ns | numeris |
| Eigen sym 6x6 | **303 ns** | 535 ns | 1.08 µs | numeris |

## Observations

- **numeris wins** at QR (2–2.5x), SVD 6x6 (1.5x), symmetric eigendecomposition (1.1–1.8x), dot product, matmul 50x50
- **nalgebra wins** at Cholesky (2x — benchmark artifact: `Result` vs `Option` in `black_box`, actual computation within 4%), LU (small margin), inverse, matmul 4x4
- **faer wins** at large dynamic matmul (200x200) — A+B packing, cache-aware blocking
- faer has high overhead at small sizes due to dynamic dispatch / runtime machinery
- SVD 6x6 improved from 1171 ns → 607 ns (1.9x speedup) thanks to global σ_max deflation and stable Givens rotations
- Eigen sym 4x4/6x6: direct closed-form eigensolvers bypass QR iteration entirely for 2x2/3x3

## Notes

- **Cholesky 2x gap is a measurement artifact**: micro-benchmarking shows raw computation is within 4% of nalgebra (4.57 ns vs 4.40 ns for 4x4). The gap comes from Criterion's `black_box` reading `Result<CholeskyDecomposition, LinalgError>` byte-by-byte (48 `ldrb` instructions) vs nalgebra's `Option<Cholesky>` using word-sized `ldr` (17 instructions). Not a real performance difference.

## Remaining Optimization Opportunities

- **Large matmul**: A-panel packing + larger tile sizes could close remaining gap with faer (~1.7x)
- **Small matmul 4x4**: regressed from 4.9 ns to 6.7 ns — investigate Kahan summation overhead in hot path
- **LU**: small margin behind nalgebra — possibly similar `Result` vs `Option` artifact
