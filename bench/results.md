# Benchmark Results

Platform: Apple Silicon (aarch64, NEON), Rust stable, `cargo bench`
Date: 2026-03-07

## Current Results

Optimizations applied (cumulative):
1. 4-accumulator dot product, cascading matmul bottom-edge micro-kernels, Cholesky left-looking column-AXPY, AXPY small-size threshold
2. Direct 2x2/3x3/4x4 inverse formulas (adjugate, bypasses LU), direct det formulas
3. Unrolled small LU/Cholesky for N<=6 (direct `data[col][row]` array access, no trait dispatch)
4. Cache-blocked matmul (KC=256 k-blocking across all SIMD kernels)
5. 8x4 f64 NEON micro-kernel (16 accumulators, 4 A-vectors x 4 B-columns per k-step)
6. SIMD dispatch for Matrix element-wise ops (add/sub/scale via slice dispatch for large matrices)
7. SIMD vecmul via axpy_pos_dispatch (column-oriented AXPY)
8. Small-matrix specializations: matmul bypass for dims <= 3, vecmul bypass for dims <= 6, element-wise ops bypass for M*N <= 36
9. Column-major loop order + direct `data[j][i]` access throughout ops (no bounds-checked indexing)
10. B-panel packing for large dynamic matmul (n > 64): pack NR=4 columns of B contiguously per k-block

| Benchmark | numeris | nalgebra | faer | Winner |
|---|---|---|---|---|
| matmul 4x4 | **4.9 ns** | 4.9 ns | 58 ns | ~tie |
| matmul 6x6 | **13.4 ns** | 20.0 ns | 87 ns | numeris |
| matmul 50x50 (dyn) | **5.76 µs** | 6.63 µs | 6.3 µs | numeris |
| matmul 200x200 (dyn) | 369 µs | 361 µs | **193 µs** | faer |
| dot 100 (dyn) | **11.6 ns** | 14.5 ns | — | numeris |
| LU 4x4 | 33.2 ns | **28.2 ns** | 203 ns | nalgebra |
| LU 6x6 | 84.7 ns | **82.1 ns** | 292 ns | nalgebra |
| LU 50x50 (dyn) | 8.4 µs | **7.5 µs** | 7.7 µs | nalgebra |
| Cholesky 4x4 | 25.2 ns | **11.8 ns** | 139 ns | nalgebra |
| Cholesky 6x6 | 70.7 ns | **39.6 ns** | 186 ns | nalgebra |
| QR 4x4 | **46.4 ns** | 90.6 ns | 303 ns | numeris |
| QR 6x6 | **85.5 ns** | 207.9 ns | 445 ns | numeris |
| SVD 4x4 | **299 ns** | 461 ns | 1278 ns | numeris |
| SVD 6x6 | 1171 ns | **925 ns** | 1858 ns | nalgebra |
| Inverse 4x4 | 27.6 ns | **23.3 ns** | — | nalgebra |
| Inverse 6x6 | 163 ns | **127 ns** | — | nalgebra |
| Eigen sym 4x4 | **165 ns** | 201 ns | 578 ns | numeris |
| Eigen sym 6x6 | **287 ns** | 528 ns | 1088 ns | numeris |

## Observations

- **numeris wins** at matmul 6x6 (1.5x), QR (2x), SVD 4x4 (1.5x), symmetric eigendecomposition (1.2-1.8x), dot product, matmul 50x50
- **nalgebra wins** at Cholesky (2x — benchmark artifact: `Result` vs `Option` in `black_box`, actual computation within 4%), LU (small margin), inverse, SVD 6x6
- **faer wins** at large dynamic matmul (200x200) — A+B packing, cache-aware blocking
- faer has high overhead at small sizes due to dynamic dispatch / runtime machinery
- matmul 4x4: dead heat with nalgebra (4.9 ns)
- matmul 6x6: numeris 33% faster than nalgebra (13.4 ns vs 20.0 ns)

## Notes

- **Cholesky 2x gap is a measurement artifact**: micro-benchmarking shows raw computation is within 4% of nalgebra (4.57 ns vs 4.40 ns for 4x4). The gap comes from Criterion's `black_box` reading `Result<CholeskyDecomposition, LinalgError>` byte-by-byte (48 `ldrb` instructions) vs nalgebra's `Option<Cholesky>` using word-sized `ldr` (17 instructions). Not a real performance difference.

## Remaining Optimization Opportunities

- **Large matmul**: A-panel packing + larger tile sizes could close remaining gap with faer (~2x)
- **SVD 6x6**: 27% behind nalgebra — likely dominated by Givens rotations in bidiagonal QR
- **LU**: small margin behind nalgebra — possibly similar `Result` vs `Option` artifact
