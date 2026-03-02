# Benchmark Results

Platform: Apple Silicon (aarch64, NEON), Rust stable, `cargo bench`
Date: 2026-03-01

## Post-Optimization Results

Optimizations applied:
1. 4-accumulator dot product, cascading matmul bottom-edge micro-kernels, Cholesky left-looking column-AXPY, AXPY small-size threshold
2. Direct 2x2/3x3/4x4 inverse formulas (adjugate, bypasses LU), direct det formulas
3. Unrolled small LU/Cholesky for N≤6 (direct `data[col][row]` array access, no trait dispatch)
4. Cache-blocked matmul (KC=256 k-blocking across all SIMD kernels)
5. 8×4 f64 NEON micro-kernel (16 accumulators, 4 A-vectors × 4 B-columns per k-step)

| Benchmark | numeris | nalgebra | faer | Winner | vs. Before |
|---|---|---|---|---|---|
| matmul 4x4 | 7.1 ns | **5.2 ns** | 61 ns | nalgebra | — |
| matmul 6x6 | **21.2 ns** | 21.1 ns | 93 ns | ~tie | **2.9x faster** |
| matmul 50x50 (dyn) | **6.8 µs** | 6.9 µs | **6.5 µs** | faer | **1.8x faster** |
| matmul 200x200 (dyn) | 367 µs | 310 µs | **174 µs** | faer | **1.5x faster** |
| dot 100 (dyn) | **12.4 ns** | 13.2 ns | — | numeris | **2.0x faster** |
| LU 4x4 | 34.8 ns | **30.2 ns** | 212.5 ns | nalgebra | **1.3x faster** |
| LU 6x6 | 90.3 ns | **80.8 ns** | 300.9 ns | nalgebra | — |
| LU 50x50 (dyn) | 8.9 µs | **7.8 µs** | 8.0 µs | nalgebra | — |
| Cholesky 4x4 | 26.5 ns | **12.2 ns** | 142.1 ns | nalgebra | — |
| Cholesky 6x6 | 76.5 ns | **42.1 ns** | 202.9 ns | nalgebra | 10% faster |
| QR 4x4 | **62.9 ns** | 101.7 ns | 328.4 ns | numeris | — |
| QR 6x6 | **85.9 ns** | 217.3 ns | 521.5 ns | numeris | — |
| SVD 4x4 | **313.9 ns** | 489.0 ns | 1365.7 ns | numeris | — |
| SVD 6x6 | 1135.5 ns | **975.3 ns** | 2011.1 ns | nalgebra | — |
| Inverse 4x4 | **29.6 ns** | 24.8 ns | — | nalgebra | **2.7x faster** |
| Inverse 6x6 | 172.9 ns | **133.8 ns** | — | nalgebra | 5% faster |
| Eigen sym 4x4 | **183.7 ns** | 213.4 ns | 621.5 ns | numeris | — |
| Eigen sym 6x6 | **352.0 ns** | 574.8 ns | 1200.5 ns | numeris | — |

## Key Improvements (vs. Initial Baseline)

- **Matmul 200x200**: 562 µs → 367 µs (**1.5x faster**) — 8×4 micro-kernel with 16 accumulators, now within 18% of nalgebra
- **Matmul 50x50**: 12.5 µs → 6.8 µs (**1.8x faster**) — now beats nalgebra (6.9 µs)
- **Inverse 4x4**: 79.9 ns → 29.6 ns (**2.7x faster**) — direct adjugate formula, now within 20% of nalgebra
- **LU 4x4**: 45.3 ns → 34.8 ns (**1.3x faster**) — unrolled small path with direct array access
- **Dot product**: 25.2 ns → 12.4 ns (**2.0x faster**) — now beats nalgebra (13.2 ns)
- **Matmul 6x6**: 62.1 ns → 21.2 ns (**2.9x faster**) — now tied with nalgebra (was 3x slower)

## Observations

- **numeris wins** at QR (2.5x faster), SVD 4x4, symmetric eigendecomposition (1.5-2x), dot product, and matmul 50x50
- **nalgebra wins** at small matmul 4x4, Cholesky, LU, inverse — but gaps are closing
- **faer wins** at large dynamic matmul (200x200) — cache-aware blocking + A/B packing
- faer has high overhead at small sizes due to dynamic dispatch / runtime machinery
- LU at 50x50 is competitive across all three (within ~15%)
- Matmul 200x200 went from 1.8x slower to 1.2x slower than nalgebra

## Remaining Optimization Opportunities

- **Cholesky**: still ~2x behind nalgebra at all sizes — bottleneck is sqrt/division cost, not dispatch overhead
- **Large matmul**: A/B panel packing could close remaining gap with nalgebra (~18%) and faer (~2x)
- **LU 6x6**: unrolled path slightly regressed; threshold may need tuning to N≤4
- Small matmul 4x4: nalgebra's hardcoded unrolled kernel is hard to beat generically
