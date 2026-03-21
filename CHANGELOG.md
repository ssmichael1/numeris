# Changelog

## 0.5.0

### Breaking Changes

- **Vector is now a column vector**: `Vector<T, N>` = `Matrix<T, N, 1>` (was `Matrix<T, 1, N>`).
  This matches nalgebra's convention. Single-index access `v[i]` still works.
- **`ColumnVector` removed**: Use `Vector` everywhere. `from_column()` replaced by `from_array()`.
- **`ColumnVector1`–`ColumnVector6` aliases removed**: Use `Vector1`–`Vector6`.
- **ODE `Solution` type**: Now `Solution<T, M, N>` (was `Solution<T, S>`), parameterized by
  matrix dimensions instead of a single size.
- **MSRV bumped to 1.77** (from 1.70).

### New Features

- **nalgebra interop** (`nalgebra` cargo feature):
  - `From`/`Into` conversions between `Matrix<T,M,N>` and `nalgebra::SMatrix<T,M,N>` (owned + ref).
  - `From`/`Into` for `DynMatrix<T>` / `DynVector<T>` and `nalgebra::DMatrix<T>` / `DVector<T>`.
  - `MatrixRef` / `MatrixMut` implemented for `nalgebra::SMatrix` and `DMatrix`, so numeris
    linalg free functions (`lu_in_place`, `cholesky_in_place`, `qr_in_place`, etc.) work
    directly on nalgebra matrices.
- **Matrix ODE integration**: `rk4`, `rk4_step`, and all `RKAdaptive` solvers now accept
  `Matrix<T, M, N>` as state (not just vectors), enabling matrix ODE integration
  (e.g., state transition matrices, matrix Riccati equations). Rosenbrock solvers remain
  vector-only due to Jacobian requirements.
- **`scaled_norm` generalized**: Moved from `Vector<T, N>` to `Matrix<T, M, N>`, using
  `frobenius_norm() / sqrt(M * N)`.

### Migration Guide

Replace `ColumnVector` with `Vector` and `from_column` with `from_array`:

```rust
// Before (0.4)
use numeris::ColumnVector;
let x = ColumnVector::from_column([1.0, 2.0]);
let val = x[(0, 0)];

// After (0.5)
use numeris::Vector;
let x = Vector::from_array([1.0, 2.0]);
let val = x[0];
```

ODE solution types gain a second dimension parameter:

```rust
// Before: Solution<T, S>
// After:  Solution<T, M, N>  (vectors are Solution<T, S, 1>)
```

## 0.4.0

- Numerical stability improvements and robustness guards.
- Spline diagonal check, matrix exponential overflow clamp, Cholesky SIMD scaling.

## 0.3.0

- Added `quad` module (Gauss-Legendre, adaptive Simpson, composite rules).
- Added `stats` sampling via xoshiro256++ RNG.
- Added lead/lag compensators, PID tuning (Ziegler-Nichols, Cohen-Coon, SIMC).
- Added `matrix!`/`vector!` macros, `prelude` module.

## 0.2.0

- Initial public release on crates.io.
- Matrix, linalg, quaternion, ODE, dynmatrix, interp, optim, estimate, control, special, stats modules.
- SIMD acceleration (NEON, SSE2, AVX, AVX-512).
