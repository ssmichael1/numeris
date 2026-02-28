# numeris

Pure-Rust numerical algorithms library, no-std compatible. Similar in scope to SciPy.
Suitable for embedded targets (no heap allocation, no floating-point-unit assumptions).

## Module Plan

Checked items are implemented; unchecked are potential future work.

- [x] **matrix** — Fixed-size matrix (stack-allocated, const-generic dimensions), size aliases up to 6×6
- [x] **linalg** — LU, Cholesky, QR decompositions; solvers, inverse, determinant; complex support
- [x] **quaternion** — Unit quaternion for rotations (SLERP, Euler, axis-angle, rotation matrices)
- [x] **ode** — ODE integration (RK4, 7 adaptive solvers with PI step control, dense output, RODAS4 stiff solver)
- [x] **dynmatrix** — Heap-allocated runtime-sized matrix/vector (`alloc` feature)
- [ ] **interp** — Interpolation (linear, cubic spline, Hermite)
- [x] **optim** — Optimization (Brent, Newton, BFGS, Gauss-Newton, Levenberg-Marquardt)
- [ ] **quad** — Numerical quadrature / integration
- [ ] **fft** — Fast Fourier Transform
- [ ] **special** — Special functions (Bessel, gamma, erf, etc.)
- [ ] **stats** — Statistics and distributions
- [ ] **poly** — Polynomial operations and root-finding
- [x] **control** — Digital IIR filters (Butterworth, Chebyshev), PID controllers, state-space systems, discrete-time control (ZOH, Tustin bilinear transform)

## Design Decisions

- **No-std / embedded first** — all code must work without `std` or heap allocation.
- **`num-traits`** for generic numeric bounds (`Zero`, `One`, `Num`, `Float`), with `default-features = false`.
- **Matrix storage** — `[[T; N]; M]` (M rows, N cols), row-major. Stack-allocated, contiguous in memory.
  Avoids `[T; M*N]` which requires unstable `generic_const_exprs`.
- **Const generics** — matrix dimensions are `const M: usize` (rows) and `const N: usize` (cols).
- **Naming** — `Matrix` is the fixed-size type (the default for embedded). `DynMatrix` (requires `alloc`)
  for runtime-sized matrices. Shared behavior via `MatrixRef`/`MatrixMut` traits.
- **Element traits** — `Scalar` (blanket trait: `Copy + PartialEq + Debug + Zero + One + Num`) for all
  matrix ops; `FloatScalar` (extends `Scalar + Float`) for quaternions and ordered comparisons;
  `LinalgScalar` for decompositions and norms (covers both real floats and `Complex<T>`).
  Integer matrices work with just `Scalar`.
- **Matrix access traits** — `MatrixRef<T>` (read-only: `nrows`, `ncols`, `get`) and
  `MatrixMut<T>: MatrixRef<T>` (adds `get_mut`). Algorithms (Cholesky, LU, etc.) are written as
  free functions taking `&mut impl MatrixMut<T>` to operate in-place, avoiding the need for
  nalgebra-style allocator/storage traits. Both `Matrix` and `DynMatrix` implement these.
- **DynMatrix** — `Vec<T>` row-major storage with runtime dimensions. Implements `MatrixRef`/`MatrixMut`,
  so all linalg free functions work automatically. `DynVector` is a newtype wrapper enforcing 1-row
  constraint with single-index access. `DynLu`/`DynCholesky`/`DynQr` wrappers call the same generic
  free functions as the fixed-size decompositions.

## Cargo Features

- **`std`** (default) — implies `alloc`. Enables `num-traits/std`, so float math (`sin`, `sqrt`, etc.)
  uses the system's native libm backed by hardware FPU. Full speed on desktop/server.
- **`alloc`** — enables `DynMatrix` and `DynVector` (heap-allocated, runtime-sized). Implied by `std`.
- **`ode`** (default) — ODE integration module (RK4, adaptive solvers).
- **`optim`** — Optimization module (root finding, BFGS, Gauss-Newton, Levenberg-Marquardt).
- **`control`** — Digital IIR filters (Butterworth, Chebyshev Type I biquad cascades).
- **`libm`** — always enabled as baseline. Provides pure-Rust software float implementations
  via the `libm` crate. When `std` is also enabled, `std` takes precedence.
- **`complex`** — adds `Complex<f32>` / `Complex<f64>` support via `num-complex`. All decompositions
  and norms work with complex elements. Zero overhead for real-only code paths.
- **`all`** — enables all features: `std`, `ode`, `optim`, `control`, `complex`.
- **No-default-features** (`--no-default-features`) — `no_std` mode for embedded. Float math
  falls back to `libm` software implementations. No heap, no OS dependencies.

## File Layout

```
src/
├── lib.rs              # crate root, re-exports
├── traits.rs           # Scalar, FloatScalar, LinalgScalar, MatrixRef, MatrixMut
├── matrix/
│   ├── mod.rs          # Matrix struct, constructors, Index, trait impls
│   ├── aliases.rs      # Size aliases: Matrix1–Matrix6, Matrix2x3, Vector1–6, etc.
│   ├── ops.rs          # Add, Sub, Neg, Mul (matrix & scalar), vecmul, transpose
│   ├── square.rs       # trace, det, diag, from_diag, pow, is_symmetric
│   ├── vector.rs       # Vector, Vector3, ColumnVector, ColumnVector3, dot, cross
│   ├── block.rs        # block, set_block, top_left/right, head, tail, segment
│   ├── norm.rs         # L1, L2, Frobenius, infinity, one norms, normalize
│   ├── slice.rs        # as_slice, row_slice, from_slice, iter, IntoIterator
│   └── util.rs         # from_fn, map, row/col access, swap_rows/cols, Display
├── dynmatrix/          # (requires `alloc` feature)
│   ├── mod.rs          # DynMatrix struct, constructors, MatrixRef/MatrixMut, Index, conversions
│   ├── aliases.rs      # Scalar aliases: DynMatrixf64, DynVectorf32, DynMatrixz64, etc.
│   ├── ops.rs          # Add, Sub, Neg, Mul (matrix product), scalar Mul/Div, element_mul/div, transpose
│   ├── mixed_ops.rs    # Matrix<T,M,N> ↔ DynMatrix interop: Mul, Add, Sub
│   ├── vector.rs       # DynVector newtype, dot, Index<usize>, conversions
│   ├── square.rs       # trace, det, diag, from_diag, is_symmetric, pow
│   ├── norm.rs         # Frobenius, L1, L2, infinity, one norms, normalize
│   ├── block.rs        # block extraction/insertion (runtime dimensions)
│   ├── slice.rs        # as_slice, iter, IntoIterator
│   ├── util.rs         # from_fn, map, sum, swap, row/col, abs, element_max, Display
│   └── linalg.rs       # DynLu, DynCholesky, DynQr wrappers + convenience methods
├── linalg/
│   ├── mod.rs          # LinalgError
│   ├── lu.rs           # LU decomposition, solve, inverse, det
│   ├── cholesky.rs     # Cholesky decomposition, solve, inverse, det, ln_det
│   └── qr.rs           # QR decomposition, least-squares solve, det
├── ode/                # (requires `ode` feature)
│   ├── mod.rs          # OdeError, Solution, DenseOutput, re-exports
│   ├── rk4.rs          # Fixed-step classic RK4 (rk4_step, rk4)
│   ├── adaptive.rs     # RKAdaptive trait, AdaptiveSettings, PI step controller
│   ├── rkf45.rs        # Runge-Kutta-Fehlberg 4(5), 6 stages
│   ├── rkts54.rs       # Tsitouras 5(4), 7 stages, FSAL, 4th-degree interpolant
│   ├── rkv65.rs        # Verner 6(5), 10 stages, 6th-degree interpolant
│   ├── rkv87.rs        # Verner 8(7), 17 stages, 7th-degree interpolant
│   ├── rkv98.rs        # Verner 9(8), 21 stages, 8th-degree interpolant
│   ├── rkv98_nointerp.rs  # Verner 9(8) without interpolation, 16 stages
│   ├── rkv98_efficient.rs # Verner "efficient" 9(8), 26 stages, 9th-degree interpolant
│   ├── rosenbrock.rs      # Rosenbrock trait, fd_jacobian, integration loop
│   └── rodas4.rs          # RODAS4: 6-stage, order 4(3), L-stable Rosenbrock
├── control/            # (requires `control` feature)
│   ├── mod.rs          # ControlError, module declarations, re-exports
│   ├── biquad.rs       # Biquad, BiquadCascade, DFII-T tick/process, bilinear transform helpers
│   ├── butterworth.rs  # butterworth_lowpass, butterworth_highpass
│   ├── chebyshev.rs    # chebyshev1_lowpass, chebyshev1_highpass
│   ├── pid.rs          # Pid<T> discrete-time PID controller with anti-windup and derivative filter
│   └── tests.rs        # comprehensive tests
├── optim/              # (requires `optim` feature)
│   ├── mod.rs          # OptimError, result/settings structs, re-exports
│   ├── root.rs         # brent, newton_1d (scalar root finding)
│   ├── line_search.rs  # backtracking_armijo (internal helper)
│   ├── bfgs.rs         # minimize_bfgs (BFGS quasi-Newton)
│   ├── gauss_newton.rs # least_squares_gn (QR-based Gauss-Newton)
│   ├── levenberg_marquardt.rs # least_squares_lm (damped normal equations)
│   ├── jacobian.rs     # finite_difference_jacobian, finite_difference_gradient
│   └── tests.rs        # comprehensive tests
└── quaternion.rs       # Quaternion rotations, SLERP, Euler, axis-angle
```

## Current Focus

Next candidates: special functions (gamma), interpolation, SVD decomposition.
