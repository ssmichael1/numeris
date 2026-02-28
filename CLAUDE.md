# numeris

Pure-Rust numerical algorithms library, no-std compatible. Similar in scope to SciPy.
Suitable for embedded targets (no heap allocation, no floating-point-unit assumptions).

## Module Plan

Checked items are implemented; unchecked are potential future work.

- [x] **matrix** — Fixed-size matrix (stack-allocated, const-generic dimensions)
- [~] **linalg** — Decompositions (LU, Cholesky done; QR, SVD future), solvers, inverse, determinant
- [ ] **ode** — ODE integration (RK4, RK45, adaptive step)
- [ ] **interp** — Interpolation (linear, cubic spline, Hermite)
- [ ] **optim** — Optimization (Newton, Levenberg-Marquardt, BFGS)
- [ ] **quad** — Numerical quadrature / integration
- [ ] **fft** — Fast Fourier Transform
- [ ] **special** — Special functions (Bessel, gamma, erf, etc.)
- [ ] **stats** — Statistics and distributions
- [ ] **quaternion** — Unit quaternion for rotations
- [ ] **poly** — Polynomial operations and root-finding

## Design Decisions

- **No-std / embedded first** — all code must work without `std` or heap allocation.
- **`num-traits`** for generic numeric bounds (`Zero`, `One`, `Num`, `Float`), with `default-features = false`.
- **Matrix storage** — `[[T; N]; M]` (M rows, N cols), row-major. Stack-allocated, contiguous in memory.
  Avoids `[T; M*N]` which requires unstable `generic_const_exprs`.
- **Const generics** — matrix dimensions are `const M: usize` (rows) and `const N: usize` (cols).
- **Naming** — `Matrix` is the fixed-size type (the default for embedded). Dynamic matrices will be
  `DynMatrix` (requires `alloc`) when added later. Shared behavior via traits.
- **Element traits** — `Scalar` (blanket trait: `Copy + PartialEq + Debug + Zero + One + Num`) for all
  matrix ops; `FloatScalar` (extends `Scalar + Float`) for operations needing floating-point (decompositions,
  trig, sqrt). Integer matrices work with just `Scalar`.
- **Matrix access traits** — `MatrixRef<T>` (read-only: `nrows`, `ncols`, `get`) and
  `MatrixMut<T>: MatrixRef<T>` (adds `get_mut`). Algorithms (Cholesky, LU, etc.) are written as
  free functions taking `&mut impl MatrixMut<T>` to operate in-place, avoiding the need for
  nalgebra-style allocator/storage traits. Both `Matrix` and future `DynMatrix` implement these.

## Cargo Features

- **`std`** (default) — enables `num-traits/std`, so float math (`sin`, `sqrt`, etc.) uses the
  system's native libm backed by hardware FPU. Full speed on desktop/server.
- **`libm`** — always enabled as baseline. Provides pure-Rust software float implementations
  via the `libm` crate. When `std` is also enabled, `std` takes precedence.
- **No-default-features** (`--no-default-features`) — `no_std` mode for embedded. Float math
  falls back to `libm` software implementations. No heap, no OS dependencies.

## File Layout

```
src/
├── lib.rs              # crate root, re-exports
├── traits.rs           # Scalar, FloatScalar, MatrixRef, MatrixMut
├── matrix/
│   ├── mod.rs          # Matrix struct, constructors, Index, trait impls
│   ├── ops.rs          # Add, Sub, Neg, Mul (matrix & scalar), vecmul, transpose
│   ├── square.rs       # trace, det, diag, from_diag, pow, is_symmetric
│   ├── vector.rs       # Vector, Vector3, ColumnVector, ColumnVector3, dot, cross
│   ├── norm.rs         # L1, L2, Frobenius, infinity, one norms, normalize
│   ├── slice.rs        # as_slice, row_slice, from_slice, iter, IntoIterator
│   └── util.rs         # from_fn, map, row/col access, Display
└── linalg/
    ├── mod.rs          # LinalgError
    ├── lu.rs           # LU decomposition, solve, inverse, det
    └── cholesky.rs     # Cholesky decomposition, solve, inverse, det, ln_det
```

## Current Focus

Next candidates: QR decomposition, quaternion module, ODE integration.
