# numeris

Pure-Rust numerical algorithms library, no-std compatible. Similar in scope to SciPy.
Suitable for embedded targets with no std feature (no heap allocation, no floating-point-unit assumptions),
but designed to be performant for more-powerful systems.

## Module Plan

Checked items are implemented; unchecked are potential future work.

- [x] **matrix** — Fixed-size matrix (stack-allocated, const-generic dimensions), size aliases up to 6×6
- [x] **linalg** — LU, Cholesky, QR, SVD decompositions; symmetric eigendecomposition (Householder + QR); real Schur decomposition (Hessenberg + Francis QR); solvers, inverse, determinant; complex support
- [x] **quaternion** — Unit quaternion for rotations (SLERP, Euler, axis-angle, rotation matrices)
- [x] **ode** — ODE integration (RK4, 7 adaptive solvers with PI step control, dense output, RODAS4 stiff solver)
- [x] **dynmatrix** — Heap-allocated runtime-sized matrix/vector (`alloc` feature)
- [x] **interp** — Interpolation (linear, Hermite, barycentric Lagrange, natural cubic spline)
- [x] **optim** — Optimization (Brent, Newton, BFGS, Gauss-Newton, Levenberg-Marquardt)
- [x] **estimate** — State estimation: EKF, UKF, SR-UKF, CKF, RTS smoother, batch least-squares
- [ ] **quad** — Numerical quadrature / integration
- [ ] **fft** — Fast Fourier Transform
- [x] **special** — Special functions (gamma, lgamma, digamma, beta, lbeta, incomplete gamma/beta, erf, erfc)
- [x] **stats** — Statistical distributions (Normal, Uniform, Exponential, Gamma, Beta, Chi-squared, Student's t, Bernoulli, Binomial, Poisson)
- [ ] **poly** — Polynomial operations and root-finding
- [x] **control** — Digital IIR filters (Butterworth, Chebyshev), PID controllers, state-space systems, discrete-time control (ZOH, Tustin bilinear transform)

## Design Decisions

- **No-std / embedded first, high-performance second** — all code must work without `std` or heap
  allocation, but on capable hardware it should be competitive with optimized libraries.
  SIMD intrinsics (`core::arch`) accelerate f32/f64 hot paths on aarch64 (NEON) and x86_64
  (SSE2/AVX/AVX-512) via compile-time `TypeId` dispatch, with zero-cost scalar fallback for
  integers and other types. No runtime feature detection — no cargo feature flag needed.
  SSE2 (x86_64) and NEON (aarch64) are always-on baseline. AVX and AVX-512 are compile-time
  opt-in via `-C target-cpu=native` or `-C target-feature=+avx2,+avx512f`. Dispatch selects
  the widest available ISA: AVX-512 > AVX > SSE2.
- **`num-traits`** for generic numeric bounds (`Zero`, `One`, `Num`, `Float`), with `default-features = false`.
- **Matrix storage** — `[[T; M]; N]` (N columns of M rows), column-major. Stack-allocated, contiguous
  in memory. Column-major matches LAPACK conventions and makes column-oriented linalg inner loops
  (Householder reflections, LU AXPY) operate on contiguous data for SIMD vectorization.
  `Matrix::new()` still accepts row-major input `[[row0], [row1], ...]` and transposes internally.
  Avoids `[T; M*N]` which requires unstable `generic_const_exprs`.
- **Const generics** — matrix dimensions are `const M: usize` (rows) and `const N: usize` (cols).
- **Naming** — `Matrix` is the fixed-size type (the default for embedded). `DynMatrix` (requires `alloc`)
  for runtime-sized matrices. Shared behavior via `MatrixRef`/`MatrixMut` traits.
- **Element traits** — `Scalar` (blanket trait: `Copy + PartialEq + Debug + Zero + One + Num`) for all
  matrix ops; `FloatScalar` (extends `Scalar + Float`) for quaternions and ordered comparisons;
  `LinalgScalar` for decompositions and norms (covers both real floats and `Complex<T>`).
  Integer matrices work with just `Scalar`.
- **Matrix access traits** — `MatrixRef<T>` (read-only: `nrows`, `ncols`, `get`, `col_as_slice`) and
  `MatrixMut<T>: MatrixRef<T>` (adds `get_mut`, `col_as_mut_slice`). `col_as_slice` returns a
  contiguous column sub-slice for SIMD-friendly inner loops. Algorithms (Cholesky, LU, etc.) are
  written as free functions taking `&mut impl MatrixMut<T>` to operate in-place, avoiding the need
  for nalgebra-style allocator/storage traits. Both `Matrix` and `DynMatrix` implement these.
- **DynMatrix** — `Vec<T>` column-major storage with runtime dimensions. Element `(row, col)` is at
  index `col * nrows + row`. `from_rows()` accepts row-major data (transposes internally);
  `from_slice()` accepts column-major data directly. Implements `MatrixRef`/`MatrixMut`,
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
- **`estimate`** — State estimation (EKF, UKF, SR-UKF, CKF, RTS smoother, batch LSQ). Implies `alloc` (sigma-point filters need temporary storage).
- **`interp`** — Interpolation (linear, Hermite, barycentric Lagrange, natural cubic spline).
- **`special`** — Special functions (gamma, lgamma, digamma, beta, lbeta, incomplete gamma/beta, erf, erfc).
- **`stats`** — Statistical distributions (Normal, Uniform, Exponential, Gamma, Beta, Chi-squared, Student's t, Bernoulli, Binomial, Poisson). Implies `special`.
- **`libm`** — always enabled as baseline. Provides pure-Rust software float implementations
  via the `libm` crate. When `std` is also enabled, `std` takes precedence.
- **`complex`** — adds `Complex<f32>` / `Complex<f64>` support via `num-complex`. All decompositions
  and norms work with complex elements. Zero overhead for real-only code paths.
- **`all`** — enables all features: `std`, `ode`, `optim`, `control`, `estimate`, `interp`, `special`, `stats`, `complex`.
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
│   ├── slice.rs        # as_slice, col_slice, from_slice, iter, IntoIterator
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
│   └── linalg.rs       # DynLu, DynCholesky, DynQr, DynSvd, DynSymmetricEigen, DynSchur wrappers
├── linalg/
│   ├── mod.rs          # LinalgError
│   ├── lu.rs           # LU decomposition, solve, inverse, det
│   ├── cholesky.rs     # Cholesky decomposition, solve, inverse, det, ln_det
│   ├── qr.rs           # QR decomposition, least-squares solve, det
│   ├── svd.rs          # Householder bidiagonalization, Golub-Kahan QR, SvdDecomposition wrapper
│   ├── symmetric_eigen.rs # Householder tridiagonalization, symmetric QR, SymmetricEigen wrapper
│   ├── hessenberg.rs   # Hessenberg reduction via Householder similarity transforms
│   └── schur.rs        # Francis double-shift QR, SchurDecomposition wrapper, eigenvalue extraction
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
├── simd/               # private SIMD acceleration (no cargo feature — always-on)
│   ├── mod.rs          # TypeId dispatch: dot, matmul, add/sub/scale/axpy slices
│   ├── scalar.rs       # generic scalar fallback (integers, complex, unknown arch)
│   ├── f64_neon.rs     # aarch64 NEON f64 kernels (2-wide)
│   ├── f32_neon.rs     # aarch64 NEON f32 kernels (4-wide)
│   ├── f64_sse2.rs     # x86_64 SSE2 f64 kernels (2-wide)
│   ├── f32_sse2.rs     # x86_64 SSE2 f32 kernels (4-wide)
│   ├── f64_avx.rs      # x86_64 AVX f64 kernels (4-wide, compile-time opt-in)
│   ├── f32_avx.rs      # x86_64 AVX f32 kernels (8-wide, compile-time opt-in)
│   ├── f64_avx512.rs   # x86_64 AVX-512 f64 kernels (8-wide, compile-time opt-in)
│   └── f32_avx512.rs   # x86_64 AVX-512 f32 kernels (16-wide, compile-time opt-in)
├── interp/             # (requires `interp` feature)
│   ├── mod.rs          # InterpError, find_interval, validate_sorted helpers, re-exports
│   ├── linear.rs       # LinearInterp<T, N> + DynLinearInterp<T>
│   ├── hermite.rs      # HermiteInterp<T, N> + DynHermiteInterp<T>
│   ├── lagrange.rs     # LagrangeInterp<T, N> + DynLagrangeInterp<T> (barycentric)
│   ├── spline.rs       # CubicSpline<T, N> + DynCubicSpline<T> (natural BCs, Thomas algorithm)
│   ├── bilinear.rs     # BilinearInterp<T, NX, NY> + DynBilinearInterp<T> (2D rectangular grid)
│   └── tests.rs        # comprehensive tests
├── control/            # (requires `control` feature)
│   ├── mod.rs          # ControlError, module declarations, re-exports
│   ├── biquad.rs       # Biquad, BiquadCascade, DFII-T tick/process, bilinear transform helpers
│   ├── butterworth.rs  # butterworth_lowpass, butterworth_highpass
│   ├── chebyshev.rs    # chebyshev1_lowpass, chebyshev1_highpass
│   ├── pid.rs          # Pid<T> discrete-time PID controller with anti-windup and derivative filter
│   └── tests.rs        # comprehensive tests
├── estimate/           # (requires `estimate` feature, implies `alloc`)
│   ├── mod.rs          # EstimateError, fd_jacobian, cholesky_with_jitter, apply_var_floor, re-exports
│   ├── ekf.rs          # Ekf<T, N, M> — EKF: predict/update/update_fd/update_gated/update_iterated
│   ├── ukf.rs          # Ukf<T, N, M> — UKF: predict/update/update_gated (requires `alloc`)
│   ├── cholupdate.rs   # Cholesky rank-1 update/downdate (private helper)
│   ├── srukf.rs        # SrUkf<T, N, M> — SR-UKF: predict/update/update_gated (requires `alloc`)
│   ├── ckf.rs          # Ckf<T, N, M> — CKF: predict/update/update_gated (requires `alloc`)
│   ├── rts.rs          # EkfStep, rts_smooth — RTS fixed-interval smoother (requires `alloc`)
│   ├── batch.rs        # BatchLsq<T, N> — Batch least-squares (fully no-std)
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
├── special/            # (requires `special` feature)
│   ├── mod.rs          # SpecialError, Lanczos constants, module decls, re-exports
│   ├── gamma_fn.rs     # gamma, lgamma (Lanczos approximation)
│   ├── digamma_fn.rs   # digamma (recurrence + asymptotic series)
│   ├── beta_fn.rs      # beta, lbeta (via lgamma)
│   ├── incgamma.rs     # gamma_inc, gamma_inc_upper (series + continued fraction)
│   ├── betainc.rs      # betainc — regularized incomplete beta I_x(a,b) (continued fraction)
│   ├── erf_fn.rs       # erf, erfc (via regularized incomplete gamma P(1/2, x²))
│   └── tests.rs        # comprehensive tests
├── stats/              # (requires `stats` feature, implies `special`)
│   ├── mod.rs          # ContinuousDistribution, DiscreteDistribution traits, StatsError, helpers
│   ├── normal.rs       # Normal<T> — Gaussian distribution
│   ├── uniform.rs      # Uniform<T> — continuous uniform on [a, b]
│   ├── exponential.rs  # Exponential<T> — exponential with rate λ
│   ├── gamma_dist.rs   # Gamma<T> — gamma with shape α and rate β
│   ├── beta_dist.rs    # Beta<T> — beta with shape parameters α, β
│   ├── chi_squared.rs  # ChiSquared<T> — chi-squared with k degrees of freedom
│   ├── student_t.rs    # StudentT<T> — Student's t with ν degrees of freedom
│   ├── bernoulli.rs    # Bernoulli<T> — Bernoulli with probability p
│   ├── binomial.rs     # Binomial<T> — binomial with n trials, probability p
│   ├── poisson.rs      # Poisson<T> — Poisson with rate λ
│   └── tests.rs        # comprehensive tests
└── quaternion.rs       # Quaternion rotations, SLERP, Euler, axis-angle
```

## Current Focus

Next candidates: SIMD extension to remaining linalg inner loops (QR, Cholesky, SVD Householder loops via col_as_slice + dot/AXPY dispatch).
