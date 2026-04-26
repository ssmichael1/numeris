# Changelog

## 0.5.10

- **`imageproc::connected_components`** — connected-components labeling via two-pass SAUF
  (Scan + Array-based Union-Find), with path compression and union by rank. Operates on any
  `MatrixRef<T>` with foreground defined as `elem != background`; accepts 4- or 8-connectivity.
  Each `Component` reports `area`, inclusive bounding box, centroid, and central second moments
  (`mu20`, `mu02`, `mu11`). Two labels-image variants:
  - `connected_components_labeled` returns a `DynMatrix<u32>` (column-major, matches the rest
    of the imageproc API) for downstream masking.
  - `connected_components_with_label_buffer` returns a row-major flat `Vec<u32>` for
    cache-friendly downstream iteration when sweeping per-component bounding boxes in
    scan order.

## 0.5.9

- **`imageproc` feature** — 2D image processing on `DynMatrix<T>` buffers, column-major,
  `BorderMode`-aware (Zero / Constant / Replicate / Reflect), no-std with `alloc`. Convolution
  inner loops dispatch through the existing SIMD dot/AXPY kernels on contiguous column slices.
  - **Filtering**: `convolve2d` (any `MatrixRef` kernel), `convolve2d_separable`,
    `gaussian_blur`, `box_blur`, `unsharp_mask`, `laplacian`, `laplacian_of_gaussian`,
    `sobel_gradients`, `scharr_gradients`, `gradient_magnitude`.
  - **Order statistics**: `rank_filter`, `percentile_filter`, `median_filter` (quickselect
    with 3×3 and 5×5 stack-array fast paths), `median_filter_u16` (Huang sliding histogram,
    bit-exact vs. quickselect), `median_pool`, `median_pool_upsampled`.
  - **Morphology** (Van Herk - Gil-Werman, ~3 compares per pixel regardless of radius):
    `max_filter`, `min_filter`, `dilate`, `erode`, `opening`, `closing`, `morphology_gradient`,
    `top_hat`, `black_hat`.
  - **Local statistics** via integral image (O(1) per pixel): `integral_image`,
    `integral_rect_sum`, `local_mean`, `local_variance`, `local_stddev`.
  - **Multi-scale**: `difference_of_gaussians`, `gaussian_pyramid`.
  - **Thresholding**: `threshold`, `threshold_otsu` (256-bin between-class variance),
    `adaptive_threshold` (local mean + offset).
  - **Edges & corners**: `canny` (Gaussian → Sobel → NMS → hysteresis),
    `harris_corners`, `shi_tomasi_corners`.
  - **Geometric**: `flip_horizontal`, `flip_vertical`, `rotate_90` / `180` / `270`,
    `pad` (BorderMode-aware), `crop`, `resize_nearest`, `resize_bilinear` (precomputed
    per-axis tables, column-contiguous inner loop).
  - Kernel generators: `gaussian_kernel_1d`, `box_kernel_1d`, `sobel_x_3x3` / `sobel_y_3x3`,
    `scharr_x_3x3` / `scharr_y_3x3`, `laplacian_3x3` / `laplacian_3x3_diag`.
- **`DynMatrix::into_vec`** — zero-copy extraction of the underlying column-major `Vec<T>`
  by moving ownership out. Useful for recovering an owned pixel buffer after chaining
  through numeric routines.

## 0.5.7

- **Binary search interpolation** — dense output `interpolate` now uses `partition_point`
  for O(log n) step lookup instead of O(n) linear scan.
- **`interpolate_batch`** — new method for interpolating at multiple sorted time points
  in a single O(n+m) pass, avoiding repeated binary searches.

## 0.5.6

- **`Quaternion::rotation_between(a, b)`** — shortest-arc unit quaternion that rotates
  vector `a` to vector `b`. Normalizes inputs, handles parallel and anti-parallel cases.

## 0.5.5

- **`serde` feature** — optional serialization/deserialization for all types:
  - `Matrix<T, M, N>` serializes as row-major `[[1,2],[3,4]]` (matches `Matrix::new()` input).
  - `Vector<T, N>` serializes as flat `[1,2,3]`.
  - `Quaternion<T>` serializes as `{"w":1.0,"x":0.0,"y":0.0,"z":0.0}`.
  - `DynMatrix<T>` serializes as `{"nrows":2,"ncols":3,"data":[[1,2,3],[4,5,6]]}`.
  - `DynVector<T>` serializes as flat `[1,2,3]`.
  - `Solution` and `DenseOutput` also serializable.
- **Improved `Display` for `Matrix`**:
  - Precision formatting: `format!("{:.2}", m)` works.
  - Vectors display as compact `[1, 2, 3]` instead of columnar.
  - Padding inside `│` borders for readability.

## 0.5.4

- **`From<[T; N]>` for `Vector`** — `let v: Vector<f64, 3> = [1.0, 2.0, 3.0].into()`.
- **`Eq` and `Hash` for `Matrix`** — enables integer matrices as HashMap keys.
- **`Default` for `Matrix`** — returns the zero matrix.
- **`AsRef<[T]>` / `AsMut<[T]>` for `Matrix`** — view as flat column-major slice.
- **`solve_matrix` on `CholeskyDecomposition` and `QrDecomposition`** — multi-RHS solve.
- **`Quaternion::from_axis_angle` normalizes axis** — no longer requires pre-normalized input.

## 0.5.3

- **`Solution` and `DenseOutput` derive `Debug`**.
- **`DynVector` iterators**: `iter()`, `iter_mut()`, `IntoIterator` impls.
- **`DynVector::from_fn(n, |i| ...)`** constructor.
- **`Matrix::t()` / `DynMatrix::t()`** shorthand for `transpose()`.
- **`LuDecomposition::solve_matrix(&b)`** — multi-RHS solve (`AX = B`).
- **`PartialOrd` / `Ord` / `Eq` for `Vector`** — lexicographic ordering.

## 0.5.2

- **`DynVector` is now N×1** (column vector), matching `Vector<T, N>` = `Matrix<T, N, 1>`.
  Previously stored as 1×N internally — `nrows()` and `ncols()` now return the expected values.
- **`DynVector::zeros`** no longer takes a dummy parameter: `DynVector::<f64>::zeros(n)`.
- **`vecmul` removed** — use `a * v` (standard `Mul` trait) instead. `Vector` is now N×1 so
  `Matrix<T,M,N> * Vector<T,N>` works directly via matrix multiplication.
- **`vector!` macro** doc corrected: creates a column vector (N×1), not a row vector.

## 0.5.1

- **`DynMatrix::zeros`** no longer takes a dummy type-inference parameter.
  Use `DynMatrix::<f64>::zeros(m, n)` instead of `DynMatrix::zeros(m, n, 0.0_f64)`.
- **`DynMatrix::eye`** same change: `DynMatrix::<f64>::eye(n)` instead of `DynMatrix::eye(n, 0.0_f64)`.

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
