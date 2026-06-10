# Changelog

## 0.5.13

- **SIMD conjugated dot in linalg Householder loops** — internal-only. A new
  private `simd::dotc_dispatch` computes `Σ conj(aᵢ)·bᵢ`, forwarding to the
  SIMD `dot_dispatch` for `f32`/`f64` (where `conj` is the identity) with a
  scalar conjugating loop for complex elements. The column-oriented Householder
  reflection loops in QR (`qr_in_place`, `qr_col_pivot_in_place`, `q()`,
  `solve`), SVD bidiagonalization, and Hessenberg reduction previously computed
  `vᴴ·A[:,j]` with a scalar loop right next to a SIMD-dispatched AXPY; they now
  use the dispatched dot, and the redundant second `split_two_col_slices` call
  per column was removed by reordering the diagonal-element update after the
  AXPY. Slices shorter than 8 elements keep the inlined scalar loop — the
  out-of-line SIMD kernel call costs more than it saves there (a 4×4 QR
  regressed ~33% without the cutoff; with it, benchmarks show ~2× on 80×80
  dynamic QR and parity at 4×4/6×6).
- **Hot-loop performance cleanups** — internal-only; no API or behavior changes.
  - `estimate`: `Ukf`, `SrUkf`, and `Ckf` `update_gated` previously ran the full
    sigma-point measurement transform (Cholesky of P, sigma points, measurement
    transform, innovation covariance, Cholesky of S) to compute the NIS, then
    called `update`, which recomputed all of it from scratch. The transform is
    now factored into a shared private `measurement_transform` /
    `apply_update` pair used by both paths, roughly halving the cost of every
    accepted gated update (and deleting ~60 duplicated lines per filter).
    `predict` also reuses the new shared sigma-point generation.
  - `optim`: `least_squares_lm` computed `j.transpose()` twice per iteration
    (for `JᵀJ` and `Jᵀr`); it is now computed once.
    `finite_difference_jacobian_dyn` / `finite_difference_gradient_dyn` cloned
    the full state vector once per column; they now clone once and
    perturb/restore in place.
  - `ode`: the PID step-size controller coefficients (β₁, β₂, β₃) were rebuilt
    from `T::from(...)` conversions and divisions on every step in both the
    adaptive RK loop and the Rosenbrock loop; they are now hoisted above the
    step loop (along with `GAMMA_DIAG` in the Rosenbrock loop).
  - `imageproc`: `morphology_gradient`, `top_hat`, and `black_hat` replaced
    their per-element bounds-checked subtraction loops with the SIMD-dispatched
    slice subtraction already used by `DynMatrix` operators; `unsharp_mask`
    reuses the blurred buffer and iterates flat slices instead of per-element
    indexing (one less full-image allocation).
  - `dynmatrix`: by-value `Neg` negates in place instead of allocating a new
    backing vector.

## 0.5.12

- **`rayon` feature (opt-in parallelism)** — a new `rayon` Cargo feature adds
  multi-threaded parallelism on runtime-sized paths without disturbing the
  no-std-first baseline. It implies `std` and is purely additive: builds without
  it are byte-for-byte unchanged, and enabling it never alters an existing
  signature. Dispatch is hidden behind a private `par` module (mirroring `simd`).
  - First slice: new parallel finite-difference routines
    `finite_difference_jacobian_dyn_par` and `finite_difference_gradient_dyn_par`
    compute their columns across threads (each an independent evaluation of `f`),
    writing into disjoint slices so the result is identical regardless of thread
    count and equal to the sequential version. They require `Fn + Sync + Send`.
    Kept as *separate* functions (not a feature-gated change to the sequential
    `finite_difference_jacobian_dyn` / `_gradient_dyn`, which keep their `FnMut`
    signatures) so that enabling `rayon` — possibly transitively, via another
    crate — can never break a caller passing a `FnMut` closure. Parallelism
    engages only above a column threshold, biased toward the regime it helps (an
    expensive `f`); see `bench/fd_jacobian` and `bench/results.md` for the
    measured crossover (≈4× at n=256 with a moderately expensive evaluation).
  - Second slice: separable convolution (`convolve2d_separable`, and thus
    `gaussian_blur` / `box_blur` and everything built on them — `unsharp_mask`,
    `laplacian_of_gaussian`, `canny`, `harris_corners`, `shi_tomasi_corners`,
    `difference_of_gaussians`, `gaussian_pyramid`) computes its output columns in
    parallel. Each output column is a disjoint write, so results are unchanged.
    The fan-out decision is gated on per-pass *work* (`nrows · ncols · klen`),
    not column count, so it accounts for image height and kernel size — a small
    image stays sequential, a large one parallelizes (~2.6× at 512² for a small
    Gaussian; see `bench/convolve`).
  - Third slice: more `imageproc` per-column kernels — the rank / median filters
    (`rank_filter`, `percentile_filter`, `median_filter` and its 3×3/5×5 fast
    paths), `resize_bilinear`, and the local-statistics queries (`local_mean`,
    `local_variance`, `local_stddev`, and `adaptive_threshold` / `median_pool_upsampled`
    built on them). The median quickselect is the most expensive per-pixel work
    in imageproc — ~3.4–3.7× at 256² (see `bench/rank`). Fan-out is gated on
    per-pass work scaled by window area, and a shared `par::work_col_threshold`
    helper now backs all the imageproc gates. That helper also scales the work
    budget by `rayon::current_num_threads()` (normalized to the 8-core tuning
    machine), so the fan-out crossover adapts to the host core count — smaller
    inputs parallelize on a 2-core laptop, larger ones are required on a
    many-core server. The summed-area-table build under the local-stats queries
    stays sequential (it is a prefix-sum scan, a separate decomposition).
  - Fourth slice: morphology (`max_filter`/`min_filter`, `dilate`/`erode`,
    `opening`/`closing`, `morphology_gradient`, `top_hat`, `black_hat`). Under
    `rayon` the separable Van Herk filter runs the horizontal direction as a
    second *vertical* pass in transposed space (`out = (V(T(V(src))))ᵀ`), so both
    filter passes and both transposes parallelize over output columns while
    keeping O(1)-per-pixel cost — ~3.9–4.1× at 512²–1024² (`bench/morphology`).
    The change is `cfg`-split: the no-`rayon` build keeps the original lean
    sequential pass (two buffers + row scratch, no transposes), so the embedded
    baseline is unchanged; the two extra full-image allocations are paid only
    when `rayon` is enabled.
  - The `Send + Sync` requirement these parallel paths place on the element type
    is expressed through a hidden `MaybeSync` marker bound that is empty (a
    blanket impl for all types) unless `rayon` is enabled, so non-`rayon`
    signatures are unconstrained and `f32`/`f64` satisfy it automatically when it
    applies.
  - Fixed-size `Matrix` Jacobians and other small stack-allocated paths stay
    sequential by design.

- **MSRV corrected to 1.80.** The declared `rust-version` was `1.77`, but core
  matrix code uses `[T]::as_flattened` / `as_flattened_mut` (stabilized in Rust
  1.80), so the crate never actually built on 1.77 with a fresh lockfile. The
  declaration now reflects reality. (rayon also requires 1.80, so the feature
  does not raise the floor.)

- **Clippy clean + CI gate.** The codebase is now `clippy -D warnings` clean and
  a clippy job was added to CI. Incidental additions from the cleanup:
  `Vector::is_empty`, a `Default` impl for `BatchLsq`, and a `SmoothedStates<T, N>`
  type alias for the `rts_smooth` return type. `Pid::with_derivative_filter` now
  rejects a NaN time constant (the precondition assert was `!(tau < 0)`, which
  let NaN through; it is now `tau >= 0`).

## 0.5.11

- **Dynamic-dimension `optim` routines** — `minimize_bfgs_dyn`,
  `least_squares_gn_dyn`, `least_squares_lm_dyn`,
  `finite_difference_gradient_dyn`, and `finite_difference_jacobian_dyn` mirror
  the fixed-size API but accept `DynVector<T>` / `DynMatrix<T>`, picking the
  parameter and residual dimensions at runtime. Settings structs
  (`BfgsSettings`, `GaussNewtonSettings`, `LmSettings`) and `OptimError` are
  shared with the fixed-size routines; results come back as
  `MinimizeResultDyn<T>` / `LeastSquaresResultDyn<T>`. The dynamic variants
  require the `alloc` feature; fixed-size routines remain no-alloc.

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
