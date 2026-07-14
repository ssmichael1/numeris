# Changelog

## 0.5.15

- **Separable convolution rewritten as single-traversal SIMD tap sums —
  `gaussian_blur` ~1.4–2× faster** — internal-only; no API change. The two 1D
  passes in `convolve2d_separable` previously ran one full AXPY sweep over the
  image per kernel tap: each tap re-read and re-wrote the whole destination
  (3 memory ops per element per tap, after a `zeros` memset). Each pass now
  makes a single traversal through a new strided `conv1d` SIMD kernel
  (`out[i] = Σ_k kernel[k] · src[i + k·stride]`, stride 1 down a column,
  stride `nrows` across columns) that accumulates the whole tap sum in four
  vector registers and stores each output element exactly once. The kernel is
  generated per ISA from two shared macros (`simd_conv1d_kernel_fma` for NEON,
  `simd_conv1d_kernel_muladd` for SSE2/AVX/AVX-512) with a generic scalar
  fallback, dispatched via the existing `TypeId` scheme. Everything built on
  the separable passes speeds up: `gaussian_blur`, `box_blur`, `unsharp_mask`,
  `laplacian_of_gaussian`, `canny`, corners, DoG, Gaussian pyramid. Measured on
  an M1 Ultra (f64, σ=1/σ=2): 32² −49%/−48%, 128² −39%/−36%, 512² (parallel)
  −21%/−68%; ~2–4× faster than SciPy's `gaussian_filter` at matched truncation
  on the same machine. Summation order per pixel is unchanged (taps in index
  order), but border-adjacent pixels now add border taps after interior taps,
  so results can differ from 0.5.14 by float-rounding noise near edges.

## 0.5.14

- **SIMD kernels de-duplicated via macros (~790 fewer lines)** — internal-only;
  no API change and results are bit-for-bit identical. The element-wise
  `add_slices` / `sub_slices` / `scale_slices` and the `axpy_neg` / `axpy_pos`
  kernels were hand-written near-identically in all eight per-ISA files
  (f32/f64 × NEON/SSE2/AVX/AVX-512) — the bodies differ only in vector width
  and intrinsic names. They are now generated from three shared macros
  (`simd_elementwise_kernels`, `simd_axpy_kernels_muladd` for x86's separate
  multiply+add, `simd_axpy_kernels_fma` for NEON's fused multiply-add), each
  invoked once per file. The seven repetitive `*_dispatch` arch/type cascades in
  `simd/mod.rs` likewise collapse into one `simd_dispatch!` macro. `dot` and
  `matmul` keep their bespoke per-ISA kernels (their reductions / micro-kernels
  genuinely diverge). Adding a new ISA or element-wise op is now a one-line
  change instead of an eight-file edit. Verified: full suite on aarch64 (NEON)
  and the x86 SSE2 path executed under Rosetta both pass; AVX / AVX-512 paths
  type-check and lint clean.
- **Docs: corrected drift for the shipped `quad` module and the `ode` feature
  default** — the `quad` module (Gauss-Legendre, adaptive Simpson, composite
  rules) has been implemented and feature-gated for some time but was still
  listed as unchecked/future work in `CLAUDE.md` and `README.md` and omitted
  from the `CLAUDE.md` file layout and Cargo-feature list; it (and the
  previously-undocumented `macros.rs`, `prelude.rs`, `serde_impl.rs`,
  `control/lead_lag.rs`, `control/pid_tune.rs`, `stats/rng.rs`, and the `all`
  feature's `quad` + `imageproc` members) are now documented. The crate-level
  Cargo-features table also incorrectly marked `ode` as a default feature; it is
  opt-in (only `std` is default), matching `Cargo.toml` and the README.
- **QR: shared Q / Qᴴ / R / det between the fixed-size and `Dyn` wrappers —
  fixes a lost-SIMD regression in dynamic QR** — internal-only; results are
  bit-for-bit identical, but dynamic QR is now faster. `DynQr::q` /
  `DynQrPivot::q` and `DynQr::solve` reconstructed Q and applied Qᴴ with scalar
  inner loops, silently forgoing the SIMD-dispatched conjugated-dot / AXPY that
  the fixed-size `QrDecomposition` already used. The Householder-application math
  now lives in four `pub(crate)` free functions over `MatrixRef` / `MatrixMut`
  in `linalg::qr` — `form_q`, `apply_qh_inplace`, `copy_r`, `diag_product` —
  called by both the fixed-size and `Dyn` wrappers, so dynamic QR gets the
  vectorized path and the Q / Qᴴ / R / det logic has a single source of truth.
- **estimate: EKF `_fd` methods collapsed to delegations** — internal-only; no
  API change and behavior is identical (the `*_fd_matches_explicit` tests still
  pass). `predict_fd` / `update_fd` / `update_fd_gated` / `update_fd_iterated`
  were byte-for-byte copies of their explicit-Jacobian twins that only differed
  in sourcing the Jacobian from `fd_jacobian`; each now delegates to its twin
  with a `|x| fd_jacobian(&f, x)` closure (using `&F: Fn`), dropping ~80 lines.
- **estimate: removed the dead `cholupdate` module** — **(minor breaking change)**
  the private rank-1 Cholesky update/downdate helper was never used in production
  (SR-UKF re-Choleskys instead), and `linalg` already exposes public
  `cholesky_rank1_update` / `cholesky_rank1_downdate`. Deleting it also removes
  the now-unreachable `EstimateError::CholdowndateFailed` variant — the only
  public-API change in this batch. Match on `EstimateError` non-exhaustively or
  drop the arm.
- **stats: `ChiSquared` now delegates to an inner `Gamma`** — internal-only;
  results are identical. `ChiSquared(k)` is exactly `Gamma(k/2, 1/2)`, and every
  method (pdf / ln_pdf / cdf / mean / variance / sample, and even the
  Wilson-Hilferty quantile guess) reduces algebraically to Gamma's, so the
  duplicated formulas were replaced by a thin wrapper holding a `Gamma<T>`.
- **control: shared biquad-cascade builder for Butterworth / Chebyshev** —
  internal-only; filter coefficients are bit-for-bit identical. The four
  designers (`butterworth_{lowpass,highpass}`, `chebyshev1_{lowpass,highpass}`)
  repeated the same skeleton — bilinear pre-warp, a conjugate-pair loop, an
  odd-order real section, and (for Chebyshev) a passband-gain normalization.
  These now go through shared `pub(super)` helpers in `biquad.rs` (`prewarp`,
  `assemble_cascade`, `cascade_gain_at`, `scale_first_section_gain`), with the
  per-family pole placement factored into small `*_pole` helpers.
- **ode: dropped the phantom `JacSource` variant** — internal-only. The
  Rosenbrock Jacobian-source enum carried an unused `T` type parameter, forcing a
  `_Phantom(PhantomData<T>)` variant with an `unreachable!()` match arm. The enum
  only needs the closure type, so `T`/`S` (and the phantom) are gone.
- **estimate: shared UKF/CKF gain-update and predict-finalization** —
  internal-only; results are identical. The UKF and CKF had byte-for-byte
  identical `apply_update` bodies and predict tails (`γ·P_sigma + Q`, symmetrize,
  variance floor). These are now free functions in `estimate` — `sigma_point_update`,
  `store_predicted`, and the shared `symmetrize_and_floor`.
- **estimate: fixed a no_std build break** — `cholesky_with_jitter` built its
  jitter ladder with `10f64.powi(..)`, which does not exist without `std`, so
  `--no-default-features` builds enabling `estimate` failed to compile. The ladder
  is now written as `{1e-9, 1e-7, 1e-5}` literals. (The `simd_elementwise_kernels`
  macro also gained the `#[allow(unused_macros)]` its two siblings already had, so
  no_std non-SIMD targets like `thumbv7em` build warning-free.) CI now also builds
  the full no_std feature surface (`estimate`/`control`/`stats`/…) on `thumbv7em`,
  not just the core matrix baseline, so this class of regression can't recur.
- **Shared forward-difference Jacobian kernel** — internal-only; results are
  identical (the `ode` variant's last-bit `·(1/h)` → `/h` change is well within
  the adaptive solver's tolerance). `optim::finite_difference_jacobian`,
  `estimate::fd_jacobian`, and the Rosenbrock private `fd_jacobian` each re-encoded
  the same step-size policy `h_j = √ε·max(|x_j|, 1)`; they now share one
  `crate::fdiff::forward_diff_jacobian` (a new private module gated on
  `optim`/`estimate`/`ode`) that takes a precomputed `f0`, so the Rosenbrock hot
  loop still pays no extra evaluation. The scalar `finite_difference_gradient`
  keeps its own (scalar-output) form. The step-size policy itself is a single
  `fdiff::fd_step` helper, so the remaining forward-difference routines that
  don't share the vector kernel — the scalar gradient and the four dynamic FD
  functions (sequential + `rayon` parallel, Jacobian + gradient) — no longer
  re-spell `√ε·max(|x_j|, 1)` inline.

## 0.5.13

- **Review follow-ups: one less image pass and ~190 lines of duplication
  removed** — internal-only; results are arithmetically identical.
  - `imageproc`: `local_variance` (and `local_stddev` / `adaptive_threshold`
    built on it) no longer materializes a full squared copy of the image before
    building its second summed-area table; a fused `integral_image_with_squares`
    builds both tables in one pass over the source.
  - `interp`: the cubic Hermite basis evaluation and the natural-spline Thomas
    solve + Horner segment evaluation were duplicated between the fixed-size
    and `Dyn` variants; both now share module-level helpers
    (`hermite_segment[_with_derivative]`, `natural_spline_coeffs`,
    `cubic_eval[_with_derivative]`).
  - `stats`: the identical `sample_array` method was copy-pasted across all ten
    distributions; it is now generated by a small `impl_sample_array!` macro.
- **stats: parameter-only constants cached at construction** — internal-only;
  results are arithmetically identical. Distribution structs now precompute the
  parts of their (log-)density that depend only on the immutable parameters,
  instead of recomputing them on every `pdf` / `ln_pdf` / `pmf` call (which
  multiplies inside the Newton-based `quantile` solvers): `Gamma` caches
  `α·ln β − lgamma(α)`, `ChiSquared` caches `−(k/2)·ln 2 − lgamma(k/2)`,
  `StudentT` caches `lgamma((ν+1)/2) − lgamma(ν/2) − ½·ln(νπ)`, `Beta` caches
  `lbeta(α, β)`, `Binomial` caches `lgamma(n+1)`, `ln p`, `ln(1−p)`, `Poisson`
  caches `ln λ`, and `Normal` caches `σ·√(2π)` and `−ln σ − ½·ln(2π)`.
- **special: deduplicated incomplete-gamma core** — `erf`/`erfc` previously
  carried a ~85-line copy of the `incgamma` series/continued-fraction
  implementation (Option-returning instead of Result-returning). The core is
  now a single `pub(crate)` Option-based routine in `incgamma.rs`; the public
  `gamma_inc` / `gamma_inc_upper` wrap it with domain checks, and `erf`/`erfc`
  call it directly.
- **Fixed: `special`/`stats` now actually build under no-std** —
  `gamma_fn.rs` and the Box-Muller sampler called inherent `f64` math methods
  (`TAU.sqrt()`, `.ln()`, `.cos()`), which only exist with `std`, so
  `--no-default-features --features special,stats` failed to compile. The
  derived constants √(2π) and ½·ln(2π) are now precomputed literals (also
  removing a per-call recomputation in `gamma`/`lgamma`), and the sampler uses
  fully-qualified `num_traits::Float` calls (libm-backed in no-std, identical
  inherent methods with `std`).
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
