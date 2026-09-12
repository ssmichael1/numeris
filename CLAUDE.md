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
- [x] **imageproc** — 2D image processing (filters, morphology, integral image/local stats, multi-scale, thresholding, Canny, corners, connected components, geometric)
- [x] **optim** — Optimization (Brent, Newton, BFGS, Gauss-Newton, Levenberg-Marquardt; `_dyn` variants on `DynVector`/`DynMatrix`)
- [x] **estimate** — State estimation: EKF, UKF, SR-UKF, CKF, RTS smoother, batch least-squares
- [x] **quad** — Numerical quadrature (Gauss-Legendre, adaptive Simpson, composite trapezoid/Simpson)
- [x] **fft** — Fast Fourier Transform (fixed-size no-alloc complex + real; `DynFft` for any length via Bluestein; 2D `DynFft2`/`DynRealFft2`; 1D/2D convolution; SIMD butterflies in the alloc tier; rayon-parallel 2D batches)
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
  These flags are **not** committed to a repo `.cargo/config.toml` (a blanket `target-cpu=native`
  is non-portable and makes virtualized CI runners `SIGILL` on AVX-512 the host detects but can't
  run). To build with wide SIMD locally, opt in per-shell, e.g.
  `RUSTFLAGS="-C target-cpu=native" cargo bench`. CI sets `target-cpu=x86-64-v3` for the x86_64
  target only (via `CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS`) to exercise the AVX path
  deterministically; aarch64 runners test NEON on the baseline.
- **`unsafe` discipline in `simd/`** — the SIMD kernels hold nearly all of the crate's
  `unsafe` (the remainder — `linalg`'s two-column split and `quad`'s `MaybeUninit` stack —
  is likewise block-documented), and
  four rules keep it auditable. (1) *No `#[target_feature]` on the kernels*: the ISA modules
  are `#[cfg(target_feature = ...)]`-gated, so availability is a property of the compilation
  unit; adding the attribute would only turn safe fns into `unsafe fn` (pre-1.86) and buys
  nothing without runtime detection, which no-std rules out anyway. (2) *One cast witness*:
  the generic-to-concrete reinterpretation in `simd/mod.rs` goes through `TypeEq<T, U>`, whose
  sole constructor performs the `TypeId` check — no dispatch site contains `unsafe`, and a
  test/cast type mismatch is unrepresentable. (3) *Structural bounds*: kernels iterate
  `chunks_exact` so each proof is "the chunk is exactly as wide as the loads covering it"
  rather than hand-computed offsets; where that is impossible (`conv1d`'s strided reads) the
  precondition is a real `assert!` at function entry, not a `debug_assert!` (this applies to the
  SIMD `matmul` length checks and `split_two_col_slices`' disjointness check too, not only
  `conv1d`). (4) *Every block
  documented*: `unsafe_op_in_unsafe_fn` is warned on crate-wide, every `unsafe fn` carries a
  `# Safety` section, and every `unsafe` block a `// SAFETY:` comment stating the argument it
  relies on (call sites restate how the caller meets the callee's contract) — enforced by
  `clippy::undocumented_unsafe_blocks` (configured in `clippy.toml`), which covers the private
  items `clippy::missing_safety_doc` does not. Confinement itself is also compiler-enforced:
  `#![deny(unsafe_code)]` at the crate root, with `#[allow(unsafe_code)]` on exactly the audited
  sites (`simd`, `linalg::split_two_col_slices`, `quad::adaptive_simpson`) — add a new site only
  with the same audit treatment, never by widening an existing `allow`. Style: prefer one `chunks_exact` iterator per loop and take `remainder()` from
  it, rather than re-calling `chunks_exact`.
- **Benchmarking `simd/` changes — alignment sensitivity** — the fixed-size `comparison`
  benchmarks run in 80–200 ns and are sensitive to *code alignment* at the ±10% level. During
  the 0.5.16 refactor, an edit to `dot` moved `lu_6x6`/`inverse_6x6` by 12–14%, reproducibly
  and with tight confidence intervals, on benchmarks that never call `dot`. Disassembly showed
  `LuDecomposition<f64, 6>::new` had a byte-identical instruction stream in both builds — only
  its address moved (0x1000700e0 → 0x100070024), because `dot` had shrunk the preceding code by
  188 bytes. The effect is discontinuous in the shift: a semantically inert never-called
  function inserted at the same point moved the same benchmarks by under 2%. So **reproducibility
  and a small p-value do not distinguish a real kernel change from an alignment shift.** Before
  attributing a swing on these benchmarks to your edit, disassemble the affected function
  (`nm` for the symbol, `objdump -d --start-address=…`) and check whether its code actually
  changed; if it did not, you are looking at layout, and "fixing" it by reshaping the kernel
  will not survive the next unrelated edit. Prefer the `_dyn` / larger-size benchmarks and
  `convolve`/`morphology` for judging real kernel throughput.
- **`num-traits`** for generic numeric bounds (`Zero`, `One`, `Num`, `Float`), with `default-features = false`.
- **Matrix storage** — `[[T; M]; N]` (N columns of M rows), column-major. Stack-allocated, contiguous
  in memory. Column-major matches LAPACK conventions and makes column-oriented linalg inner loops
  (Householder reflections, LU AXPY) operate on contiguous data for SIMD vectorization.
  `Matrix::new()` still accepts row-major input `[[row0], [row1], ...]` and transposes internally.
  Avoids `[T; M*N]` which requires unstable `generic_const_exprs`.
- **Const generics** — matrix dimensions are `const M: usize` (rows) and `const N: usize` (cols).
- **Naming** — `Matrix` is the fixed-size type (the default for embedded). `DynMatrix` (requires `alloc`)
  for runtime-sized matrices. `Vector<T, N>` = `Matrix<T, N, 1>` (column vector, matching nalgebra).
  Shared behavior via `MatrixRef`/`MatrixMut` traits.
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
- **`ode`** — ODE integration module (RK4, adaptive solvers).
- **`optim`** — Optimization module (root finding, BFGS, Gauss-Newton, Levenberg-Marquardt).
- **`control`** — Digital IIR filters (Butterworth, Chebyshev Type I biquad cascades).
- **`estimate`** — State estimation (EKF, UKF, SR-UKF, CKF, RTS smoother, batch LSQ). Implies `alloc` (sigma-point filters need temporary storage).
- **`interp`** — Interpolation (linear, Hermite, barycentric Lagrange, natural cubic spline).
- **`imageproc`** — 2D image processing on `DynMatrix` (convolution, filters, morphology, integral image / local stats, thresholding, Canny, Harris/Shi-Tomasi corners, DoG / Gaussian pyramid, connected components, geometric ops, `BorderMode`). Implies `alloc`.
- **`quad`** — Numerical quadrature (Gauss-Legendre, adaptive Simpson, composite trapezoid/Simpson). All no-alloc.
- **`fft`** — Fast Fourier Transform. Fixed-size no-alloc complex FFT (power-of-two `N ≤ 4096`,
  with/without a precomputed `TwiddleTable`) and real `rfft`/`irfft`. With `alloc`: `DynFft`
  planner for any length (power-of-two radix + Bluestein chirp-z for prime/awkward sizes),
  `DynRealFft` (half-size packing both directions), FFT-based `fft_convolve`/`fft_correlate`
  (power-of-two padded, real plan), 2D `DynFft2`/`DynRealFft2` over column-major `DynMatrix`
  (separable row–column; the strided row axis runs on a cache-blocked transposed copy so both
  passes are contiguous SIMD batches, `rayon`-parallel over columns above the shared work gate),
  and `fft_convolve2d`/`fft_correlate2d`. Plans are read-only during a transform: `make_scratch`
  + `forward_with`/`inverse_with` (`DynFftScratch`/`DynRealFftScratch`) run a shared `&` plan
  through a caller-owned scratch — the mechanism the 2D batches use per worker.
  `fftshift`/`ifftshift` are no-alloc and `fftshift2d`/`ifftshift2d` allocate nothing.
  The `DynFft` power-of-two path deinterleaves to structure-of-arrays re/im and runs radix-4
  (plus one trailing radix-2) butterflies through shared per-ISA SIMD kernel macros (NEON/SSE2/AVX/AVX-512, scalar
  fallback); the no-std fixed tier stays scalar. Re-exports `num_complex::Complex` (like `complex`).
- **`special`** — Special functions (gamma, lgamma, digamma, beta, lbeta, incomplete gamma/beta, erf, erfc).
- **`stats`** — Statistical distributions (Normal, Uniform, Exponential, Gamma, Beta, Chi-squared, Student's t, Bernoulli, Binomial, Poisson). Implies `special`.
- **`libm`** — always enabled as baseline. Provides pure-Rust software float implementations
  via the `libm` crate. When `std` is also enabled, `std` takes precedence.
- **`complex`** — adds `Complex<f32>` / `Complex<f64>` support via `num-complex`. All decompositions
  and norms work with complex elements. Zero overhead for real-only code paths.
- **`nalgebra`** — conversions between numeris and nalgebra types (`From`/`Into`, `MatrixRef`/`MatrixMut` impls).
  Enables `nalgebra/std`. `nalgebra::SMatrix` and `DMatrix` can be used directly with numeris linalg free functions.
- **`serde`** — serialize/deserialize `Matrix`, `Vector`, `Quaternion`, `DynMatrix`, `DynVector`, `Solution`.
  Row-major format for matrices (matches `Matrix::new()`), flat arrays for vectors.
- **`rayon`** — opt-in multi-threaded parallelism on runtime-sized paths (heap-backed `DynMatrix` /
  `imageproc` / `_dyn` routines). Implies `std` (rayon needs threads). The crate MSRV is 1.80 (the
  floor for `[T]::as_flattened` in core matrix code, and for rayon). Purely additive: no-std builds are
  unaffected and enabling it never changes an existing signature. Dispatch lives in the private `par` module (mirrors `simd`), gated on
  `any(imageproc, all(optim, alloc, rayon))`. Only disjoint-output operations (Jacobian columns, image
  columns) are parallelized — never order-sensitive reductions. Users so far:
  `optim::finite_difference_jacobian_dyn_par` / `finite_difference_gradient_dyn_par` — **separate
  `_par` functions** requiring `Fn + Sync + Send`, so the sequential `_dyn` routines keep their
  `FnMut` signatures (additive, no feature-unification footgun) — plus many `imageproc` per-column
  kernels: separable convolution (`convolve2d_separable` →
  `gaussian_blur`/`box_blur` and all filters built on them: `unsharp_mask`, `laplacian_of_gaussian`,
  `canny`, Harris/Shi-Tomasi corners, DoG, Gaussian pyramid), rank/median filters (`rank_filter`,
  `percentile_filter`, `median_filter`), `resize_bilinear`, local-statistics queries
  (`local_mean`/`local_variance`/`local_stddev`, `adaptive_threshold`), and morphology
  (`dilate`/`erode`/`opening`/`closing`/`max_filter`/`min_filter`/gradient/top-hat/black-hat). All
  parallelize over output columns, gated on per-pass work via the shared `par::work_col_threshold`
  helper. Morphology's horizontal Van Herk pass is run as a transposed vertical pass under `rayon`
  (`out = (V(T(V(src))))ᵀ`), `cfg`-split so the no-`rayon` build keeps the lean two-buffer sequential
  pass. Separable convolution (`convolve2d_separable` / `_into`, hence `gaussian_blur` and everything built
  on it) is *banded*: it parallelizes over bands of output columns through `par::for_each_chunk_mut_init`,
  each band running its vertical pass into a per-job halo slab and its horizontal pass from that slab
  into its own disjoint columns of `dst` — no whole-image intermediate. The `fft` 2D transforms
  (`DynFft2` / `DynRealFft2`, hence `fft_convolve2d`) batch their column passes and cache-blocked
  transposes through the same helpers, each worker building its own `DynFftScratch` via
  `for_each_chunk_mut_init`, so the plan (twiddles) is shared read-only; a single 1D FFT is never
  multithreaded (its stages are a serial chain). Not yet parallel: the integral-image scan (prefix sum; needs a two-pass decomposition).
  The `imageproc` / `fft` `Send + Sync` element requirement is carried by a hidden `par::MaybeSync` marker
  bound (empty blanket impl without `rayon`, `Send + Sync` with it; gated on `imageproc` or `fft`+`alloc`), so a single
  signature serves both builds without `cfg`-split twins — invisible for `f32`/`f64`, hence additive.
- **`all`** — enables all features: `std`, `ode`, `optim`, `quad`, `control`, `estimate`, `interp`, `imageproc`, `fft`, `special`, `stats`, `complex`, `nalgebra`, `serde`, `rayon`.
- **No-default-features** (`--no-default-features`) — `no_std` mode for embedded. Float math
  falls back to `libm` software implementations. No heap, no OS dependencies.

## File Layout

```
src/
├── lib.rs              # crate root, re-exports
├── traits.rs           # Scalar, FloatScalar, LinalgScalar, MatrixRef, MatrixMut
├── macros.rs           # matrix! / vector! constructor macros
├── prelude.rs          # convenience re-exports (Matrix, Vector, Quaternion, traits)
├── serde_impl.rs       # (requires `serde` feature) Serialize/Deserialize for all types
├── fdiff.rs            # (requires `optim`/`estimate`/`ode`) shared forward-difference Jacobian kernel
├── matrix/
│   ├── mod.rs          # Matrix struct, constructors, Index, trait impls
│   ├── aliases.rs      # Size aliases: Matrix1–Matrix6, Matrix2x3, Vector1–6, etc.
│   ├── ops.rs          # Add, Sub, Neg, Mul (matrix & scalar), transpose
│   ├── square.rs       # trace, det, diag, from_diag, pow, is_symmetric
│   ├── vector.rs       # Vector (N×1 column), Vector3, dot, cross, outer
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
│   ├── mod.rs          # TypeId dispatch (via the `TypeEq` cast witness): dot, matmul,
│   │                   #   add/sub/scale/scale-in-place/axpy slices, strided conv1d,
│   │                   #   fft_butterfly / fft_butterfly4 (SoA radix-2 / radix-4, macro-shared across ISAs)
│   ├── scalar.rs       # generic scalar fallback (integers, complex, unknown arch); fft_butterfly / fft_butterfly4 references
│   ├── f64_neon.rs     # aarch64 NEON f64 kernels (2-wide)
│   ├── f32_neon.rs     # aarch64 NEON f32 kernels (4-wide)
│   ├── f64_sse2.rs     # x86_64 SSE2 f64 kernels (2-wide)
│   ├── f32_sse2.rs     # x86_64 SSE2 f32 kernels (4-wide)
│   ├── f64_avx.rs      # x86_64 AVX f64 kernels (4-wide, compile-time opt-in)
│   ├── f32_avx.rs      # x86_64 AVX f32 kernels (8-wide, compile-time opt-in)
│   ├── f64_avx512.rs   # x86_64 AVX-512 f64 kernels (8-wide, compile-time opt-in)
│   └── f32_avx512.rs   # x86_64 AVX-512 f32 kernels (16-wide, compile-time opt-in)
├── par/                # private parallelism dispatch (requires `rayon` feature to multi-thread; gated on imageproc / fft+alloc / optim+rayon)
│   └── mod.rs          # for_each_chunk_mut (sequential chunks_mut / rayon par_chunks_mut over disjoint output chunks); for_each_chunk_mut_init (same, plus a per-worker scratch value via for_each_init — for the banded separable convolution and the fft 2D batches); work_col_threshold; MaybeSync marker bound
├── nalgebra_interop.rs # (requires `nalgebra` feature) From/Into, MatrixRef/MatrixMut for nalgebra types
├── interp/             # (requires `interp` feature)
│   ├── mod.rs          # InterpError, find_interval, validate_sorted helpers, re-exports
│   ├── linear.rs       # LinearInterp<T, N> + DynLinearInterp<T>
│   ├── hermite.rs      # HermiteInterp<T, N> + DynHermiteInterp<T>
│   ├── lagrange.rs     # LagrangeInterp<T, N> + DynLagrangeInterp<T> (barycentric)
│   ├── spline.rs       # CubicSpline<T, N> + DynCubicSpline<T> (natural BCs, Thomas algorithm)
│   ├── bilinear.rs     # BilinearInterp<T, NX, NY> + DynBilinearInterp<T> (2D rectangular grid)
│   └── tests.rs        # comprehensive tests
├── imageproc/          # (requires `imageproc` feature, implies `alloc`)
│   ├── mod.rs          # ImageError, module decls, re-exports
│   ├── border.rs       # BorderMode<T> (Zero/Constant/Replicate/Reflect), fetch_border
│   ├── kernels.rs      # gaussian_kernel_1d, box_kernel_1d, sobel/scharr/laplacian 3x3
│   ├── convolve.rs     # convolve2d (dense, any MatrixRef kernel), convolve2d_separable + _into (banded: per-band vertical pass into a halo slab, horizontal pass from it; single-pass strided SIMD tap sums; test-only two-pass reference pins bit-identity)
│   ├── filters.rs      # gaussian_blur (+ _into, gaussian_blur_kernel), box_blur, sobel/scharr_gradients, laplacian, laplacian_of_gaussian, unsharp_mask, gradient_magnitude
│   ├── geometric.rs    # flip_horizontal/vertical, rotate_90/180/270, pad (BorderMode-aware), crop, resize_nearest
│   ├── integral.rs     # integral_image (SAT), integral_rect_sum (O(1) rectangle query)
│   ├── local_stats.rs  # local_mean, local_variance, local_stddev via integral images (O(1) per pixel)
│   ├── morphology.rs   # max/min_filter, dilate/erode (Van Herk O(1) amortized); opening, closing, morphology_gradient, top_hat, black_hat
│   ├── multiscale.rs   # difference_of_gaussians, gaussian_pyramid (blur + 2× decimate)
│   ├── pool.rs         # median_pool (block-decimating), median_pool_upsampled (pool + bilinear)
│   ├── rank.rs         # rank/percentile/median_filter (radius 1,2 stack-array fast paths, else quickselect); median_filter_u16 (Huang sliding histogram)
│   ├── resize.rs       # resize_bilinear (precomputed tables, column-contiguous inner loop)
│   ├── threshold.rs    # threshold (binary), threshold_otsu (256-bin between-class variance), adaptive_threshold (local mean + offset)
│   ├── canny.rs        # Canny edge detector (Gaussian → Sobel → NMS → double threshold → hysteresis)
│   ├── corners.rs      # harris_corners, shi_tomasi_corners (structure tensor + response)
│   ├── connected.rs    # Connectivity enum, Component, connected_components, connected_components_labeled, connected_components_with_label_buffer (SAUF union-find)
│   └── tests.rs        # comprehensive tests
├── control/            # (requires `control` feature)
│   ├── mod.rs          # ControlError, module declarations, re-exports
│   ├── biquad.rs       # Biquad, BiquadCascade, DFII-T tick/process, bilinear transform helpers
│   ├── butterworth.rs  # butterworth_lowpass, butterworth_highpass
│   ├── chebyshev.rs    # chebyshev1_lowpass, chebyshev1_highpass
│   ├── pid.rs          # Pid<T> discrete-time PID controller with anti-windup and derivative filter
│   ├── lead_lag.rs     # lead_compensator, lag_compensator (bilinear-transform design)
│   ├── pid_tune.rs     # FopdtModel PID tuning (Ziegler-Nichols, Cohen-Coon, SIMC), ziegler_nichols_ultimate
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
│   ├── line_search.rs  # backtracking_armijo + backtracking_armijo_dyn (internal helpers)
│   ├── bfgs.rs         # minimize_bfgs (BFGS quasi-Newton)
│   ├── gauss_newton.rs # least_squares_gn (QR-based Gauss-Newton)
│   ├── levenberg_marquardt.rs # least_squares_lm (damped normal equations)
│   ├── jacobian.rs     # finite_difference_jacobian, finite_difference_gradient
│   ├── dyn_optim.rs    # minimize_bfgs_dyn, least_squares_gn_dyn, least_squares_lm_dyn, finite_difference_*_dyn (requires `alloc`)
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
│   ├── rng.rs          # Rng (xoshiro256++) — sample()/sample_array() backing for all distributions
│   └── tests.rs        # comprehensive tests
├── quad/               # (requires `quad` feature)
│   ├── mod.rs          # gauss_legendre (N=1..10,15,20), adaptive_simpson, trapezoid, simpson (all no-alloc)
│   └── tests.rs        # comprehensive tests
├── fft/                # (requires `fft` feature)
│   ├── mod.rs          # FftError, cast helper, re-exports, module rustdoc ("not FFTW")
│   ├── twiddle.rs      # TwiddleTable<T, N> (fixed-size precomputed twiddles)
│   ├── radix.rs        # size-generic radix-2 core: bit_reverse, radix2_table, radix2_inline
│   ├── fixed.rs        # fixed-size no-alloc fft/ifft/fft_inplace/ifft_inplace
│   ├── real.rs         # rfft/irfft (fixed, half-size packing both ways) + DynRealFft/DynRealFftScratch (alloc)
│   ├── shift.rs        # fftshift/ifftshift (no-alloc, any element type)
│   ├── dynfft.rs       # DynFft planner + DynFftScratch (alloc): power-of-two SoA/SIMD + Bluestein; forward_with/inverse_with
│   ├── fft2.rs         # DynFft2 / DynRealFft2 2D FFT (transposed row pass, par-batched columns) + fftshift2d/ifftshift2d (alloc)
│   ├── bluestein.rs    # chirp-z transform for arbitrary/prime N on the SoA core (alloc)
│   ├── soa.rs          # SoaPlan (twiddles) / SoaScratch (re/im): fused len-2/4 pass, radix-4 stages (+ trailing radix-2), SIMD butterfly orchestration (alloc)
│   ├── convolve.rs     # fft_convolve/fft_correlate (1D) + fft_convolve2d/fft_correlate2d (alloc)
│   └── tests.rs        # comprehensive tests (vs naive DFT, round-trip, Parseval, SIMD==scalar)
└── quaternion.rs       # Quaternion rotations, SLERP, Euler, axis-angle
```

## Pre-Push / Release Checklist

Before merging to `main` (`main` is branch-protected — all changes must land via PR), user-facing
changes must be reflected in every documentation surface — otherwise the sources drift apart and
one of them becomes a lie. When adding/removing/renaming a public API, feature flag, module, or
behavior, update all of the following:

- [ ] **`src/lib.rs`** — crate-level rustdoc: per-module summary bullet, Cargo-features table row, any `pub use` re-exports
- [ ] **`README.md`** — Features bullet list, Cargo-features table, module `<details>` block, Module plan checkbox
- [ ] **`CHANGELOG.md`** — new version entry describing the change (match the style of recent entries; bump `Cargo.toml` version if not already bumped)
- [ ] **`docs/` (mkdocs)** — module page under `docs/` covering the new API, and `mkdocs.yml` nav if a new page was added; regenerate any demo plots in `docs/examples/` whose output changed
- [ ] **`CLAUDE.md`** — Module Plan, Cargo Features, and File Layout sections (keep module descriptions current, not frozen at the initial commit)

A PR that only updates source code without touching these surfaces is incomplete. The only
exception is internal-only changes (private helpers, test refactors, SIMD kernel swaps with no
API impact) — in that case, still consider whether the `CHANGELOG` deserves a line.

## Git Hooks

The repo ships a `pre-commit` hook in `.githooks/` that runs `cargo fmt` on the staged Rust files
and re-stages them, so unformatted code can never be committed (this mirrors the CI
`cargo fmt --check` gate). Enable it once per clone:

```bash
git config core.hooksPath .githooks
```

## Current Focus

Next candidates: SIMD extension to remaining linalg inner loops. The column-oriented
Householder dot products in QR / SVD / Hessenberg now go through `simd::dotc_dispatch`
(conjugated dot; forwards to `dot_dispatch` for real floats). Remaining: Cholesky inner
loops, and the row-oriented (right-side) Householder applications in SVD / Hessenberg
U/V/Q accumulation, which would need the reflector cached in a contiguous buffer first.
