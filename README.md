# numeris

Pure-Rust numerical algorithms library, no-std compatible. Similar in scope to SciPy, suitable for embedded targets (no heap allocation, no FPU assumptions).

## Features

- **Fixed-size matrices** — stack-allocated, const-generic `Matrix<T, M, N>`
- **Dynamic matrices** — heap-allocated `DynMatrix<T>` with runtime dimensions (optional `alloc` feature)
- **Linear algebra** — LU, Cholesky, and QR decompositions with solve, inverse, and determinant
- **ODE integration** — fixed-step RK4 and 7 adaptive Runge-Kutta solvers with dense output
- **Optimization** — root finding (Brent, Newton), BFGS minimization, Gauss-Newton and Levenberg-Marquardt least squares
- **Complex number support** — all decompositions work with `Complex<f32>` / `Complex<f64>` (optional feature)
- **Quaternions** — unit quaternion rotations, SLERP, Euler angles, rotation matrices
- **Norms** — L1, L2, Frobenius, infinity, one norms
- **No-std / embedded** — runs without `std` or heap; float math falls back to software `libm`

## Quick start

```toml
[dependencies]
numeris = "0.1"
```

```rust
use numeris::{Matrix, Vector};

// Solve a linear system Ax = b
let a = Matrix::new([
    [2.0_f64, 1.0, -1.0],
    [-3.0, -1.0, 2.0],
    [-2.0, 1.0, 2.0],
]);
let b = Vector::from_array([8.0, -11.0, -3.0]);
let x = a.solve(&b).unwrap(); // x = [2, 3, -1]

// Cholesky decomposition of a symmetric positive-definite matrix
let spd = Matrix::new([[4.0, 2.0], [2.0, 3.0]]);
let chol = spd.cholesky().unwrap();
let inv = chol.inverse();

// QR decomposition and least-squares
let a = Matrix::new([
    [1.0_f64, 0.0],
    [1.0, 1.0],
    [1.0, 2.0],
]);
let b = Vector::from_array([1.0, 2.0, 4.0]);
let x = a.qr().unwrap().solve(&b); // least-squares fit

// Quaternion rotation
use numeris::Quaternion;
let q = Quaternion::from_axis_angle(
    &Vector::from_array([0.0, 0.0, 1.0]),
    std::f64::consts::FRAC_PI_2,
);
let v = Vector::from_array([1.0, 0.0, 0.0]);
let rotated = q * v; // [0, 1, 0]
```

## Dynamic matrices

When dimensions aren't known at compile time, use `DynMatrix` (requires `alloc`, included with default `std` feature):

```rust
use numeris::{DynMatrix, DynVector};

// Runtime-sized matrix
let a = DynMatrix::from_slice(3, 3, &[
    2.0_f64, 1.0, -1.0,
    -3.0, -1.0, 2.0,
    -2.0, 1.0, 2.0,
]);
let b = DynVector::from_slice(&[8.0, -11.0, -3.0]);
let x = a.solve(&b).unwrap();

// Convert between fixed and dynamic
use numeris::Matrix;
let fixed = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
let dynamic: DynMatrix<f64> = fixed.into();
let back: Matrix<f64, 2, 2> = (&dynamic).try_into().unwrap();

// Mixed arithmetic
let result = fixed * &dynamic; // Matrix * DynMatrix → DynMatrix
```

Type aliases: `DynMatrixf64`, `DynMatrixf32`, `DynVectorf64`, `DynVectorf32`, `DynMatrixz64` (complex, requires `complex` feature), etc.

## ODE integration

Fixed-step RK4 and 7 adaptive Runge-Kutta solvers with embedded error estimation and dense output:

```rust
use numeris::ode::{RKAdaptive, RKTS54, AdaptiveSettings};
use numeris::Vector;

// Harmonic oscillator: y'' = -y
let y0 = Vector::from_array([1.0_f64, 0.0]);
let tau = 2.0 * std::f64::consts::PI;
let sol = RKTS54::integrate(
    0.0, tau, &y0,
    |_t, y| Vector::from_array([y[1], -y[0]]),
    &AdaptiveSettings::default(),
).unwrap();
assert!((sol.y[0] - 1.0).abs() < 1e-6); // cos(2π) ≈ 1
```

| Solver | Stages | Order | FSAL | Interpolant |
|---|---|---|---|---|
| `RKF45` | 6 | 5(4) | no | — |
| `RKTS54` | 7 | 5(4) | yes | 4th degree |
| `RKV65` | 10 | 6(5) | no | 6th degree |
| `RKV87` | 17 | 8(7) | no | 7th degree |
| `RKV98` | 21 | 9(8) | no | 8th degree |
| `RKV98NoInterp` | 16 | 9(8) | no | — |
| `RKV98Efficient` | 26 | 9(8) | no | 9th degree |

## Optimization

Root finding, unconstrained minimization, and nonlinear least squares (requires `optim` feature):

```toml
[dependencies]
numeris = { version = "0.1", features = ["optim"] }
```

```rust
use numeris::optim::{brent, minimize_bfgs, least_squares_lm, RootSettings, BfgsSettings, LmSettings};
use numeris::{Matrix, Vector};

// Root finding: solve x² - 2 = 0
let root = brent(|x| x * x - 2.0, 0.0, 2.0, &RootSettings::default()).unwrap();
assert!((root.x - std::f64::consts::SQRT_2).abs() < 1e-12);

// BFGS minimization: minimize (x-1)² + (y-2)²
let min = minimize_bfgs(
    |x: &Vector<f64, 2>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2),
    |x: &Vector<f64, 2>| Vector::from_array([2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)]),
    &Vector::from_array([0.0, 0.0]),
    &BfgsSettings::default(),
).unwrap();
assert!((min.x[0] - 1.0).abs() < 1e-6);

// Levenberg-Marquardt: fit y = a * exp(b * x)
let t = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
let y = [2.0, 2.7, 3.65, 4.95, 6.7];
let fit = least_squares_lm(
    |x: &Vector<f64, 2>| {
        let mut r = Vector::<f64, 5>::zeros();
        for i in 0..5 { r[i] = x[0] * (x[1] * t[i]).exp() - y[i]; }
        r
    },
    |x: &Vector<f64, 2>| {
        let mut j = Matrix::<f64, 5, 2>::zeros();
        for i in 0..5 {
            let e = (x[1] * t[i]).exp();
            j[(i, 0)] = e;
            j[(i, 1)] = x[0] * t[i] * e;
        }
        j
    },
    &Vector::from_array([1.0, 0.1]),
    &LmSettings::default(),
).unwrap();
assert!(fit.cost < 0.1);
```

| Algorithm | Function | Use case |
|---|---|---|
| Brent's method | `brent` | Bracketed scalar root finding |
| Newton's method | `newton_1d` | Scalar root finding with derivative |
| BFGS | `minimize_bfgs` | Unconstrained minimization |
| Gauss-Newton | `least_squares_gn` | Nonlinear least squares (QR-based) |
| Levenberg-Marquardt | `least_squares_lm` | Nonlinear least squares (damped) |

Finite-difference utilities: `finite_difference_gradient` and `finite_difference_jacobian` for when analytical derivatives aren't available.

## Complex matrices

Enable the `complex` feature to use decompositions with complex elements:

```toml
[dependencies]
numeris = { version = "0.1", features = ["complex"] }
```

```rust
use numeris::{Complex, Matrix, Vector};

type C = Complex<f64>;

let a = Matrix::new([
    [C::new(2.0, 1.0), C::new(1.0, -1.0)],
    [C::new(1.0, 0.0), C::new(3.0, 2.0)],
]);
let b = Vector::from_array([C::new(5.0, 3.0), C::new(7.0, 4.0)]);
let x = a.solve(&b).unwrap();

// Hermitian positive-definite Cholesky (A = L * L^H)
let hpd = Matrix::new([
    [C::new(4.0, 0.0), C::new(2.0, 1.0)],
    [C::new(2.0, -1.0), C::new(5.0, 0.0)],
]);
let chol = hpd.cholesky().unwrap();
```

Complex support adds zero overhead to real-valued code paths. The `LinalgScalar` trait methods (`modulus`, `conj`, etc.) are `#[inline]` identity functions for `f32`/`f64`, fully erased by the compiler.

## Cargo features

| Feature | Default | Description |
|---|---|---|
| `std` | yes | Implies `alloc`. Uses hardware FPU via system libm. |
| `alloc` | via `std` | Enables `DynMatrix` / `DynVector` (heap-allocated, runtime-sized). |
| `ode` | yes | ODE integration (RK4, adaptive solvers). |
| `optim` | no | Optimization (root finding, BFGS, Gauss-Newton, LM). |
| `libm` | baseline | Pure-Rust software float math. Always available as fallback. |
| `complex` | no | Adds `Complex<f32>` / `Complex<f64>` support via `num-complex`. |
| `all` | no | All features: `std` + `ode` + `optim` + `complex`. |

```bash
# Default (std + ode)
cargo build

# All features
cargo build --features all

# No-std for embedded
cargo build --no-default-features --features libm

# No-std with dynamic matrices
cargo build --no-default-features --features "libm,alloc"

# With optimization and complex support
cargo build --features "optim,complex"
```

## Module overview

### `matrix` — Fixed-size matrix

`Matrix<T, M, N>` with `[[T; N]; M]` row-major storage.

- Arithmetic: `+`, `-`, `*` (matrix and scalar), negation, element-wise multiply/divide
- Indexing: `m[(i, j)]`, row/column access, block extraction and insertion
- Square: `trace`, `det`, `diag`, `from_diag`, `pow`, `is_symmetric`, `eye`
- Norms: `frobenius_norm`, `norm_inf`, `norm_one`
- Utilities: `transpose`, `from_fn`, `map`, `swap_rows`, `swap_cols`, `sum`, `abs`
- Iteration: `iter()`, `iter_mut()`, `as_slice()`, `IntoIterator`

Vectors are type aliases: `Vector<T, N>` = `Matrix<T, 1, N>` (row vector), `ColumnVector<T, N>` = `Matrix<T, N, 1>`.

Vector-specific: `dot`, `cross`, `outer`, `norm`, `norm_l1`, `normalize`.

#### Size aliases

Convenience aliases are provided for common sizes (all re-exported from the crate root):

| Square | Rectangular (examples) | Vectors |
|---|---|---|
| `Matrix1` .. `Matrix6` | `Matrix2x3`, `Matrix3x4`, `Matrix4x6`, ... | `Vector1` .. `Vector6` |
| | All M×N combinations for M,N ∈ 1..6, M≠N | `ColumnVector1` .. `ColumnVector6` |

```rust
use numeris::{Matrix3, Matrix4x3, Vector3};

let rotation: Matrix3<f64> = Matrix3::eye();
let points: Matrix4x3<f64> = Matrix4x3::zeros(); // 4 rows, 3 cols
let v: Vector3<f64> = Vector3::from_array([1.0, 2.0, 3.0]);
```

### `dynmatrix` — Dynamic matrix (requires `alloc`)

`DynMatrix<T>` with `Vec<T>` row-major storage and runtime dimensions.

- Same arithmetic, norms, block ops, and utilities as fixed `Matrix`
- Mixed ops: `Matrix * DynMatrix`, `DynMatrix + Matrix`, etc. → `DynMatrix`
- `DynVector<T>` newtype with single-index access and `dot`
- Conversions: `From<Matrix>` → `DynMatrix`, `TryFrom<&DynMatrix>` → `Matrix`
- Full linalg: `DynLu`, `DynCholesky`, `DynQr` wrappers + convenience methods

Type aliases: `DynMatrixf64`, `DynMatrixf32`, `DynVectorf64`, `DynVectorf32`, `DynMatrixi32`, `DynMatrixi64`, `DynMatrixu32`, `DynMatrixu64`, `DynMatrixz64`, `DynMatrixz32` (complex).

### `linalg` — Decompositions

All decompositions provide both free functions (operating on `&mut impl MatrixMut<T>`) and wrapper structs with `solve()`, `inverse()`, `det()`.

| Decomposition | Free function | Fixed struct | Dynamic struct | Notes |
|---|---|---|---|---|
| LU | `lu_in_place` | `LuDecomposition` | `DynLu` | Partial pivoting |
| Cholesky | `cholesky_in_place` | `CholeskyDecomposition` | `DynCholesky` | A = LL^H (Hermitian) |
| QR | `qr_in_place` | `QrDecomposition` | `DynQr` | Householder reflections, least-squares |

Convenience methods on `Matrix`: `a.lu()`, `a.cholesky()`, `a.qr()`, `a.solve(&b)`, `a.inverse()`, `a.det()`.

Same convenience methods on `DynMatrix`: `a.lu()`, `a.cholesky()`, `a.qr()`, `a.solve(&b)`, `a.inverse()`, `a.det()`.

### `ode` — ODE integration

Fixed-step `rk4` / `rk4_step` and 7 adaptive Runge-Kutta solvers via the `RKAdaptive` trait. PI step-size controller (Söderlind & Wang 2006). Dense output / interpolation available for most solvers (gated behind `std`).

### `optim` — Optimization (requires `optim` feature)

- **Root finding**: `brent` (bracketed, superlinear convergence), `newton_1d` (with derivative)
- **Minimization**: `minimize_bfgs` (BFGS quasi-Newton with Armijo line search)
- **Least squares**: `least_squares_gn` (Gauss-Newton via QR), `least_squares_lm` (Levenberg-Marquardt via damped normal equations)
- **Finite differences**: `finite_difference_gradient`, `finite_difference_jacobian` for numerical derivatives
- All algorithms use `FloatScalar` bound (real-valued), configurable via settings structs with `Default` impls for `f32` and `f64`

### `quaternion` — Unit quaternion rotations

`Quaternion<T>` with scalar-first convention `[w, x, y, z]`.

- Construction: `new`, `identity`, `from_axis_angle`, `from_euler`, `from_rotation_matrix`, `rotx`, `roty`, `rotz`
- Operations: `*` (Hamilton product), `* Vector3` (rotation), `conjugate`, `inverse`, `normalize`, `slerp`
- Conversion: `to_rotation_matrix`, `to_axis_angle`, `to_euler`

### `traits` — Element traits

| Trait | Bounds | Used by |
|---|---|---|
| `Scalar` | `Copy + PartialEq + Debug + Zero + One + Num` | All matrix ops |
| `FloatScalar` | `Scalar + Float + LinalgScalar<Real=Self>` | Quaternions, ordered comparisons |
| `LinalgScalar` | `Scalar` + `modulus`, `conj`, `re`, `lsqrt`, `lln`, `from_real` | Decompositions, norms |
| `MatrixRef<T>` | Read-only `get(row, col)` | Generic algorithms |
| `MatrixMut<T>` | Adds `get_mut(row, col)` | In-place decompositions |

## Module plan

Checked items are implemented; unchecked are potential future work.

- [x] **matrix** — Fixed-size matrix (stack-allocated, const-generic dimensions), size aliases up to 6×6
- [x] **linalg** — LU, Cholesky, QR decompositions; solvers, inverse, determinant; complex support
- [x] **quaternion** — Unit quaternion for rotations (SLERP, Euler, axis-angle, rotation matrices)
- [x] **ode** — ODE integration (RK4, 7 adaptive solvers, dense output)
- [x] **dynmatrix** — Heap-allocated runtime-sized matrix/vector (`alloc` feature)
- [ ] **interp** — Interpolation (linear, cubic spline, Hermite)
- [x] **optim** — Optimization (Brent, Newton, BFGS, Gauss-Newton, Levenberg-Marquardt)
- [ ] **quad** — Numerical quadrature / integration
- [ ] **fft** — Fast Fourier Transform
- [ ] **special** — Special functions (Bessel, gamma, erf, etc.)
- [ ] **stats** — Statistics and distributions
- [ ] **poly** — Polynomial operations and root-finding

## Design decisions

- **Stack-allocated**: `[[T; N]; M]` storage, no heap. Dimensions are const generics.
- **Heap-allocated**: `DynMatrix` uses `Vec<T>` for runtime dimensions, behind `alloc` feature.
- **`num-traits`**: Generic numeric bounds with `default-features = false`.
- **In-place algorithms**: Decompositions operate on `&mut impl MatrixMut<T>`, avoiding allocator/storage trait complexity. Both `Matrix` and `DynMatrix` implement `MatrixMut`, so the same free functions work for both.
- **Integer matrices**: Work with `Scalar` (all basic ops). Float-only operations (`det`, norms, decompositions) require `LinalgScalar` or `FloatScalar`.
- **Complex support**: Additive, behind a feature flag. Zero cost for real-only usage.

## License

MIT
