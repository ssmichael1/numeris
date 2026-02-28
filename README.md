# numeris

Pure-Rust numerical algorithms library, no-std compatible. Similar in scope to SciPy, suitable for embedded targets (no heap allocation, no FPU assumptions).

## Features

- **Fixed-size matrices** — stack-allocated, const-generic `Matrix<T, M, N>`
- **Linear algebra** — LU, Cholesky, and QR decompositions with solve, inverse, and determinant
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
| `std` | yes | Uses hardware FPU via system libm. Full speed on desktop/server. |
| `libm` | baseline | Pure-Rust software float math. Always available as fallback. |
| `complex` | no | Adds `Complex<f32>` / `Complex<f64>` support via `num-complex`. |

```bash
# Default (std)
cargo build

# No-std for embedded
cargo build --no-default-features --features libm

# With complex number support
cargo build --features complex
```

## Module overview

### `matrix` — Fixed-size matrix

`Matrix<T, M, N>` with `[[T; N]; M]` row-major storage.

- Arithmetic: `+`, `-`, `*` (matrix and scalar), negation, element-wise multiply
- Indexing: `m[(i, j)]`, row/column access, block extraction and insertion
- Square: `trace`, `det`, `diag`, `from_diag`, `pow`, `is_symmetric`, `eye`
- Norms: `frobenius_norm`, `norm_inf`, `norm_one`
- Utilities: `transpose`, `from_fn`, `map`, `swap_rows`, `swap_cols`, `sum`, `abs`
- Iteration: `iter()`, `iter_mut()`, `as_slice()`, `IntoIterator`

Vectors are type aliases: `Vector<T, N>` = `Matrix<T, 1, N>` (row vector), `ColumnVector<T, N>` = `Matrix<T, N, 1>`.

Vector-specific: `dot`, `cross`, `outer`, `norm`, `norm_l1`, `normalize`.

### `linalg` — Decompositions

All decompositions provide both free functions (operating on `&mut impl MatrixMut<T>`) and wrapper structs with `solve()`, `inverse()`, `det()`.

| Decomposition | Free function | Struct | Notes |
|---|---|---|---|
| LU | `lu_in_place` | `LuDecomposition` | Partial pivoting |
| Cholesky | `cholesky_in_place` | `CholeskyDecomposition` | A = LL^H (Hermitian) |
| QR | `qr_in_place` | `QrDecomposition` | Householder reflections, least-squares |

Convenience methods on `Matrix`: `a.lu()`, `a.cholesky()`, `a.qr()`, `a.solve(&b)`, `a.inverse()`, `a.det()`.

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

## Design decisions

- **Stack-allocated**: `[[T; N]; M]` storage, no heap. Dimensions are const generics.
- **`num-traits`**: Generic numeric bounds with `default-features = false`.
- **In-place algorithms**: Decompositions operate on `&mut impl MatrixMut<T>`, avoiding allocator/storage trait complexity.
- **Integer matrices**: Work with `Scalar` (all basic ops). Float-only operations (`det`, norms, decompositions) require `LinalgScalar` or `FloatScalar`.
- **Complex support**: Additive, behind a feature flag. Zero cost for real-only usage.

## License

MIT
