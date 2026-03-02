# Linear Algebra

numeris provides six matrix decompositions, each available as:

- **Free functions** operating in-place on `&mut impl MatrixMut<T>` — work on both `Matrix` and `DynMatrix`
- **Wrapper structs** with `solve()`, `inverse()`, `det()` convenience methods
- **Matrix convenience methods** — `a.lu()`, `a.cholesky()`, etc.

## Overview

| Decomposition | Struct | Dynamic wrapper | Use case |
|---|---|---|---|
| LU | `LuDecomposition` | `DynLu` | General square systems, det, inverse |
| Cholesky | `CholeskyDecomposition` | `DynCholesky` | SPD / Hermitian PD systems |
| QR | `QrDecomposition` | `DynQr` | Least-squares, orthogonal basis |
| SVD | `SvdDecomposition` | `DynSvd` | Rank, condition number, pseudoinverse |
| Symmetric Eigen | `SymmetricEigen` | `DynSymmetricEigen` | Real eigenvalues, eigenvectors |
| Schur | `SchurDecomposition` | `DynSchur` | General eigenvalues (complex pairs) |

## LU Decomposition

Partial pivoting LU: `P A = L U`.

```rust
use numeris::{Matrix, Vector};

let a = Matrix::new([
    [2.0_f64, 1.0, -1.0],
    [-3.0,   -1.0,  2.0],
    [-2.0,    1.0,  2.0],
]);
let b = Vector::from_array([8.0, -11.0, -3.0]);

let lu = a.lu().unwrap();

// Solve Ax = b
let x   = lu.solve(&b);       // x = [2, 3, -1]

// Other operations
let inv = lu.inverse();        // A^{-1}
let det = lu.det();            // determinant

// Direct solve on Matrix (uses LU internally)
let x2 = a.solve(&b).unwrap();
```

!!! note "Small matrices"
    For N ≤ 4, `inverse()` and `det()` use direct closed-form formulas (adjugate method), bypassing LU entirely. This is 2–3x faster than the general path.

## Cholesky Decomposition

For symmetric positive-definite (SPD) matrices: `A = L Lᴴ`.

Works with both real and complex (Hermitian positive-definite) matrices.

```rust
use numeris::Matrix;

let a = Matrix::new([
    [4.0_f64, 2.0],
    [2.0,     3.0],
]);

let chol = a.cholesky().unwrap();

// Solve Ax = b (forward + back substitution, no LU needed)
let b = numeris::Vector::from_array([1.0_f64, 2.0]);
let x = chol.solve(&b);

// Other operations
let inv    = chol.inverse();
let det    = chol.det();
let ln_det = chol.ln_det();    // log-determinant (stable for large matrices)
```

## QR Decomposition

Householder QR: `A = Q R`.

```rust
use numeris::{Matrix, Vector};

// Square system — exact solve
let a = Matrix::new([[2.0_f64, 1.0], [1.0, 3.0]]);
let b = Vector::from_array([5.0_f64, 10.0]);
let qr = a.qr().unwrap();
let x  = qr.solve(&b);

// Overdetermined system — least-squares
let a_rect = Matrix::new([
    [1.0_f64, 0.0],
    [1.0,     1.0],
    [1.0,     2.0],
]);
let b_rect = Vector::from_array([1.0_f64, 2.0, 4.0]);
let x_ls = a_rect.qr().unwrap().solve(&b_rect);

// Determinant
let det = a.qr().unwrap().det();
```

## SVD

Golub-Kahan implicit-shift QR on a bidiagonal form.

`A = U Σ Vᵀ`, where `A` is `M × N` with `M ≥ N`.

```rust
use numeris::Matrix;

let a = Matrix::new([
    [1.0_f64, 2.0],
    [3.0,     4.0],
    [5.0,     6.0],
]);

let svd = a.svd().unwrap();

// Components
let sigma = svd.singular_values();   // [σ₀, σ₁] sorted descending
let u = svd.u();                      // 3×3 orthogonal (or 3×2 thin)
let vt = svd.vt();                    // 2×2 orthogonal (Vᵀ)

// Rank and condition number
let rank = svd.rank(1e-10);          // rank with tolerance
let cond = svd.condition_number();   // σ_max / σ_min

// Shortcut: only singular values (faster, no U or V)
let sigma_only = a.singular_values_only().unwrap();
```

!!! info "M < N matrices"
    `DynSvd` handles `M < N` by transposing internally. The fixed `SvdDecomposition` requires `M ≥ N` at compile time.

## Symmetric Eigendecomposition

Householder tridiagonalization + implicit QR with Wilkinson shift.

For real symmetric (or Hermitian) matrices only. Produces real eigenvalues.

```rust
use numeris::Matrix;

let a = Matrix::new([
    [4.0_f64, 1.0, 0.0],
    [1.0,     3.0, 1.0],
    [0.0,     1.0, 2.0],
]);

let eig = a.eig_symmetric().unwrap();

let vals = eig.eigenvalues();    // sorted ascending
let vecs = eig.eigenvectors();   // columns = orthonormal eigenvectors
// Reconstruction: A ≈ vecs * diag(vals) * vecs^T

// Shortcut: eigenvalues only (faster, no eigenvectors)
let vals_only = a.eigenvalues_symmetric().unwrap();
```

## Schur Decomposition

Francis double-shift QR on the upper Hessenberg form.

Produces quasi-upper-triangular `S` with 1×1 (real eigenvalue) and 2×2 (conjugate complex pair) diagonal blocks, and orthogonal `Q` such that `A = Q S Qᵀ`.

```rust
use numeris::Matrix;

// 90° rotation — eigenvalues ±i
let a = Matrix::new([[0.0_f64, -1.0], [1.0, 0.0]]);

let schur = a.schur().unwrap();
let s = schur.s();   // quasi-upper-triangular
let q = schur.q();   // orthogonal factor

// Extract eigenvalues (real and imaginary parts)
let (re, im) = a.eigenvalues().unwrap();
// re = [0, 0], im = [1, -1]  (conjugate pair ±i)
```

## DynMatrix Variants

All six decompositions have dynamic wrappers in `DynMatrix`:

```rust
use numeris::{DynMatrix, DynVector};

let a = DynMatrix::from_rows(3, 3, &[
    4.0_f64, 2.0, 0.0,
    2.0,     3.0, 1.0,
    0.0,     1.0, 2.0,
]);
let b = DynVector::from_slice(&[6.0_f64, 7.0, 3.0]);

let lu   = a.lu().unwrap();
let chol = a.cholesky().unwrap();
let qr   = a.qr().unwrap();
let svd  = a.svd().unwrap();
let eig  = a.eig_symmetric().unwrap();
let sch  = a.schur().unwrap();

let x   = a.solve(&b).unwrap();
let inv = a.inverse().unwrap();
```

## Complex Matrices

Enable the `complex` feature to use decompositions with complex elements:

```toml
numeris = { version = "0.2", features = ["complex"] }
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

// Hermitian positive-definite Cholesky (A = L L^H)
let hpd = Matrix::new([
    [C::new(4.0, 0.0), C::new(2.0,  1.0)],
    [C::new(2.0,-1.0), C::new(5.0,  0.0)],
]);
let chol = hpd.cholesky().unwrap();
```

Complex overhead is zero for real code paths — `conj()` and `re()` on `f64` are identity functions, fully inlined and erased by the compiler.

## Error Handling

All fallible operations return `Result<_, LinalgError>`:

```rust
use numeris::linalg::LinalgError;

match a.cholesky() {
    Ok(chol) => { /* use chol */ }
    Err(LinalgError::NotPositiveDefinite) => { /* not SPD */ }
    Err(LinalgError::SingularMatrix)      => { /* singular */ }
    Err(LinalgError::ConvergenceFailure)  => { /* QR iteration didn't converge */ }
    Err(e) => { /* other */ }
}
```
