# DynMatrix

`DynMatrix<T>` is a heap-allocated matrix with runtime dimensions — the dynamic counterpart to `Matrix<T, M, N>`. Requires the `alloc` feature (enabled by default via `std`).

## Storage Layout

`DynMatrix<T>` stores elements in a `Vec<T>` in **column-major** order: element `(row, col)` is at index `col * nrows + row`. This matches the fixed `Matrix` layout and enables the same SIMD inner-loop optimizations.

- `from_rows()` accepts **row-major** input (transposes internally)
- `from_slice()` accepts **column-major** input directly

## Constructors

```rust
use numeris::{DynMatrix, DynVector};

// From row-major data
let a = DynMatrix::from_rows(3, 3, &[
    2.0_f64, 1.0, -1.0,
    -3.0,   -1.0,  2.0,
    -2.0,    1.0,  2.0,
]);

// From column-major data
let b = DynMatrix::from_slice(2, 3, &[
    1.0_f64, 4.0,   // col 0
    2.0,     5.0,   // col 1
    3.0,     6.0,   // col 2
]);

// All zeros / all ones
let z = DynMatrix::<f64>::zeros(4, 4);
let o = DynMatrix::<f64>::ones(2, 5);

// Identity
let id = DynMatrix::<f64>::eye(3);

// From function f(row, col)
let m = DynMatrix::<f64>::from_fn(3, 3, |r, c| (r * 3 + c) as f64);

// Dynamic vector (enforces 1-column constraint)
let v = DynVector::from_slice(&[1.0_f64, 2.0, 3.0]);
```

## Indexing and Access

```rust
let a = DynMatrix::from_rows(2, 3, &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

// Element access
let v = a[(0, 2)];          // row 0, col 2 → 3.0

// Dimensions
let r = a.nrows();          // 2
let c = a.ncols();          // 3

// Mutable element
let mut b = a.clone();
b[(1, 1)] = 99.0;

// Dynamic vector single-index
let v = DynVector::from_slice(&[10.0_f64, 20.0, 30.0]);
let elem = v[1];            // 20.0
```

## Arithmetic

```rust
let a = DynMatrix::from_rows(2, 2, &[1.0_f64, 3.0, 2.0, 4.0]);
let b = DynMatrix::from_rows(2, 2, &[5.0_f64, 7.0, 6.0, 8.0]);

let c = &a + &b;            // element-wise add
let d = &a - &b;            // element-wise subtract
let e = &a * &b;            // matrix multiply
let f = -&a;                // negation

let g = &a * 2.0_f64;      // scalar multiply
let h = &a / 2.0_f64;      // scalar divide

let p = a.element_mul(&b);  // element-wise multiply
let q = a.element_div(&b);  // element-wise divide

// Transpose
let at = a.transpose();
```

## Mixed Operations with Fixed Matrix

`Matrix<T, M, N>` and `DynMatrix<T>` interoperate via `mixed_ops`:

```rust
use numeris::{Matrix, DynMatrix};

let fixed = Matrix::new([[1.0_f64, 2.0], [3.0, 4.0]]);
let dyn_  = DynMatrix::from_rows(2, 2, &[5.0_f64, 7.0, 6.0, 8.0]);

// All combinations produce DynMatrix
let r1: DynMatrix<f64> = &fixed * &dyn_;   // Matrix * DynMatrix
let r2: DynMatrix<f64> = &dyn_  * &fixed;  // DynMatrix * Matrix
let r3: DynMatrix<f64> = &fixed + &dyn_;   // Matrix + DynMatrix
let r4: DynMatrix<f64> = &dyn_  + &fixed;  // DynMatrix + Matrix
```

## Conversions

```rust
use numeris::{Matrix, DynMatrix};

// Fixed → Dynamic (infallible)
let fixed = Matrix::new([[1.0_f64, 2.0], [3.0, 4.0]]);
let dyn_: DynMatrix<f64> = fixed.into();

// Dynamic → Fixed (fallible — panics if dimensions don't match)
let back: Matrix<f64, 2, 2> = (&dyn_).try_into().unwrap();

// DynVector ↔ DynMatrix
use numeris::DynVector;
let v = DynVector::from_slice(&[1.0_f64, 2.0, 3.0]);
let m: DynMatrix<f64> = v.into();  // 3×1 matrix
```

## Norms and Utilities

```rust
let a = DynMatrix::from_rows(2, 2, &[3.0_f64, 4.0, 0.0, 0.0]);

let frob = a.frobenius_norm();
let inf  = a.norm_inf();
let one  = a.norm_one();
let s    = a.sum();
let abs_ = a.abs();
let m    = a.map(|x| x * x);

// Swap rows / columns
let mut b = a.clone();
b.swap_rows(0, 1);
b.swap_cols(0, 1);

// Dynamic vector norms
let v = DynVector::from_slice(&[3.0_f64, 4.0]);
let n  = v.norm();     // L2 = 5.0
let n1 = v.norm_l1();  // L1 = 7.0
let d  = v.dot(&v);    // dot product = 25.0
```

## Block Operations

```rust
let big = DynMatrix::from_rows(4, 4, &[
     1.0_f64,  2.0,  3.0,  4.0,
     5.0,  6.0,  7.0,  8.0,
     9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0,
]);

// Extract sub-block (row_start, col_start, nrows, ncols)
let sub = big.block(1, 1, 2, 2);  // [[6,7],[10,11]]

// Insert sub-block
let mut m = DynMatrix::<f64>::zeros(4, 4);
let patch = DynMatrix::from_rows(2, 2, &[9.0_f64, 8.0, 7.0, 6.0]);
m.set_block(1, 1, &patch);
```

## Square Operations

```rust
let a = DynMatrix::from_rows(3, 3, &[
    4.0_f64, 2.0, 0.0,
    2.0,     3.0, 1.0,
    0.0,     1.0, 2.0,
]);

let tr  = a.trace();
let det = a.det();
let dg  = a.diag();         // DynVector
let sym = a.is_symmetric();
```

## Linear Algebra

All decompositions work the same as on `Matrix` — the same free functions operate on both via `MatrixRef`/`MatrixMut` traits:

```rust
let a = DynMatrix::from_rows(3, 3, &[
    4.0_f64, 2.0, 0.0,
    2.0,     3.0, 1.0,
    0.0,     1.0, 2.0,
]);
let b = DynVector::from_slice(&[1.0_f64, 2.0, 3.0]);

let lu   = a.lu().unwrap();
let chol = a.cholesky().unwrap();
let qr   = a.qr().unwrap();
let svd  = a.svd().unwrap();
let eig  = a.eig_symmetric().unwrap();
let sch  = a.schur().unwrap();

let x   = a.solve(&b).unwrap();
let inv = a.inverse().unwrap();
let det = a.det();
```

See [Linear Algebra](linalg.md) for full decomposition details.

## Type Aliases

| Alias | Type |
|---|---|
| `DynMatrixf64` | `DynMatrix<f64>` |
| `DynMatrixf32` | `DynMatrix<f32>` |
| `DynVectorf64` | `DynVector<f64>` |
| `DynVectorf32` | `DynVector<f32>` |
| `DynMatrixi32` | `DynMatrix<i32>` |
| `DynMatrixi64` | `DynMatrix<i64>` |
| `DynMatrixu32` | `DynMatrix<u32>` |
| `DynMatrixu64` | `DynMatrix<u64>` |
| `DynMatrixz64` | `DynMatrix<Complex<f64>>` (requires `complex`) |
| `DynMatrixz32` | `DynMatrix<Complex<f32>>` (requires `complex`) |
