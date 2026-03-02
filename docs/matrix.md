# Matrix

`Matrix<T, M, N>` is the core fixed-size matrix type — stack-allocated, const-generic, column-major storage.

## Storage Layout

```
Matrix<T, M, N>  →  [[T; M]; N]   (N columns, each of length M)
```

Element `(row, col)` is at `data[col][row]`. This is **column-major** (Fortran / LAPACK order), which makes column-oriented inner loops — AXPY, Householder reflections, dot products — operate on contiguous memory.

`Matrix::new()` accepts **row-major** input (as you'd write on paper) and transposes internally:

```rust
use numeris::Matrix;

// Written as rows, stored column-major
let a: Matrix<f64, 2, 3> = Matrix::new([
    [1.0, 2.0, 3.0],   // row 0
    [4.0, 5.0, 6.0],   // row 1
]);
assert_eq!(a[(0, 2)], 3.0);  // row 0, col 2
assert_eq!(a[(1, 0)], 4.0);  // row 1, col 0
```

## Constructors

```rust
use numeris::{Matrix, Matrix3};

// From row-major nested array
let a = Matrix::new([[1.0_f64, 2.0], [3.0, 4.0]]);

// All zeros / all ones / identity
let z = Matrix::<f64, 3, 3>::zeros();
let o = Matrix::<f64, 2, 4>::ones();
let id = Matrix3::<f64>::eye();

// From function f(row, col)
let m = Matrix::<f64, 3, 3>::from_fn(|r, c| (r * 3 + c) as f64);

// From column-major slice
let data = [1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]; // col0=[1,4], col1=[2,5], col2=[3,6]
let m = Matrix::<f64, 2, 3>::from_col_slice(&data);
```

## Indexing

```rust
let a = Matrix::new([[1.0_f64, 2.0], [3.0, 4.0]]);

// Element access
let v = a[(0, 1)];        // row 0, col 1 → 2.0

// Row / column access
let row = a.row(0);       // Matrix<f64, 1, 2>
let col = a.col(1);       // Matrix<f64, 2, 1>

// Mutable element
let mut b = a;
b[(1, 0)] = 99.0;
```

## Arithmetic

```rust
use numeris::{Matrix, Vector};

let a = Matrix::new([[1.0_f64, 2.0], [3.0, 4.0]]);
let b = Matrix::new([[5.0_f64, 6.0], [7.0, 8.0]]);

// Matrix operations
let c = a + b;              // element-wise add
let d = a - b;              // element-wise subtract
let e = a * b;              // matrix multiply (2×2 * 2×2)
let f = -a;                 // negation

// Scalar operations
let g = a * 2.0;
let h = a / 2.0;

// Matrix-vector multiply
let v = Vector::from_array([1.0_f64, 0.0]);
let w = a.vecmul(&v);       // a * v → Vector<f64, 2>

// Transpose
let at = a.transpose();     // Matrix<f64, 2, 2>

// Element-wise multiply / divide
let p = a.element_mul(&b);
let q = a.element_div(&b);
```

## Square Matrix Operations

```rust
use numeris::Matrix3;

let m = Matrix3::new([[4.0_f64, 2.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 2.0]]);

let tr = m.trace();          // sum of diagonal = 9.0
let d  = m.det();            // determinant (via LU for N>4, direct formula for N≤4)
let dg = m.diag();           // diagonal as Vector<f64, 3>
let p2 = m.pow(2);           // matrix power (integer exponent)
let sym = m.is_symmetric();  // true/false

// Construct diagonal matrix
let v = numeris::Vector::from_array([1.0_f64, 2.0, 3.0]);
let d = Matrix3::from_diag(&v);
```

## Norms

```rust
let a = Matrix::new([[3.0_f64, 0.0], [4.0, 0.0]]);

let frob = a.frobenius_norm();  // √(9+16) = 5.0
let inf  = a.norm_inf();        // max row sum = 7.0
let one  = a.norm_one();        // max col sum = 7.0
```

## Vectors

`Vector<T, N>` is a row vector — a type alias for `Matrix<T, 1, N>`.
`ColumnVector<T, N>` is a column vector — a type alias for `Matrix<T, N, 1>`.

```rust
use numeris::{Vector, ColumnVector, Vector3};

let v = Vector::from_array([3.0_f64, 4.0, 0.0]);
let u = Vector::from_array([1.0_f64, 0.0, 0.0]);

let d   = v.dot(&u);           // 3.0
let n   = v.norm();            // 5.0
let n1  = v.norm_l1();         // 7.0
let hat = v.normalize();       // unit vector

let cross = v.cross(&u);       // 3D cross product → Vector3<f64>
let outer = v.outer(&u);       // outer product → Matrix<f64, 3, 3>

// Column vector construction
let cv = ColumnVector::from_column([1.0_f64, 2.0, 3.0]);
```

## Block Operations

```rust
use numeris::{Matrix, Matrix3};

let big = Matrix3::new([
    [1.0_f64, 2.0, 3.0],
    [4.0,     5.0, 6.0],
    [7.0,     8.0, 9.0],
]);

// Extract sub-block starting at (row_start, col_start), size M×N
let sub: Matrix<f64, 2, 2> = big.block::<2, 2>(0, 0); // [[1,2],[4,5]]

// Extract top-left / top-right / bottom-left / bottom-right corners
let tl: Matrix<f64, 2, 2> = big.top_left::<2, 2>();
let tr: Matrix<f64, 2, 1> = big.top_right::<2, 1>();

// Head / tail for vectors
use numeris::Vector;
let v = Vector::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0]);
let h: numeris::Vector<f64, 3> = v.head::<3>();   // [1, 2, 3]
let t: numeris::Vector<f64, 2> = v.tail::<2>();   // [4, 5]

// Insert sub-block
let mut m = Matrix3::<f64>::zeros();
let patch = Matrix::<f64, 2, 2>::new([[9.0, 8.0], [7.0, 6.0]]);
m.set_block(1, 1, &patch);
```

## Size Aliases

Square matrices and row vectors up to 6×6:

| Square | Rectangular (examples) | Row vectors | Column vectors |
|---|---|---|---|
| `Matrix1<T>` | `Matrix2x3<T>` | `Vector1<T>` | `ColumnVector1<T>` |
| `Matrix2<T>` | `Matrix3x4<T>` | `Vector2<T>` | `ColumnVector2<T>` |
| `Matrix3<T>` | `Matrix4x6<T>` | `Vector3<T>` | `ColumnVector3<T>` |
| … up to `Matrix6<T>` | All M×N for M,N ∈ 1..=6, M≠N | … `Vector6<T>` | … `ColumnVector6<T>` |

```rust
use numeris::{Matrix3, Matrix4x3, Vector3, ColumnVector3};

let rot: Matrix3<f64>     = Matrix3::eye();
let pts: Matrix4x3<f64>   = Matrix4x3::zeros();   // 4 rows, 3 cols
let v:   Vector3<f64>     = Vector3::from_array([1.0, 2.0, 3.0]);
let cv:  ColumnVector3<f64> = ColumnVector3::from_column([4.0, 5.0, 6.0]);
```

## Utilities

```rust
let a = Matrix::new([[1.0_f64, -2.0], [-3.0, 4.0]]);

let s = a.sum();        // sum of all elements = 0.0
let b = a.abs();        // element-wise abs → [[1,2],[3,4]]
let c = a.map(|x| x * x);  // element-wise map

// Row/column swap
let mut m = Matrix3::<f64>::eye();
m.swap_rows(0, 2);
m.swap_cols(0, 1);

// As flat slice (column-major order)
let s: &[f64] = a.as_slice();
```

## Iteration

```rust
let a = Matrix::new([[1.0_f64, 2.0], [3.0, 4.0]]);

for val in a.iter() {
    // visits in column-major order: 1.0, 3.0, 2.0, 4.0
}

for val in a {    // IntoIterator, consumes a
    // same order
}
```

## Linear Algebra

Linear algebra operations are documented on the [Linalg](linalg.md) page. Convenience methods available directly on `Matrix`:

```rust
let a = Matrix::new([[4.0_f64, 2.0], [2.0, 3.0]]);

let lu   = a.lu().unwrap();
let chol = a.cholesky().unwrap();
let qr   = a.qr().unwrap();
let svd  = a.svd().unwrap();
let eig  = a.eig_symmetric().unwrap();
let sch  = a.schur().unwrap();

let inv  = a.inverse().unwrap();
let det  = a.det();
let (re, im) = a.eigenvalues().unwrap();
```
