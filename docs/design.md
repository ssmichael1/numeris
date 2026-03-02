# Design

Key architectural decisions in numeris.

## Column-Major Storage

`Matrix<T, M, N>` stores data as `[[T; M]; N]` — N columns of M rows. Element `(row, col)` is at `data[col][row]`.

This is **column-major** (Fortran / LAPACK order). The reasons:

1. **LAPACK compatibility**: all standard numerical algorithms (LU, QR, Cholesky, etc.) are formulated and optimized for column-major storage.
2. **SIMD inner loops**: Householder reflections, LU elimination, and BLAS-1 AXPY (`y += α·x`) operate on a single column — a contiguous block of M floats. `col_as_slice()` returns this directly, enabling SIMD dispatch with no gathering.
3. **Matmul**: the j-k-i loop order accesses A in column-major (contiguous i-access) and B by row (B[k,j] strides), matching the memory layout for the SIMD-vectorized i-loop.

`Matrix::new()` accepts **row-major** input (as written on paper) and transposes internally, so the API is natural. `DynMatrix::from_rows()` does the same.

## Stack Allocation and Const Generics

Fixed-size matrices are fully stack-allocated. There is no heap usage, no `Box`, no `Vec` — just `[[T; M]; N]` on the stack. This enables:

- **No-std / embedded**: works on targets with no heap allocator.
- **Zero-cost abstractions**: the compiler can see all dimensions at monomorphization time. Dead branches are eliminated. SIMD kernel dispatch via `TypeId` is a compile-time decision.
- **LLVM optimization**: the compiler knows the loop bounds and can unroll, vectorize, and schedule optimally.

Avoids `[T; M*N]` flat storage (which would require unstable `generic_const_exprs` for expressions like `const LEN: usize = M * N`).

## MatrixRef / MatrixMut Traits

All decomposition free functions are written against two simple accessor traits:

```rust
pub trait MatrixRef<T> {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn get(&self, row: usize, col: usize) -> T;
    fn col_as_slice(&self, col: usize) -> &[T];
}

pub trait MatrixMut<T>: MatrixRef<T> {
    fn get_mut(&mut self, row: usize, col: usize) -> &mut T;
    fn col_as_mut_slice(&mut self, col: usize) -> &mut [T];
}
```

Both `Matrix<T, M, N>` and `DynMatrix<T>` implement these traits. The consequence: **the same LU, Cholesky, QR, SVD, Eigen, and Schur code handles fixed and dynamic matrices**. There are no separate implementations — just one set of free functions.

`col_as_slice()` / `col_as_mut_slice()` are the key methods: they return contiguous `&[T]` slices of individual columns, enabling SIMD dispatch to operate on contiguous data without gathering.

## LinalgScalar Hierarchy

Three element traits form a hierarchy:

```
Scalar          Copy + PartialEq + Debug + Zero + One + Num
  └─ FloatScalar    Scalar + Float + LinalgScalar<Real=Self>
  └─ LinalgScalar   Scalar + modulus + conj + re + lsqrt + lln + from_real
```

| Trait | Used by | Examples |
|---|---|---|
| `Scalar` | Matrix ops, arithmetic, iteration | `i32`, `u64`, `f32`, `f64`, `Complex<f64>` |
| `FloatScalar` | Quaternion, ODE, optim, estimate, ordered comparisons | `f32`, `f64` |
| `LinalgScalar` | Decompositions, norms | `f32`, `f64`, `Complex<f32>`, `Complex<f64>` |

**Integer matrices work**: `Matrix<i32, 3, 3>` supports arithmetic, indexing, iteration, and transpose. It does not support `det()`, norms, or decompositions (those require `LinalgScalar`).

**Complex is additive**: `LinalgScalar` methods on `f64` — `conj()`, `re()`, `modulus()` — are `#[inline]` identity functions, fully erased by the compiler. Complex support adds zero code to real-valued paths.

## In-Place Algorithms

Decompositions operate in-place on `&mut impl MatrixMut<T>`. This avoids:

- Allocator/storage traits (nalgebra's approach)
- Cloning the input (scipy's approach)
- Extra indirection through trait objects

The pattern:

```rust
pub fn cholesky_in_place<T: LinalgScalar>(
    a: &mut impl MatrixMut<T>,
) -> Result<(), LinalgError> {
    // modifies a in-place: a → L (lower triangular factor)
}
```

Wrapper structs (`CholeskyDecomposition`, etc.) own the modified matrix and provide `solve()`, `inverse()`, `det()` as convenience methods.

## SIMD Dispatch

SIMD is selected at monomorphization time via `TypeId`:

```rust
pub fn dot_dispatch<T: Scalar>(a: &[T], b: &[T]) -> T {
    use core::any::TypeId;
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: we just verified T = f64
        unsafe { f64_dot(transmute(a), transmute(b)) as T }
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        unsafe { f32_dot(transmute(a), transmute(b)) as T }
    } else {
        scalar_dot(a, b)  // generic fallback for integers, complex, etc.
    }
}
```

The `TypeId` check is a compile-time constant — dead branches are eliminated by LLVM at monomorphization. For integer `T`, the entire SIMD path compiles away; for `f64`, only the f64 path remains.

The `Scalar` trait has a `'static` bound (required by `TypeId`). This is backwards-compatible — all scalar types are `'static`.

## DynMatrix Storage

`DynMatrix<T>` uses `Vec<T>` in column-major order: element `(row, col)` is at index `col * nrows + row`. Implementing `MatrixRef`/`MatrixMut` means all decomposition free functions work automatically, with no duplicate code.

`DynVector` is a newtype wrapper around `DynMatrix` that enforces the single-column invariant and provides single-index access.

## Avoiding Unstable Features

numeris uses only stable Rust (MSRV 1.70). Constraints:

- `generic_const_exprs` is unstable → no `[T; M*N]` flat storage
- `min_const_generics` (stable since 1.51) → `[[T; M]; N]` two-level storage works
- `core::arch` intrinsics for SIMD are stable on aarch64 and x86_64

## num-traits Integration

numeris uses `num-traits` with `default-features = false` for generic numeric bounds. This avoids pulling in `std` through `num-traits` in no-std mode. Key traits used:

- `Zero`, `One` — additive/multiplicative identity
- `Num` — basic arithmetic (Scalar bound)
- `Float` — transcendentals, `sqrt`, `sin`, etc. (FloatScalar bound)

When `std` is enabled, `Float` delegates to the system's hardware libm. Without `std`, it uses `libm`'s software implementations via the `libm` feature.
