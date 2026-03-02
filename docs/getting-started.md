# Getting Started

## Installation

Add numeris to your `Cargo.toml`:

```toml
[dependencies]
numeris = "0.2"
```

The default features include `std` and `ode`. To enable additional modules, list them explicitly:

```toml
[dependencies]
numeris = { version = "0.2", features = ["optim", "control", "estimate", "interp", "special", "stats", "complex"] }

# Or enable everything at once:
numeris = { version = "0.2", features = ["all"] }
```

## Cargo Features

| Feature | Default | Description |
|---|---|---|
| `std` | **yes** | Implies `alloc`. Uses hardware FPU via system libm. Full float speed. |
| `alloc` | via `std` | Enables `DynMatrix` / `DynVector` (heap-allocated, runtime-sized). |
| `ode` | **yes** | ODE integration — RK4, 7 adaptive solvers, RODAS4 stiff solver. |
| `optim` | no | Optimization — root finding, BFGS, Gauss-Newton, Levenberg-Marquardt. |
| `control` | no | Digital IIR filters (Butterworth, Chebyshev Type I) and PID controller. |
| `estimate` | no | State estimation — EKF, UKF, SR-UKF, CKF, RTS smoother, batch LSQ. Implies `alloc`. |
| `interp` | no | Interpolation — linear, Hermite, Lagrange, cubic spline, bilinear. |
| `special` | no | Special functions — gamma, lgamma, digamma, beta, betainc, erf. |
| `stats` | no | Statistical distributions (10 families). Implies `special`. |
| `complex` | no | `Complex<f32>` / `Complex<f64>` support for all decompositions. |
| `libm` | baseline | Pure-Rust software float math. Always on as fallback. |
| `all` | no | All of the above. |

## Build Variants

```bash
# Default (std + ode)
cargo build

# All features — the kitchen sink
cargo build --features all

# No-std embedded target
cargo build --no-default-features --features libm

# No-std with heap (DynMatrix available)
cargo build --no-default-features --features "libm,alloc"

# Optimization + complex numbers
cargo build --features "optim,complex"

# State estimation (also enables DynMatrix)
cargo build --features estimate
```

## SIMD Acceleration

SIMD is **always-on** for `f32` and `f64` — no feature flag required.

- **aarch64**: NEON intrinsics, always available
- **x86_64**: SSE2 always available; AVX and AVX-512 via compiler flags

To enable AVX/AVX-512 on x86_64:

```bash
# Enable all native CPU features (recommended for desktop/server)
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Or explicitly
RUSTFLAGS="-C target-feature=+avx2,+avx512f" cargo build --release
```

## First Examples

### Matrix arithmetic

```rust
use numeris::{Matrix, Vector, Matrix3};

// Matrix creation — new() takes row-major input, stores column-major
let a = Matrix::new([
    [1.0_f64, 2.0, 3.0],
    [4.0,     5.0, 6.0],
    [7.0,     8.0, 9.0],
]);

// Size aliases
let id: Matrix3<f64> = Matrix3::eye();
let z: Matrix3<f64>  = Matrix3::zeros();

// Arithmetic
let b = a * id;          // matrix multiply
let c = a + &b;          // element-wise add
let d = a * 2.0;         // scalar multiply

// Vectors
let v = Vector::from_array([1.0_f64, 0.0, 0.0]);
let w = a.vecmul(&v);    // A * v (matrix-vector)
let dot = v.dot(&w);

// Indexing
let elem = a[(1, 2)];    // row 1, col 2 = 6.0
```

### Linear system solve

```rust
use numeris::{Matrix, Vector};

let a = Matrix::new([
    [2.0_f64, 1.0, -1.0],
    [-3.0,   -1.0,  2.0],
    [-2.0,    1.0,  2.0],
]);
let b = Vector::from_array([8.0, -11.0, -3.0]);

// High-level convenience
let x = a.solve(&b).unwrap();
assert!((x[0] - 2.0).abs() < 1e-12);
assert!((x[1] - 3.0).abs() < 1e-12);
assert!((x[2] + 1.0).abs() < 1e-12);

// Or access the decomposition directly
let lu = a.lu().unwrap();
let x2 = lu.solve(&b);
let inv = lu.inverse();
let det = lu.det();
```

### ODE integration

```rust
use numeris::ode::{RKAdaptive, RKTS54, AdaptiveSettings};
use numeris::Vector;

// Simple harmonic oscillator: [x, v]' = [v, -x]
let y0 = Vector::from_array([1.0_f64, 0.0]);  // x=1, v=0
let tau = 2.0 * std::f64::consts::PI;

let sol = RKTS54::integrate(
    0.0, tau, &y0,
    |_t, y| Vector::from_array([y[1], -y[0]]),
    &AdaptiveSettings::default(),
).unwrap();

// After one full period, x ≈ 1, v ≈ 0
assert!((sol.y[0] - 1.0).abs() < 1e-6);
assert!(sol.y[1].abs() < 1e-6);
```

### Dynamic matrices

```rust
use numeris::{DynMatrix, DynVector};

let a = DynMatrix::from_rows(3, 3, &[
    2.0_f64, 1.0, -1.0,
    -3.0,   -1.0,  2.0,
    -2.0,    1.0,  2.0,
]);
let b = DynVector::from_slice(&[8.0, -11.0, -3.0]);
let x = a.solve(&b).unwrap();
```

See the [Matrix](matrix.md) and [DynMatrix](dynmatrix.md) pages for the full API.
