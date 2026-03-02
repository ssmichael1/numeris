# numeris

**A mostly self-contained numerical algorithms library with minimal external dependencies.**

numeris is suitable for use on **embedded microprocessors** with no operating system or heap allocation (`no-std`), while still taking advantage of SIMD instructions to be highly performant on more capable hardware. The same code compiles and runs correctly in both environments.

!!! warning "Alpha software"
    APIs are unstable and may change without notice before 1.0.

## Feature Highlights

| Area | What's included |
|---|---|
| **Matrix** | Stack-allocated `Matrix<T, M, N>` with const-generic dimensions; `DynMatrix<T>` for runtime sizes |
| **Linear Algebra** | LU, Cholesky, QR, SVD, symmetric eigendecomposition, real Schur decomposition |
| **ODE** | Fixed-step RK4 + 7 adaptive Runge-Kutta solvers + RODAS4 stiff solver |
| **Optimization** | Brent, Newton, BFGS, Gauss-Newton, Levenberg-Marquardt |
| **State Estimation** | EKF, UKF, SR-UKF, CKF, RTS smoother, batch least-squares |
| **Interpolation** | Linear, Hermite, Lagrange, cubic spline, bilinear |
| **Special Functions** | gamma, lgamma, digamma, beta, betainc, incomplete gamma, erf |
| **Statistics** | 10 distributions (Normal, Gamma, Beta, Student's t, Poisson, …) |
| **Digital Control** | Butterworth/Chebyshev IIR filters, PID controller |
| **Quaternion** | Unit quaternion rotations, SLERP, Euler angles, rotation matrices |
| **SIMD** | NEON (aarch64), SSE2/AVX/AVX-512 (x86_64) — always-on, no feature flag |

## Quick Start

Add to `Cargo.toml`:

```toml
[dependencies]
numeris = "0.2"
```

```rust
use numeris::{Matrix, Vector};

// Solve Ax = b
let a = Matrix::new([
    [2.0_f64, 1.0, -1.0],
    [-3.0, -1.0,  2.0],
    [-2.0,  1.0,  2.0],
]);
let b = Vector::from_array([8.0, -11.0, -3.0]);
let x = a.solve(&b).unwrap(); // x = [2, 3, -1]

// Cholesky decomposition
let spd = Matrix::new([[4.0_f64, 2.0], [2.0, 3.0]]);
let inv = spd.cholesky().unwrap().inverse();

// Symmetric eigendecomposition
let sym = Matrix::new([[4.0_f64, 1.0], [1.0, 3.0]]);
let eig = sym.eig_symmetric().unwrap();
let vals = eig.eigenvalues();   // sorted ascending
let vecs = eig.eigenvectors();  // columns = eigenvectors

// Quaternion rotation
use numeris::Quaternion;
let q = Quaternion::from_axis_angle(
    &Vector::from_array([0.0, 0.0, 1.0]),
    std::f64::consts::FRAC_PI_2,
);
let v = Vector::from_array([1.0, 0.0, 0.0]);
let rotated = q * v; // ≈ [0, 1, 0]
```

## Navigation

- **[Getting Started](getting-started.md)** — installation, feature flags, first examples
- **Module docs** — detailed pages for each module (use the top tabs)
- **[Performance](performance.md)** — SIMD tiers, benchmark numbers vs. nalgebra and faer
- **[No-std / Embedded](no-std.md)** — usage without std or heap allocation
- **[Design](design.md)** — architectural decisions and trait hierarchy
- **[API Reference ↗](https://docs.rs/numeris)** — full rustdoc on docs.rs
