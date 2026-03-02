//! # numeris
//!
//! Pure-Rust numerical algorithms library, no-std compatible. Similar in scope
//! to SciPy, suitable for embedded targets (no heap allocation, no FPU assumptions).
//!
//! ## Quick start
//!
//! ```
//! use numeris::{Matrix, Vector};
//!
//! // Solve a linear system Ax = b
//! let a = Matrix::new([
//!     [2.0_f64, 1.0, -1.0],
//!     [-3.0, -1.0, 2.0],
//!     [-2.0, 1.0, 2.0],
//! ]);
//! let b = Vector::from_array([8.0, -11.0, -3.0]);
//! let x = a.solve(&b).unwrap(); // x = [2, 3, -1]
//! ```
//!
//! ## Modules
//!
//! - [`matrix`] — Fixed-size `Matrix<T, M, N>` with const-generic dimensions.
//!   Stack-allocated `[[T; M]; N]` column-major storage (matches LAPACK conventions).
//!   `Matrix::new()` accepts row-major input and transposes internally.
//!   Includes arithmetic, indexing, norms, block operations, and iteration.
//!   [`Vector<T, N>`] and [`ColumnVector<T, N>`] are type aliases for 1-row and
//!   1-column matrices.
//!
//! - [`dynmatrix`] — Heap-allocated `DynMatrix<T>` with runtime dimensions
//!   (requires `alloc` feature, included with `std`). `Vec<T>` column-major storage
//!   (`col * nrows + row`). `from_rows()` accepts row-major data (transposes
//!   internally); `from_slice()` accepts column-major data directly.
//!   Implements [`MatrixRef`] / [`MatrixMut`], so all linalg free functions work
//!   automatically. [`DynVector<T>`] newtype for single-index vector access.
//!   Includes `DynLu`, `DynCholesky`, `DynQr`, `DynSvd`, `DynSymmetricEigen`,
//!   `DynSchur` wrapper structs.
//!
//! - [`linalg`] — LU (partial pivoting), Cholesky (A = LL^H), QR (Householder),
//!   SVD (Householder bidiagonalization + Golub-Kahan implicit-shift QR),
//!   symmetric/Hermitian eigendecomposition (Householder tridiagonalization +
//!   implicit QR with Wilkinson shift), and real Schur decomposition (Hessenberg
//!   reduction + Francis double-shift QR). Each provides `solve()`, `inverse()`,
//!   `det()`, `eigenvalues()`, etc. Free functions operate on
//!   `&mut impl MatrixMut<T>` for in-place use; wrapper structs offer a
//!   higher-level API. Convenience methods on both `Matrix` and `DynMatrix`.
//!
//! - [`ode`] — Fixed-step RK4 and 7 adaptive Runge-Kutta solvers (RKF45,
//!   RKTS54, RKV65, RKV87, RKV98, RKV98NoInterp, RKV98Efficient). PI step-size
//!   controller with dense output / interpolation. Requires `ode` feature.
//!
//! - [`optim`] — Optimization: scalar root finding ([`optim::brent`],
//!   [`optim::newton_1d`]), BFGS quasi-Newton minimization ([`optim::minimize_bfgs`]),
//!   Gauss-Newton ([`optim::least_squares_gn`]) and Levenberg-Marquardt
//!   ([`optim::least_squares_lm`]) nonlinear least squares. Finite-difference
//!   Jacobian and gradient utilities. Requires `optim` feature.
//!
//! - [`control`] — Digital IIR filters: [`control::Biquad`] second-order section and
//!   [`control::BiquadCascade`] for cascaded filters. Design functions for Butterworth
//!   and Chebyshev Type I lowpass/highpass. No `complex` feature dependency.
//!   Requires `control` feature.
//!
//! - [`estimate`] — State estimation: [`estimate::Ekf`] (Extended Kalman Filter),
//!   [`estimate::Ukf`] (Unscented Kalman Filter), [`estimate::SrUkf`] (Square-Root UKF),
//!   [`estimate::Ckf`] (Cubature Kalman Filter), [`estimate::rts_smooth`] (RTS smoother),
//!   and [`estimate::BatchLsq`] (batch least-squares). Closure-based dynamics and
//!   measurement models, Joseph-form covariance update, Merwe-scaled sigma points.
//!   EKF and BatchLsq are fully no-std; sigma-point filters and RTS require `alloc`.
//!   Requires `estimate` feature.
//!
//! - [`interp`] — Interpolation: [`interp::LinearInterp`], [`interp::HermiteInterp`],
//!   [`interp::LagrangeInterp`] (barycentric), and [`interp::CubicSpline`] (natural BCs).
//!   Fixed-size (const N, stack-allocated, no-std) and dynamic variants (`Dyn*`, requires
//!   `alloc`). Out-of-bounds evaluations extrapolate. Requires `interp` feature.
//!
//! - [`special`] — Special functions: [`special::gamma`], [`special::lgamma`],
//!   [`special::digamma`], [`special::beta`] / [`special::lbeta`],
//!   regularized incomplete gamma ([`special::gamma_inc`] / [`special::gamma_inc_upper`]),
//!   regularized incomplete beta ([`special::betainc`]),
//!   and error functions ([`special::erf`] / [`special::erfc`]).
//!   Generic over `FloatScalar` (f32/f64), fully no-std. Requires `special` feature.
//!
//! - [`stats`] — Statistical distributions with [`stats::ContinuousDistribution`] and
//!   [`stats::DiscreteDistribution`] traits. Continuous: [`stats::Normal`],
//!   [`stats::Uniform`], [`stats::Exponential`], [`stats::Gamma`], [`stats::Beta`],
//!   [`stats::ChiSquared`], [`stats::StudentT`]. Discrete: [`stats::Bernoulli`],
//!   [`stats::Binomial`], [`stats::Poisson`]. Requires `stats` feature (implies `special`).
//!
//! - [`quaternion`] — Unit quaternion for 3D rotations. Scalar-first `[w, x, y, z]`.
//!   Construct from axis-angle, Euler angles, or rotation matrices. Supports
//!   Hamilton product, vector rotation, SLERP, and conversion back to matrices.
//!
//! - [`traits`] — Element trait hierarchy:
//!   - [`Scalar`] — all matrix elements (`Copy + PartialEq + Debug + Zero + One + Num`)
//!   - [`FloatScalar`] — real floats (`Scalar + Float`), used by quaternions
//!   - [`LinalgScalar`] — real floats and complex numbers, used by decompositions and norms
//!   - [`MatrixRef`] / [`MatrixMut`] — generic read/write access for algorithms
//!
//! ## Complex matrices
//!
//! Enable the `complex` feature to use decompositions with `Complex<f32>` /
//! `Complex<f64>`. Cholesky generalizes to Hermitian (A = LL^H), QR uses
//! complex Householder reflections, and norms return real values. Zero overhead
//! for real-only code paths.
//!
//! ## Cargo features
//!
//! | Feature   | Default  | Description |
//! |-----------|----------|-------------|
//! | `std`     | yes      | Implies `alloc`. Hardware FPU via system libm |
//! | `alloc`   | via std  | `DynMatrix` / `DynVector` (heap-allocated, runtime-sized) |
//! | `ode`     | yes      | ODE integration (RK4, adaptive solvers) |
//! | `optim`   | no       | Optimization (root finding, BFGS, Gauss-Newton, LM) |
//! | `control` | no       | Digital IIR filters (Butterworth, Chebyshev Type I) |
//! | `estimate`| no       | State estimation (EKF, UKF). Implies `alloc` |
//! | `interp`  | no       | Interpolation (linear, Hermite, Lagrange, cubic spline) |
//! | `special` | no       | Special functions (gamma, beta, erf, incomplete gamma/beta) |
//! | `stats`   | no       | Statistical distributions (Normal, Gamma, etc.). Implies `special` |
//! | `libm`    | baseline | Pure-Rust software float fallback |
//! | `complex` | no       | `Complex<f32>` / `Complex<f64>` support via `num-complex` |
//! | `all`     | no       | All features |

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
pub mod dynmatrix;
pub mod linalg;
pub mod matrix;
mod simd;
#[cfg(feature = "ode")]
pub mod ode;
#[cfg(feature = "control")]
pub mod control;
#[cfg(feature = "estimate")]
pub mod estimate;
#[cfg(feature = "interp")]
pub mod interp;
#[cfg(feature = "optim")]
pub mod optim;
#[cfg(feature = "special")]
pub mod special;
#[cfg(feature = "stats")]
pub mod stats;
pub mod quaternion;
pub mod traits;

pub use matrix::vector::{ColumnVector, ColumnVector3, Vector, Vector3};
pub use matrix::Matrix;
pub use matrix::aliases::{
    Matrix1, Matrix2, Matrix3, Matrix4, Matrix5, Matrix6,
    Matrix1x2, Matrix1x3, Matrix1x4, Matrix1x5, Matrix1x6,
    Matrix2x1, Matrix2x3, Matrix2x4, Matrix2x5, Matrix2x6,
    Matrix3x1, Matrix3x2, Matrix3x4, Matrix3x5, Matrix3x6,
    Matrix4x1, Matrix4x2, Matrix4x3, Matrix4x5, Matrix4x6,
    Matrix5x1, Matrix5x2, Matrix5x3, Matrix5x4, Matrix5x6,
    Matrix6x1, Matrix6x2, Matrix6x3, Matrix6x4, Matrix6x5,
    Vector1, Vector2, Vector4, Vector5, Vector6,
    ColumnVector1, ColumnVector2, ColumnVector4, ColumnVector5, ColumnVector6,
};
#[cfg(feature = "alloc")]
pub use dynmatrix::{
    DynMatrix, DynVector,
    DynMatrixf32, DynMatrixf64, DynMatrixi32, DynMatrixi64, DynMatrixu32, DynMatrixu64,
    DynVectorf32, DynVectorf64, DynVectori32, DynVectori64, DynVectoru32, DynVectoru64,
};
#[cfg(all(feature = "alloc", feature = "complex"))]
pub use dynmatrix::{DynMatrixz32, DynMatrixz64, DynVectorz32, DynVectorz64};
pub use quaternion::Quaternion;
pub use traits::{FloatScalar, LinalgScalar, MatrixMut, MatrixRef, Scalar};

#[cfg(feature = "complex")]
pub use num_complex::Complex;
