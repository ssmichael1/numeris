# No-std / Embedded

numeris is designed to run on embedded targets with no operating system, no heap, and no floating-point hardware. All core modules compile and run correctly in `no_std` environments.

## Quick Start for Embedded

```toml
# Cargo.toml
[dependencies]
numeris = { version = "0.2", default-features = false, features = ["libm"] }
```

This gives you:

- Fixed-size `Matrix<T, M, N>` (stack-allocated, no heap)
- `Quaternion<T>` rotations
- All linear algebra decompositions (LU, Cholesky, QR, SVD, Eigen, Schur)
- ODE integration: RK4 (`rk4`, `rk4_step` — fully no-alloc)
- Special functions (with `special` feature)

## Feature Flag Guide

| Goal | `features` | `default-features` |
|---|---|---|
| Full desktop use | `["all"]` or default | yes |
| No-std, stack-only | `["libm"]` | **no** |
| No-std + heap | `["libm", "alloc"]` | **no** |
| No-std + ODE | `["libm", "ode"]` | **no** |
| No-std + filters | `["libm", "control"]` | **no** |
| No-std + EKF only | `["libm", "estimate"]` — but estimate implies alloc | **no** |

!!! info "Alloc without std"
    The `alloc` feature enables `DynMatrix`, `DynVector`, and the sigma-point filters (UKF, SR-UKF, CKF). It works on targets with a global allocator but no full `std`. On bare-metal embedded, you need to provide your own allocator via `#[global_allocator]`.

## Float Math

| Mode | Float source | Performance |
|---|---|---|
| `std` enabled | System libm (hardware FPU) | Hardware speed |
| `no_std` + `libm` | Pure-Rust `libm` crate | Software emulation |

numeris selects float math via the `num-traits` `Float` trait. When `std` is enabled, this delegates to the system's hardware-backed `sin`, `sqrt`, etc. Without `std`, it uses the `libm` crate's software implementations — accurate to within 1 ULP on all platforms.

## Modules Available Without Heap

These modules work with `#![no_std]` and zero heap allocation:

| Module | No-heap | Notes |
|---|---|---|
| `matrix` | ✓ | Stack-allocated `Matrix<T, M, N>` |
| `quaternion` | ✓ | |
| `linalg` | ✓ | All six decompositions |
| `ode` | ✓ (RK4) | Adaptive solvers need `alloc` for dense output |
| `control` | ✓ | Filters and PID |
| `special` | ✓ | |
| `stats` | ✓ | |
| `interp` | ✓ | Fixed-size variants (`LinearInterp<T, N>`, etc.) |
| `optim` | ✓ | Root finding, BFGS, GN, LM |
| `estimate` | ✓ (EKF, BatchLsq) | UKF/SR-UKF/CKF require `alloc` |
| `dynmatrix` | ✗ | Requires `alloc` |

## Example: Embedded EKF (no heap)

```rust
#![no_std]
#![no_main]

use numeris::estimate::Ekf;
use numeris::{ColumnVector, Matrix};

// 4-state IMU attitude EKF (quaternion + bias)
// Works on Cortex-M4/M7 with FPU, no heap required
static mut EKF: Option<Ekf<f32, 4, 3>> = None;

pub fn init() {
    let x0 = ColumnVector::<f32, 4>::zeros();
    let p0 = Matrix::<f32, 4, 4>::eye();

    unsafe { EKF = Some(Ekf::new(x0, p0)); }
}

pub fn imu_update(gyro: [f32; 3], dt: f32) {
    let q  = Matrix::<f32, 4, 4>::eye();  // process noise
    let r  = Matrix::<f32, 3, 3>::eye();  // measurement noise

    unsafe {
        if let Some(ref mut ekf) = EKF {
            ekf.predict(
                |x| {
                    // Quaternion kinematics: q_dot = 0.5 * Omega(gyro - bias) * q
                    ColumnVector::zeros()  // simplified
                },
                |_x| Matrix::eye(),
                Some(&q),
            );
        }
    }
}
```

## Example: Fixed-Step RK4 (no heap, no alloc)

```rust
#![no_std]
use numeris::ode::rk4_step;
use numeris::Vector;

// Satellite orbital mechanics — called from interrupt
pub fn propagate(t: f64, y: &Vector<f64, 6>, dt: f64) -> Vector<f64, 6> {
    rk4_step(t, y, |_t, state| {
        // Two-body gravity: f = [v; -μ/r³ * r]
        let r = state.head::<3>();
        let v = state.tail::<3>();
        let r3 = r.norm().powi(3);
        let mu = 3.986e14_f64;

        let mut deriv = Vector::zeros();
        for i in 0..3 { deriv[i] = v[i]; }
        for i in 0..3 { deriv[i+3] = -mu / r3 * r[i]; }
        deriv
    }, dt)
}
```

## Target Considerations

| Target class | Heap | FPU | SIMD | Recommended config |
|---|---|---|---|---|
| Cortex-M0/M0+ | no | no | no | `no-default-features`, `libm` |
| Cortex-M4F/M7 | optional | yes | no | `no-default-features`, `libm` + FPU target |
| RISC-V (bare-metal) | optional | optional | no | `no-default-features`, `libm` |
| aarch64 (Linux) | yes | yes | NEON | default or `all` |
| x86_64 (Linux/macOS) | yes | yes | SSE2/AVX | default or `all` |

## Cortex-M with FPU

On Cortex-M4F/M7 with hardware FPU, you can get near-hardware-speed float math by using the target's native float ABI:

```bash
cargo build \
  --target thumbv7em-none-eabihf \
  --no-default-features \
  --features "libm,ode,estimate" \
  --release
```

The `thumbv7em-none-eabihf` target uses the VFP/FPU instructions natively. numeris's `f32` and `f64` operations will use hardware float through `libm`'s software path on this target class — for best performance, you may want to configure a linker override that routes `libm` calls to the Cortex-M CMSIS-DSP library.

## Integer Matrices

Integer `Matrix<i32, M, N>`, `Matrix<u64, M, N>`, etc. work with any scalar type implementing the `Scalar` trait (no float required). Only float-specific operations (det, norms, decompositions) require `FloatScalar` or `LinalgScalar`.

```rust
#![no_std]
use numeris::Matrix;

let a = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
let b = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
let c = a * b;   // integer matrix multiply — no float, no alloc
```
