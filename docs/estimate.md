# State Estimation

Six estimators for nonlinear state estimation and offline batch processing. All use const-generic state (`N`) and measurement (`M`) dimensions with closure-based dynamics and measurement models.

Requires the `estimate` Cargo feature (implies `alloc`):

```toml
numeris = { version = "0.2", features = ["estimate"] }
```

## Filter Comparison

| Filter | Struct | Jacobians | No-std | When to use |
|---|---|---|---|---|
| Extended Kalman | `Ekf<T, N, M>` | User-supplied or FD | **Yes** | Linear-ish dynamics, fast, no-alloc |
| Unscented Kalman | `Ukf<T, N, M>` | Not needed | No (`alloc`) | Moderately nonlinear dynamics |
| Square-Root UKF | `SrUkf<T, N, M>` | Not needed | No (`alloc`) | UKF + guaranteed PD covariance |
| Cubature Kalman | `Ckf<T, N, M>` | Not needed | No (`alloc`) | No tuning parameters, 2N points |
| RTS Smoother | `rts_smooth` | From EKF forward pass | No (`alloc`) | Offline batch smoothing |
| Batch Least-Squares | `BatchLsq<T, N>` | Linear H matrix | **Yes** | Linear observations, offline |

All support `f32` and `f64`. Process noise `Q` is optional (`Some(&q)` or `None`).

## Extended Kalman Filter (EKF)

The EKF linearizes nonlinear dynamics at the current estimate using Jacobians. Fully no-std and no-alloc.

```rust
use numeris::estimate::Ekf;
use numeris::{ColumnVector, Matrix};

// 2-state constant-velocity model, 1-dimensional position measurement
let x0 = ColumnVector::from_column([0.0_f64, 1.0]);  // [pos, vel]
let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
let mut ekf = Ekf::<f64, 2, 1>::new(x0, p0);

let dt = 0.1_f64;
let q = Some(Matrix::new([[0.01_f64, 0.0], [0.0, 0.01]]));  // process noise
let r = Matrix::new([[0.5_f64]]);                              // measurement noise

// Prediction step: provide dynamics f(x) and Jacobian F = ∂f/∂x
ekf.predict(
    |x| ColumnVector::from_column([
        x[(0, 0)] + dt * x[(1, 0)],
        x[(1, 0)],
    ]),
    |_x| Matrix::new([[1.0_f64, dt], [0.0, 1.0]]),
    q.as_ref(),
);

// Update step: provide measurement h(x) and Jacobian H = ∂h/∂x
ekf.update(
    &ColumnVector::from_column([0.12_f64]),  // measurement z
    |x| ColumnVector::from_column([x[(0, 0)]]),
    |_x| Matrix::new([[1.0_f64, 0.0]]),
    &r,
).unwrap();

// Access state and covariance
let x = ekf.state();       // ColumnVector<f64, 2>
let p = ekf.covariance();  // Matrix<f64, 2, 2>
```

### Finite-Difference Jacobians

When analytic Jacobians are unavailable, use `predict_fd` / `update_fd`:

```rust
ekf.predict_fd(
    |x| ColumnVector::from_column([x[(0,0)] + dt*x[(1,0)], x[(1,0)]]),
    q.as_ref(),
);
ekf.update_fd(
    &ColumnVector::from_column([0.12]),
    |x| ColumnVector::from_column([x[(0,0)]]),
    &r,
).unwrap();
```

Finite differences use forward differences with step size `√ε` (≈ 1.5×10⁻⁸ for `f64`).

## Unscented Kalman Filter (UKF)

Propagates 2N+1 sigma points through the nonlinear dynamics — no Jacobians needed. Uses Merwe-scaled sigma points with tunable parameters.

```rust
use numeris::estimate::Ukf;
use numeris::{ColumnVector, Matrix};

let mut ukf = Ukf::<f64, 2, 1>::new(x0, p0);

// Optional tuning: alpha (spread), beta (prior on distribution), kappa
// Default: alpha=0.001, beta=2.0, kappa=0.0 (good for most cases)
let mut ukf_tuned = Ukf::<f64, 2, 1>::with_params(x0, p0, 0.001, 2.0, 0.0);

ukf.predict(
    |x| ColumnVector::from_column([x[(0,0)] + dt*x[(1,0)], x[(1,0)]]),
    q.as_ref(),
).unwrap();

ukf.update(
    &ColumnVector::from_column([0.12]),
    |x| ColumnVector::from_column([x[(0,0)]]),
    &r,
).unwrap();
```

## Square-Root UKF (SR-UKF)

Propagates the Cholesky factor `S` (where `P = SSᵀ`) rather than `P` directly. Guarantees positive-definiteness of the covariance even in the presence of numerical round-off.

```rust
use numeris::estimate::SrUkf;
use numeris::{ColumnVector, Matrix};

// Construct from covariance P (Cholesky computed internally)
let mut srukf = SrUkf::<f64, 2, 1>::from_covariance(x0, p0);

// Or from the Cholesky factor S directly
let s0 = p0.cholesky().unwrap().l();  // lower-triangular factor
let mut srukf2 = SrUkf::<f64, 2, 1>::new(x0, s0);

srukf.predict(
    |x| ColumnVector::from_column([x[(0,0)] + dt*x[(1,0)], x[(1,0)]]),
    q.as_ref(),
).unwrap();

srukf.update(
    &ColumnVector::from_column([0.12]),
    |x| ColumnVector::from_column([x[(0,0)]]),
    &r,
).unwrap();

let p = srukf.covariance();  // reconstructed P = SSᵀ
```

## Cubature Kalman Filter (CKF)

Third-degree spherical-radial cubature rule: 2N equally-weighted cubature points `x ± √N · L · eᵢ` where `L = chol(P)`. No tuning parameters.

```rust
use numeris::estimate::Ckf;
use numeris::{ColumnVector, Matrix};

let mut ckf = Ckf::<f64, 2, 1>::new(x0, p0);

ckf.predict(
    |x| ColumnVector::from_column([x[(0,0)] + dt*x[(1,0)], x[(1,0)]]),
    q.as_ref(),
).unwrap();

ckf.update(
    &ColumnVector::from_column([0.12]),
    |x| ColumnVector::from_column([x[(0,0)]]),
    &r,
).unwrap();
```

## RTS Smoother

Rauch–Tung–Striebel fixed-interval smoother: runs an EKF forward pass, stores `EkfStep` records, then runs a backward pass to produce smoothed estimates. Smoothed covariance is always ≤ filtered covariance.

```rust
use numeris::estimate::{Ekf, EkfStep, rts_smooth};
use numeris::{ColumnVector, Matrix};

// Forward EKF pass — store each step
let mut ekf = Ekf::<f64, 2, 1>::new(x0, p0);
let mut steps: Vec<EkfStep<f64, 2>> = Vec::new();

for z in &measurements {
    let step = ekf.predict_store(
        |x| ColumnVector::from_column([x[(0,0)] + dt*x[(1,0)], x[(1,0)]]),
        |_x| Matrix::new([[1.0_f64, dt], [0.0, 1.0]]),
        q.as_ref(),
    );
    steps.push(step);

    ekf.update(
        &ColumnVector::from_column([*z]),
        |x| ColumnVector::from_column([x[(0,0)]]),
        |_x| Matrix::new([[1.0_f64, 0.0]]),
        &r,
    ).unwrap();
}

// Backward RTS smoother pass
let smoothed = rts_smooth(&steps).unwrap();
// smoothed[k].x — smoothed state at step k
// smoothed[k].p — smoothed covariance at step k
```

## Batch Least-Squares

Information-form batch least-squares: accumulate `Λ = Σ HᵀR⁻¹H`, `η = Σ HᵀR⁻¹z`, then solve `Λx = η`. Fully no-std, no-alloc.

Supports mixed measurement dimensions via method-level const generic `M`.

```rust
use numeris::estimate::BatchLsq;
use numeris::{ColumnVector, Matrix};

let mut lsq = BatchLsq::<f64, 2>::new();  // 2-state

// Add observations with (possibly different) H dimensions
let h1 = Matrix::new([[1.0_f64, 0.0]]);    // 1×2: observe position
let h2 = Matrix::new([[0.0_f64, 1.0]]);    // 1×2: observe velocity
let r  = Matrix::new([[0.1_f64]]);

lsq.add_observation(&ColumnVector::from_column([1.05_f64]), &h1, &r).unwrap();
lsq.add_observation(&ColumnVector::from_column([0.95_f64]), &h1, &r).unwrap();
lsq.add_observation(&ColumnVector::from_column([1.0_f64]),  &h2, &r).unwrap();

// Solve: returns (state estimate, covariance)
let (x_est, p_est) = lsq.solve().unwrap();

// With a prior (Tikhonov regularization):
let x_prior = ColumnVector::from_column([0.0_f64, 0.0]);
let p_prior  = Matrix::new([[1.0_f64, 0.0], [0.0, 1.0]]);
let mut lsq_prior = BatchLsq::<f64, 2>::with_prior(&x_prior, &p_prior).unwrap();
```

## Choosing a Filter

```
Are your dynamics highly nonlinear?
  → No:  EKF (fast, no-alloc, good enough for mildly nonlinear systems)
  → Yes: UKF or CKF

Do you need guaranteed positive-definite covariance in long runs?
  → SrUkf (propagates Cholesky factor)

Do you want zero tuning parameters?
  → CKF (2N points, equal weights)

Are you doing offline post-processing with all data available?
  → RTS smoother (better estimates than forward-only filtering)

Are your measurements linear in the state?
  → BatchLsq (simple, no-alloc, optimal for linear-Gaussian problems)
```

## Error Handling

```rust
use numeris::estimate::EstimateError;

match ekf.update(&z, h, hj, &r) {
    Ok(())                                      => { /* success */ }
    Err(EstimateError::CovarianceNotPD)         => { /* P lost positive-definiteness */ }
    Err(EstimateError::SingularInnovation)      => { /* S = HPHᵀ + R is singular */ }
    Err(EstimateError::CholdowndateFailed)      => { /* SR-UKF Cholesky downdate failed */ }
}
```
