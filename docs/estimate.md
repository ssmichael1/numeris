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

## Robustness Features

All filters share the same set of robustness builder methods and a consistent update API:

| Feature | Method | Applies to |
|---|---|---|
| Cholesky with jitter | automatic (internal) | all filters |
| Covariance floor | `.with_min_variance(v)` | EKF, UKF, SR-UKF, CKF |
| Fading memory | `.with_fading_memory(γ)` | EKF, UKF, SR-UKF, CKF |
| Innovation gating | `update_gated(…, gate)` | EKF, UKF, SR-UKF, CKF |
| Iterated update | `update_iterated(…, max_iter, tol)` | EKF only |
| Symmetric update | `P − K·S·Kᵀ` (automatic) | UKF, CKF |
| NIS output | returned from every `update` | all filters |

### Cholesky with jitter

When Cholesky of the predicted covariance `P` or innovation covariance `S` fails due to accumulated floating-point error, the filter automatically retries with `P + ε·I` for ε ∈ {1e-9, 1e-7, 1e-5} before returning `CovarianceNotPD`. This prevents spurious failures on mathematically positive-definite matrices.

### Covariance floor

```rust
let mut ekf = Ekf::<f64, 2, 1>::new(x0, p0)
    .with_min_variance(1e-6);  // P[i,i] >= 1e-6 after every step
```

Clamps the diagonal of `P` after every predict and update. Prevents variance from degenerating to zero or going negative from repeated updates with very precise measurements.

### Fading memory

```rust
let mut ekf = Ekf::<f64, 2, 1>::new(x0, p0)
    .with_fading_memory(1.02);  // scale predicted covariance by 1.02
```

Scales the propagated covariance by `γ` before adding `Q`: predicted `P = γ · F P Fᵀ + Q`. Values `γ > 1` inflate uncertainty after prediction to compensate for unmodeled dynamics or model mismatch. Default `γ = 1.0` is the standard Kalman filter.

### Innovation gating

```rust
// Chi-squared 99% gate thresholds by measurement dimension:
// M=1: 6.63 | M=2: 9.21 | M=3: 11.34 | M=6: 16.81
let result = ekf.update_gated(&z, h, hj, &r, 9.21)?;
match result {
    None        => { /* measurement rejected as outlier */ }
    Some(nis)   => { /* accepted; nis = yᵀ S⁻¹ y */ }
}
```

Computes the Normalized Innovation Squared (NIS) before applying the state update. If `NIS > gate`, the update is skipped and `Ok(None)` is returned — the state and covariance are unchanged. Consistent measurements satisfy `NIS ~ χ²(M)`.

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

// Update step: returns NIS = yᵀ S⁻¹ y (chi-squared distributed under M dof)
let nis = ekf.update(
    &ColumnVector::from_column([0.12_f64]),  // measurement z
    |x| ColumnVector::from_column([x[(0, 0)]]),
    |_x| Matrix::new([[1.0_f64, 0.0]]),
    &r,
).unwrap();

// Access state and covariance
let x = ekf.state();       // &ColumnVector<f64, 2>
let p = ekf.covariance();  // &Matrix<f64, 2, 2>
```

The covariance update uses the **Joseph form** `P = (I−KH)P(I−KH)ᵀ + KRKᵀ` for numerical stability.

### Finite-Difference Jacobians

When analytic Jacobians are unavailable, use `predict_fd` / `update_fd`:

```rust
ekf.predict_fd(
    |x| ColumnVector::from_column([x[(0,0)] + dt*x[(1,0)], x[(1,0)]]),
    q.as_ref(),
);
let nis = ekf.update_fd(
    &ColumnVector::from_column([0.12]),
    |x| ColumnVector::from_column([x[(0,0)]]),
    &r,
).unwrap();
```

Finite differences use forward differences with step size `√ε` (≈ 1.5×10⁻⁸ for `f64`).

### Iterated EKF (IEKF)

For highly nonlinear measurement models (attitude-to-pixel projection, range/bearing), the IEKF re-linearizes at the current iterate rather than the predicted state:

```rust
// update_iterated re-linearizes at each step until ‖Δx‖ < tol
let nis = ekf.update_iterated(
    &z,
    |x| h(x),           // measurement model
    |x| hj(x),          // Jacobian — or use update_fd_iterated for FD
    &r,
    20,                  // max iterations
    1e-8,               // convergence tolerance ‖Δx‖
).unwrap();
```

The final covariance is updated with the Joseph form at the converged linearization point. For linear `h`, converges in one iteration and exactly reproduces the standard EKF.

## Unscented Kalman Filter (UKF)

Propagates 2N+1 sigma points through the nonlinear dynamics — no Jacobians needed.

```rust
use numeris::estimate::Ukf;
use numeris::{ColumnVector, Matrix};

let mut ukf = Ukf::<f64, 2, 1>::new(x0, p0);
// Default: alpha=1.0, beta=2.0, kappa=0.0 — all weights non-negative.
// For tighter sigma point spread, adjust alpha via with_params:
let mut ukf_tuned = Ukf::<f64, 2, 1>::with_params(x0, p0, 0.1, 2.0, 0.0);

ukf.predict(
    |x| ColumnVector::from_column([x[(0,0)] + dt*x[(1,0)], x[(1,0)]]),
    q.as_ref(),
).unwrap();

let nis = ukf.update(
    &ColumnVector::from_column([0.12]),
    |x| ColumnVector::from_column([x[(0,0)]]),
    &r,
).unwrap();
```

**Default sigma-point parameters (`alpha=1.0`, `beta=2.0`, `kappa=0.0`):** All weights are non-negative for any state dimension. The central sigma point weight `wm₀ = 0`, so the mean comes entirely from the 2N peripheral points. Reduce `alpha` toward 0.1–0.5 for tighter clustering (note: `alpha < ~0.52` with `kappa=0` gives negative central covariance weight `wc₀`).

The covariance update uses `P − K·S·Kᵀ` (manifestly symmetric PSD-subtracted form).

## Square-Root UKF (SR-UKF)

Propagates the Cholesky factor `S` (where `P = SSᵀ`) rather than `P` directly. Guarantees positive-definiteness of the covariance even in the presence of numerical round-off.

```rust
use numeris::estimate::SrUkf;
use numeris::{ColumnVector, Matrix};

// Construct from covariance P (Cholesky computed internally, with jitter fallback)
let mut srukf = SrUkf::<f64, 2, 1>::from_covariance(x0, p0).unwrap();

// Or from the Cholesky factor S directly
let s0 = p0.cholesky().unwrap().l_full();  // lower-triangular factor
let mut srukf2 = SrUkf::<f64, 2, 1>::new(x0, s0);

let nis = srukf.update(
    &ColumnVector::from_column([0.12]),
    |x| ColumnVector::from_column([x[(0,0)]]),
    &r,
).unwrap();

let p = srukf.covariance();  // reconstructed P = SSᵀ
```

After each update the covariance is updated as `P − K·S·Kᵀ` and then re-Choleskyized (with jitter fallback). The Cholesky factor `srukf.s` is always lower-triangular.

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

let nis = ckf.update(
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

Is your measurement model highly nonlinear (attitude→pixel, range/bearing)?
  → Iterated EKF (IEKF) or UKF

Do you need guaranteed positive-definite covariance in long runs?
  → SrUkf (propagates Cholesky factor directly)

Do you want zero tuning parameters?
  → CKF (2N points, equal weights)

Are you processing outlier-prone data (false star matches, bad sensors)?
  → Use update_gated with a chi-squared threshold

Is your process model imperfect (aerodynamic drag uncertainty, etc.)?
  → Use with_fading_memory(γ) with γ = 1.01–1.05

Are you doing offline post-processing with all data available?
  → RTS smoother (better estimates than forward-only filtering)

Are your measurements linear in the state?
  → BatchLsq (simple, no-alloc, optimal for linear-Gaussian problems)
```

## Normalized Innovation Squared (NIS)

Every `update` call returns `Result<T, EstimateError>` where `T` is the NIS:

```
NIS = yᵀ S⁻¹ y    where  y = z − h(x̄),  S = H P Hᵀ + R
```

Under a consistent filter, NIS is chi-squared distributed with `M` degrees of freedom. Use this for filter consistency monitoring and measurement gating:

| M | 95% threshold | 99% threshold |
|---|---|---|
| 1 | 3.84 | 6.63 |
| 2 | 5.99 | 9.21 |
| 3 | 7.81 | 11.34 |
| 6 | 12.59 | 16.81 |

A NIS consistently above the threshold indicates filter inconsistency (underestimated uncertainty or model error). A NIS consistently below indicates overestimated uncertainty.

## Error Handling

```rust
use numeris::estimate::EstimateError;

match ekf.update(&z, h, hj, &r) {
    Ok(nis)                                     => { /* success; nis ~ χ²(M) */ }
    Err(EstimateError::CovarianceNotPD)         => { /* P lost positive-definiteness (even with jitter) */ }
    Err(EstimateError::SingularInnovation)      => { /* S = HPHᵀ + R singular (even with jitter) */ }
    Err(EstimateError::CholdowndateFailed)      => { /* SR-UKF Cholesky downdate failed */ }
}
```

`CovarianceNotPD` and `SingularInnovation` are only returned after all three jitter levels (1e-9, 1e-7, 1e-5) fail — indicating a genuinely degenerate state.
