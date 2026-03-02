# ODE Integration

numeris provides fixed-step and adaptive ODE integrators for non-stiff and stiff systems. All solvers work on `Vector<T, N>` state vectors and closure-based dynamics functions.

Requires the `ode` feature (default).

## Fixed-Step RK4

Classic 4th-order Runge-Kutta — no error estimation, fixed step size.

```rust
use numeris::ode::{rk4, rk4_step};
use numeris::Vector;

let f = |_t: f64, y: &Vector<f64, 2>| {
    Vector::from_array([y[1], -y[0]])  // harmonic oscillator
};

let y0 = Vector::from_array([1.0_f64, 0.0]);
let dt = 0.01;
let t_end = 2.0 * std::f64::consts::PI;

// Integrate from t=0 to t=t_end
let sol = rk4(0.0, t_end, &y0, f, dt);

// Or take a single step
let y1 = rk4_step(0.0, &y0, f, dt);
```

RK4 is fully no-std and no-alloc — suitable for embedded real-time control loops.

## Adaptive Solvers

Seven adaptive Runge-Kutta solvers via the `RKAdaptive` trait. All use embedded error estimation and a PI step-size controller (Söderlind & Wang 2006) for automatic step adjustment.

```rust
use numeris::ode::{RKAdaptive, RKTS54, AdaptiveSettings};
use numeris::Vector;

let y0 = Vector::from_array([1.0_f64, 0.0]);
let tau = 2.0 * std::f64::consts::PI;

let sol = RKTS54::integrate(
    0.0, tau, &y0,
    |_t, y| Vector::from_array([y[1], -y[0]]),
    &AdaptiveSettings::default(),
).unwrap();

println!("y(2π) = {:?}", sol.y);       // final state
println!("steps taken = {}", sol.steps); // number of accepted steps
```

### Solver Table

| Solver | Stages | Order | FSAL | Interpolant | Best for |
|---|---|---|---|---|---|
| `RKF45` | 6 | 5(4) | no | — | Classic baseline |
| `RKTS54` | 7 | 5(4) | yes | 4th degree | General purpose |
| `RKV65` | 10 | 6(5) | no | 6th degree | Moderate accuracy |
| `RKV87` | 17 | 8(7) | no | 7th degree | High accuracy |
| `RKV98` | 21 | 9(8) | no | 8th degree | Very high accuracy |
| `RKV98NoInterp` | 16 | 9(8) | no | — | Very high acc., no dense output |
| `RKV98Efficient` | 26 | 9(8) | no | 9th degree | Max accuracy + dense output |

**FSAL** (First Same As Last): the last function evaluation of one step is reused as the first of the next, saving one function evaluation per step.

### AdaptiveSettings

```rust
use numeris::ode::AdaptiveSettings;

let settings = AdaptiveSettings {
    rtol: 1e-8,       // relative tolerance (default 1e-6)
    atol: 1e-10,      // absolute tolerance (default 1e-9)
    h0:   Some(0.01), // initial step size (auto if None)
    hmax: Some(1.0),  // maximum step size (unlimited if None)
    max_steps: 100_000, // step limit
    ..AdaptiveSettings::default()
};
```

### Solution struct

```rust
let sol = RKTS54::integrate(0.0, 1.0, &y0, f, &settings).unwrap();

let y_final = &sol.y;       // final state Vector<T, N>
let t_final =  sol.t;       // final time (= t_end if successful)
let n_steps  = sol.steps;   // number of accepted steps
let n_reject = sol.rejected; // number of rejected steps
```

## Dense Output (Interpolation)

Solvers with an interpolant can return intermediate values at arbitrary times without re-integrating. Requires `std` feature (uses `Vec` internally).

```rust
use numeris::ode::{RKAdaptive, RKTS54, AdaptiveSettings};
use numeris::Vector;

let y0 = Vector::from_array([1.0_f64, 0.0]);

let mut sol = RKTS54::integrate_dense(
    0.0, 6.28, &y0,
    |_t, y| Vector::from_array([y[1], -y[0]]),
    &AdaptiveSettings::default(),
).unwrap();

// Interpolate at arbitrary time point
let y_at_pi = sol.interpolate(std::f64::consts::PI);
// y[0] ≈ -1.0 (cosine at π)
```

## Stiff Solver: RODAS4

For stiff ODEs (chemical kinetics, circuit simulation, orbital mechanics with drag), use `RODAS4` — an L-stable linearly-implicit Rosenbrock method. It solves linear systems involving the Jacobian instead of nonlinear Newton iterations, making it suitable for very stiff problems without tuning.

```rust
use numeris::ode::{Rosenbrock, RODAS4, AdaptiveSettings};
use numeris::{Vector, Matrix};

// Stiff decay: y' = -1000y,  y(0) = 1
let y0 = Vector::from_array([1.0_f64]);

// With user-supplied Jacobian
let sol = RODAS4::integrate(
    0.0, 0.01, &y0,
    |_t, y| Vector::from_array([-1000.0 * y[0]]),
    |_t, _y| Matrix::new([[-1000.0_f64]]),
    &AdaptiveSettings::default(),
).unwrap();
assert!((sol.y[0] - (-10.0_f64).exp()).abs() < 1e-8);

// With automatic finite-difference Jacobian (no need to supply ∂f/∂y)
let sol2 = RODAS4::integrate_auto(
    0.0, 0.01, &y0,
    |_t, y| Vector::from_array([-1000.0 * y[0]]),
    &AdaptiveSettings::default(),
).unwrap();
```

### Van der Pol oscillator (μ = 1000)

```rust
use numeris::ode::{Rosenbrock, RODAS4, AdaptiveSettings};
use numeris::{Vector, Matrix};

let mu = 1000.0_f64;
let y0 = Vector::from_array([2.0_f64, 0.0]);

let sol = RODAS4::integrate(
    0.0, 3000.0, &y0,
    |_t, y| Vector::from_array([y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]]),
    |_t, y| Matrix::new([
        [0.0,                                    1.0],
        [-2.0 * mu * y[0] * y[1] - 1.0, mu * (1.0 - y[0] * y[0])],
    ]),
    &AdaptiveSettings::default(),
).unwrap();
```

### RODAS4 Properties

| Property | Value |
|---|---|
| Stages | 6 |
| Order | 4(3) embedded pair |
| L-stable | Yes (no oscillation for arbitrarily large step sizes) |
| Stiffly accurate | Yes |
| Jacobian | User-supplied or auto finite-difference |

## Error Handling

```rust
use numeris::ode::OdeError;

match RKTS54::integrate(0.0, 1.0, &y0, f, &settings) {
    Ok(sol) => { /* success */ }
    Err(OdeError::MaxStepsExceeded) => { /* increase max_steps or rtol/atol */ }
    Err(OdeError::StepSizeTooSmall) => { /* likely a stiff problem — use RODAS4 */ }
    Err(OdeError::SingularJacobian) => { /* RODAS4 specific */ }
}
```
