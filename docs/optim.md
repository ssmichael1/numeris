# Optimization

Root finding, unconstrained minimization, and nonlinear least squares.

Requires the `optim` Cargo feature:

```toml
numeris = { version = "0.2", features = ["optim"] }
```

## Algorithm Summary

| Algorithm | Function | Use case |
|---|---|---|
| Brent's method | `brent` | Bracketed scalar root finding — superlinear convergence |
| Newton 1D | `newton_1d` | Scalar root finding with analytic derivative |
| BFGS | `minimize_bfgs` | Unconstrained smooth minimization |
| Gauss-Newton | `least_squares_gn` | Nonlinear least squares (QR-based, full-rank Jacobian) |
| Levenberg-Marquardt | `least_squares_lm` | Nonlinear least squares (damped, more robust) |

## Root Finding

### Brent's Method

Bracketed root finding — guaranteed to converge given a sign change.

```rust
use numeris::optim::{brent, RootSettings};

// Solve x² - 2 = 0 on [0, 2]
let result = brent(
    |x| x * x - 2.0,
    0.0_f64, 2.0,
    &RootSettings::default(),
).unwrap();

assert!((result.x - std::f64::consts::SQRT_2).abs() < 1e-12);
println!("root = {}, f(root) = {}, iters = {}", result.x, result.fx, result.iterations);
```

### Newton 1D

Scalar Newton's method with analytic derivative — quadratic convergence near the root.

```rust
use numeris::optim::{newton_1d, RootSettings};

// Solve cos(x) = 0 near x = 1.5
let result = newton_1d(
    |x| x.cos(),
    |x| -x.sin(),
    1.5_f64,
    &RootSettings::default(),
).unwrap();

assert!((result.x - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
```

### Settings

```rust
use numeris::optim::RootSettings;

let settings = RootSettings {
    tol:       1e-12,   // convergence tolerance on |f(x)| and bracket width
    max_iter:  200,
    ..RootSettings::default()
};
```

## BFGS Minimization

Quasi-Newton unconstrained minimization with Armijo backtracking line search. Requires both a function and its gradient.

```rust
use numeris::optim::{minimize_bfgs, BfgsSettings};
use numeris::Vector;

// Minimize the Rosenbrock function: (1-x)² + 100(y-x²)²
let result = minimize_bfgs(
    |v: &Vector<f64, 2>| {
        let x = v[0]; let y = v[1];
        (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
    },
    |v: &Vector<f64, 2>| {
        let x = v[0]; let y = v[1];
        Vector::from_array([
            -2.0 * (1.0 - x) - 400.0 * x * (y - x * x),
             200.0 * (y - x * x),
        ])
    },
    &Vector::from_array([-1.0, 1.0]),   // initial guess
    &BfgsSettings::default(),
).unwrap();

assert!((result.x[0] - 1.0).abs() < 1e-5);
assert!((result.x[1] - 1.0).abs() < 1e-5);
```

### Using Finite-Difference Gradient

When an analytic gradient is unavailable:

```rust
use numeris::optim::{minimize_bfgs, finite_difference_gradient, BfgsSettings};
use numeris::Vector;

let f = |v: &Vector<f64, 2>| (v[0] - 1.0).powi(2) + (v[1] - 2.0).powi(2);
let grad = |v: &Vector<f64, 2>| finite_difference_gradient(f, v);

let result = minimize_bfgs(f, grad, &Vector::from_array([0.0, 0.0]), &BfgsSettings::default()).unwrap();
```

### BfgsSettings

```rust
use numeris::optim::BfgsSettings;

let settings = BfgsSettings {
    grad_tol:  1e-8,    // gradient norm convergence criterion
    max_iter:  1000,
    ..BfgsSettings::default()
};
```

## Gauss-Newton Least Squares

QR-based Gauss-Newton for nonlinear least squares. Works best when the Jacobian is full rank and the residual is small at the solution.

```rust
use numeris::optim::{least_squares_gn, GnSettings};
use numeris::{Matrix, Vector};

// Fit y = a * exp(b * x) to noisy data
let t = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
let y = [2.0_f64, 2.7, 3.65, 4.95, 6.7];

let result = least_squares_gn(
    // Residual function: r_i = model(x) - y_i
    |x: &Vector<f64, 2>| {
        let mut r = Vector::<f64, 5>::zeros();
        for i in 0..5 { r[i] = x[0] * (x[1] * t[i]).exp() - y[i]; }
        r
    },
    // Jacobian ∂r/∂x
    |x: &Vector<f64, 2>| {
        let mut j = Matrix::<f64, 5, 2>::zeros();
        for i in 0..5 {
            let e = (x[1] * t[i]).exp();
            j[(i, 0)] = e;
            j[(i, 1)] = x[0] * t[i] * e;
        }
        j
    },
    &Vector::from_array([1.0, 0.5]),    // initial guess [a, b]
    &GnSettings::default(),
).unwrap();

println!("a = {:.4}, b = {:.4}", result.x[0], result.x[1]);
println!("cost = {:.6}", result.cost);
```

## Levenberg-Marquardt

Damped Gauss-Newton with adaptive regularization — more robust than pure GN when the Jacobian is ill-conditioned or the initial guess is far from the solution.

```rust
use numeris::optim::{least_squares_lm, LmSettings};
use numeris::{Matrix, Vector};

let t = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
let y = [2.0_f64, 2.7, 3.65, 4.95, 6.7];

let result = least_squares_lm(
    |x: &Vector<f64, 2>| {
        let mut r = Vector::<f64, 5>::zeros();
        for i in 0..5 { r[i] = x[0] * (x[1] * t[i]).exp() - y[i]; }
        r
    },
    |x: &Vector<f64, 2>| {
        let mut j = Matrix::<f64, 5, 2>::zeros();
        for i in 0..5 {
            let e = (x[1] * t[i]).exp();
            j[(i, 0)] = e;
            j[(i, 1)] = x[0] * t[i] * e;
        }
        j
    },
    &Vector::from_array([1.0, 0.1]),
    &LmSettings::default(),
).unwrap();

assert!(result.cost < 0.1);
```

### LmSettings

```rust
use numeris::optim::LmSettings;

let settings = LmSettings {
    grad_tol:  1e-8,      // ∥Jᵀr∥ convergence criterion
    step_tol:  1e-8,      // step size convergence criterion
    cost_tol:  1e-8,      // cost reduction convergence criterion
    lambda0:   1e-3,      // initial damping factor
    max_iter:  200,
    ..LmSettings::default()
};
```

## Finite Difference Utilities

```rust
use numeris::optim::{finite_difference_gradient, finite_difference_jacobian};
use numeris::{Matrix, Vector};

// Gradient of a scalar function ℝⁿ → ℝ
let f = |x: &Vector<f64, 3>| x[0].powi(2) + x[1].powi(2) + x[2].powi(2);
let x = Vector::from_array([1.0_f64, 2.0, 3.0]);
let g: Vector<f64, 3> = finite_difference_gradient(f, &x);
// g ≈ [2, 4, 6]

// Jacobian of a vector function ℝⁿ → ℝᵐ  (M residuals, N parameters)
let r = |x: &Vector<f64, 2>| Vector::from_array([x[0].powi(2) - 1.0, x[1].powi(2) - 4.0]);
let x2 = Vector::from_array([1.0_f64, 2.0]);
let j: Matrix<f64, 2, 2> = finite_difference_jacobian(r, &x2);
```

## Result Types

```rust
use numeris::optim::{OptimResult, LsqResult};

// Scalar root finding
// result.x         — solution scalar
// result.fx        — f(x) at solution
// result.iterations — iterations taken

// BFGS minimization
// result.x         — solution Vector<T, N>
// result.fx        — f(x) at solution
// result.grad_norm — ∥∇f∥ at solution
// result.iterations — iterations taken

// Least squares
// result.x         — solution Vector<T, N>
// result.cost      — ½‖r‖² at solution
// result.iterations — iterations taken
```

## Error Handling

```rust
use numeris::optim::OptimError;

match least_squares_lm(r, j, &x0, &settings) {
    Ok(result)                          => { /* converged */ }
    Err(OptimError::MaxIterExceeded)    => { /* increase max_iter or relax tolerances */ }
    Err(OptimError::LineSearchFailed)   => { /* BFGS line search failure */ }
    Err(OptimError::SingularJacobian)   => { /* Gauss-Newton needs better initial guess */ }
    Err(e) => { /* other */ }
}
```
