# Numerical Quadrature

Numerical integration (quadrature) with four methods: Gauss-Legendre, adaptive Simpson, composite trapezoid, and composite Simpson. All methods are no-alloc and no-std compatible.

Requires the `quad` Cargo feature:

```toml
numeris = { version = "0.2", features = ["quad"] }
```

## Method Summary

| Method | Function | Best for |
|---|---|---|
| Gauss-Legendre | `gauss_legendre::<T, N>` | Smooth functions, high accuracy with few evaluations |
| Adaptive Simpson | `adaptive_simpson` | Unknown smoothness, automatic error control |
| Composite Trapezoid | `trapezoid` | Simple, predictable, linear-exact |
| Composite Simpson | `simpson` | Higher-order composite rule, cubic-exact |

## Gauss-Legendre Quadrature

N-point Gauss-Legendre quadrature on `[a, b]`. Exact for polynomials of degree ≤ 2N − 1. The number of points N is a const generic parameter.

Supported orders: N = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20.

```rust
use numeris::quad::gauss_legendre;

// 3-point GL: exact for degree ≤ 5
let result = gauss_legendre::<f64, 3>(|x| x * x, 0.0, 1.0);
assert!((result - 1.0 / 3.0).abs() < 1e-15);

// 10-point GL: excellent accuracy for smooth functions
let pi = core::f64::consts::PI;
let result = gauss_legendre::<f64, 10>(|x| x.sin(), 0.0, pi);
assert!((result - 2.0).abs() < 1e-14);

// 20-point GL: exact for degree ≤ 39
let result = gauss_legendre::<f64, 20>(|x| x.powi(38), -1.0, 1.0);
assert!((result - 2.0 / 39.0).abs() < 1e-10);
```

Reversed limits produce a negated result (∫ₐᵇ = −∫ᵦₐ):

```rust
use numeris::quad::gauss_legendre;

let forward = gauss_legendre::<f64, 5>(|x| x * x, 0.0, 1.0);
let reverse = gauss_legendre::<f64, 5>(|x| x * x, 1.0, 0.0);
assert!((forward + reverse).abs() < 1e-15);
```

## Adaptive Simpson

Automatic subdivision with error control. Uses an explicit stack (no recursion) for no-std compatibility. Maximum subdivision depth is 50.

```rust
use numeris::quad::adaptive_simpson;

// Integrate sin(x) from 0 to pi with tolerance 1e-12
let result = adaptive_simpson(|x: f64| x.sin(), 0.0, core::f64::consts::PI, 1e-12).unwrap();
assert!((result - 2.0).abs() < 1e-12);

// Works with less smooth functions too
let result = adaptive_simpson(|x: f64| 1.0 / x, 1.0, core::f64::consts::E, 1e-12).unwrap();
assert!((result - 1.0).abs() < 1e-12);  // ln(e) - ln(1) = 1

// Gaussian integral: ∫₀⁶ exp(-x²) dx ≈ √π/2
let exact = core::f64::consts::PI.sqrt() / 2.0;
let result = adaptive_simpson(|x: f64| (-x * x).exp(), 0.0, 6.0, 1e-10).unwrap();
assert!((result - exact).abs() < 1e-10);
```

## Composite Trapezoid

Composite trapezoidal rule with `n` equally-spaced subintervals. Exact for linear functions. Error is O(h²).

```rust
use numeris::quad::trapezoid;

// Exact for linear functions (any n)
let result = trapezoid(|x: f64| 3.0 * x + 2.0, 0.0, 1.0, 1);
assert!((result - 3.5).abs() < 1e-15);

// Converges for smooth functions with enough subintervals
let result = trapezoid(|x: f64| x.sin(), 0.0, core::f64::consts::PI, 10000);
assert!((result - 2.0).abs() < 1e-7);
```

## Composite Simpson

Composite Simpson's 1/3 rule with `n` subintervals (`n` must be even). Exact for cubic polynomials. Error is O(h⁴).

```rust
use numeris::quad::simpson;

// Exact for cubics (any even n)
let result = simpson(|x: f64| x * x * x, 0.0, 1.0, 2);
assert!((result - 0.25).abs() < 1e-15);

// Higher accuracy than trapezoid for the same n
let result = simpson(|x: f64| x.exp(), 0.0, 1.0, 200);
let exact = 1.0_f64.exp() - 1.0;
assert!((result - exact).abs() < 1e-10);
```

## f32 Support

All methods work with `f32`:

```rust
use numeris::quad::{gauss_legendre, adaptive_simpson, trapezoid, simpson};

let gl = gauss_legendre::<f32, 5>(|x| x * x, 0.0_f32, 1.0_f32);
assert!((gl - 1.0 / 3.0_f32).abs() < 1e-6);

let a = adaptive_simpson(|x: f32| x.sin(), 0.0_f32, core::f32::consts::PI, 1e-6_f32).unwrap();
assert!((a - 2.0_f32).abs() < 1e-6);
```

## Error Handling

```rust
use numeris::quad::{adaptive_simpson, QuadError};

// Tolerance must be positive
let r = adaptive_simpson(|x: f64| x, 0.0, 1.0, 0.0);
assert_eq!(r, Err(QuadError::InvalidInput));

let r = adaptive_simpson(|x: f64| x, 0.0, 1.0, -1.0);
assert_eq!(r, Err(QuadError::InvalidInput));
```

## Choosing a Method

- **Gauss-Legendre** — best for smooth functions when you can choose the order upfront. 10-point GL gives ~14 digits for well-behaved integrands. No iteration overhead.
- **Adaptive Simpson** — best when you need guaranteed accuracy but don't know the integrand's smoothness. Automatically concentrates evaluations where the function changes rapidly.
- **Composite Simpson** — good middle ground for fixed-budget integration. O(h⁴) convergence.
- **Composite Trapezoid** — simplest rule. Use when the integrand is only C⁰ or when implementing higher-level adaptive schemes.
