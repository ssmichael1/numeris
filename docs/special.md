# Special Functions

Mathematical special functions used throughout probability and statistics.

Requires the `special` Cargo feature:

```toml
numeris = { version = "0.2", features = ["special"] }
```

All functions work with both `f32` and `f64`, are no-std compatible, and have no heap allocation.

## Gamma Function

The gamma function `Γ(x)` generalizes the factorial: `Γ(n) = (n-1)!` for positive integers.

Implemented via the Lanczos approximation (g=7, n=9 coefficients) with the reflection formula for negative arguments. Exact factorial lookup table for integer arguments 1–21.

```rust
use numeris::special::{gamma, lgamma};

// Gamma function
let g = gamma(5.0_f64);          // 4! = 24.0
let g_half = gamma(0.5_f64);     // √π ≈ 1.7724538509
let g_neg = gamma(-0.5_f64);     // -2√π ≈ -3.5449077018

// Log-gamma: more numerically stable for large arguments
let lg = lgamma(100.0_f64);      // log(99!) ≈ 359.13
let lg2 = lgamma(0.5_f64);       // log(√π) ≈ 0.5724

// Poles: gamma(0), gamma(-1), gamma(-2), ... → ±∞
assert!(gamma(0.0_f64).is_infinite());
```

## Digamma Function

The digamma function `ψ(x) = d/dx ln Γ(x)` — the logarithmic derivative of the gamma function.

Implemented via recurrence (shifts argument to x ≥ 6) + 7-term asymptotic expansion. Reflection formula for negative arguments.

```rust
use numeris::special::digamma;

let psi1 = digamma(1.0_f64);   // -γ ≈ -0.5772156649 (Euler-Mascheroni constant)
let psi2 = digamma(2.0_f64);   //  1 - γ ≈ 0.4227843351
let psi_neg = digamma(-0.5_f64); // ψ(-1/2) ≈ 0.03648997...

// Poles at non-positive integers → NaN
assert!(digamma(0.0_f64).is_nan());
```

## Beta Function

`B(a, b) = Γ(a)Γ(b) / Γ(a+b)`. Implemented via `lgamma` delegation.

```rust
use numeris::special::{beta, lbeta};

let b = beta(2.0_f64, 3.0_f64);   // 1/12 ≈ 0.08333...
let lb = lbeta(2.0_f64, 3.0_f64); // log(1/12) ≈ -2.4849...

// B(1/2, 1/2) = π (related to the arc-sine distribution)
let b_half = beta(0.5_f64, 0.5_f64); // ≈ 3.14159...
```

## Regularized Incomplete Gamma

`P(a, x) = γ(a, x) / Γ(a)` (lower) and `Q(a, x) = Γ(a, x) / Γ(a)` (upper).

- Series expansion for `x < a + 1` (fast convergence near zero)
- Lentz continued fraction for `x ≥ a + 1` (fast convergence in the tail)
- Maximum 200 iterations; returns `SpecialError::ConvergenceFailure` if not converged

```rust
use numeris::special::{gamma_inc, gamma_inc_upper};

// P(a, x): probability that a Gamma(a,1) random variable ≤ x
let p = gamma_inc(2.0_f64, 1.0_f64);          // ≈ 0.2642
let p_full = gamma_inc(2.0_f64, f64::INFINITY); // 1.0

// Q(a, x) = 1 - P(a, x)
let q = gamma_inc_upper(2.0_f64, 1.0_f64);    // ≈ 0.7358

// Domain: a > 0, x ≥ 0
```

## Regularized Incomplete Beta

`I_x(a, b)`: regularized incomplete beta function. Used in the CDFs of the Beta, F, and Student's t distributions.

Implemented via Lentz continued fraction with symmetry relation `I_x(a,b) = 1 - I_{1-x}(b,a)`.

```rust
use numeris::special::betainc;

// I_x(a, b): CDF of Beta(a, b) at x
let i = betainc(0.5_f64, 2.0_f64, 3.0_f64);  // I_{0.5}(2, 3) ≈ 0.6875

// Boundary cases
assert_eq!(betainc(0.0_f64, 2.0_f64, 3.0_f64), 0.0);
assert_eq!(betainc(1.0_f64, 2.0_f64, 3.0_f64), 1.0);
```

## Error Function

`erf(x)` and `erfc(x) = 1 - erf(x)`, implemented via the regularized incomplete gamma function for numerical stability:

```
erf(x)  = sign(x) · P(1/2, x²)
erfc(x) = Q(1/2, x²)   for x ≥ 0
```

Using `erfc` for large positive `x` avoids catastrophic cancellation in `1 - erf(x)`.

```rust
use numeris::special::{erf, erfc};

let e  = erf(1.0_f64);         // ≈ 0.8427007929
let ec = erfc(1.0_f64);        // ≈ 0.1572992071
assert!((e + ec - 1.0).abs() < 1e-15);

// Symmetry
assert_eq!(erf(-1.0_f64), -erf(1.0_f64));

// Tails
let e3 = erfc(3.0_f64);       // ≈ 2.21e-5 (complement is more accurate than 1-erf)
let e6 = erfc(6.0_f64);       // ≈ 2.15e-17
```

## Float Type Support

All functions work with both `f32` and `f64`:

```rust
use numeris::special::{gamma, erf};

let g32 = gamma(5.0_f32);     // f32 result
let e64 = erf(1.0_f64);       // f64 result
```

## Error Handling

```rust
use numeris::special::SpecialError;

match gamma_inc(-1.0_f64, 1.0_f64) {  // a must be > 0
    Err(SpecialError::DomainError)        => { /* invalid argument */ }
    Err(SpecialError::ConvergenceFailure) => { /* series/CF didn't converge (200 iterations) */ }
    Ok(p) => { /* success */ }
}
```

## Function Reference

| Function | Signature | Notes |
|---|---|---|
| `gamma(x)` | `T → T` | Γ(x); poles at 0, -1, -2, … → ±∞ |
| `lgamma(x)` | `T → T` | ln\|Γ(x)\|; use for large arguments |
| `digamma(x)` | `T → T` | ψ(x) = d/dx ln Γ(x); poles → NaN |
| `beta(a, b)` | `T, T → T` | B(a,b) = Γ(a)Γ(b)/Γ(a+b) |
| `lbeta(a, b)` | `T, T → T` | ln B(a,b); more stable |
| `gamma_inc(a, x)` | `T, T → Result<T>` | Regularized lower incomplete gamma P(a,x) |
| `gamma_inc_upper(a, x)` | `T, T → Result<T>` | Regularized upper incomplete gamma Q(a,x) |
| `betainc(x, a, b)` | `T, T, T → T` | Regularized incomplete beta I_x(a,b) |
| `erf(x)` | `T → T` | Error function |
| `erfc(x)` | `T → T` | Complementary error function 1 - erf(x) |
