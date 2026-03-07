# Interpolation

Four interpolation methods plus bilinear 2D interpolation. Each method has a fixed-size variant (stack-allocated, const-generic number of knots) and a dynamic variant (heap-allocated, requires `alloc`).

Requires the `interp` Cargo feature:

```toml
numeris = { version = "0.2", features = ["interp"] }
```

## Method Summary

| Method | Struct | Continuity | Best for |
|---|---|---|---|
| Linear | `LinearInterp` | C⁰ | Fast, piecewise linear data |
| Hermite | `HermiteInterp` | C¹ | Data with known derivatives |
| Lagrange | `LagrangeInterp` | Cⁿ⁻¹ | Global polynomial through all points |
| Cubic Spline | `CubicSpline` | C² | Smooth curves, natural boundary conditions |
| Bilinear | `BilinearInterp` | C⁰ | 2D rectangular grid data |

Dynamic variants: `DynLinearInterp`, `DynHermiteInterp`, `DynLagrangeInterp`, `DynCubicSpline`, `DynBilinearInterp`.

## Linear Interpolation

Piecewise linear interpolation between knots. O(log N) per evaluation (binary search).

Out-of-bounds queries extrapolate using the nearest segment.

```rust
use numeris::interp::LinearInterp;

// Fixed-size: 4 knots
let xs = [0.0_f64, 1.0, 2.0, 3.0];
let ys = [0.0_f64, 1.0, 0.0, 1.0];
let interp = LinearInterp::new(xs, ys).unwrap();

let y = interp.eval(1.5);  // 0.5
```

Dynamic variant:

```rust
use numeris::interp::DynLinearInterp;

let xs = vec![0.0_f64, 1.0, 2.0, 3.0];
let ys = vec![0.0_f64, 1.0, 0.0, 1.0];
let interp = DynLinearInterp::new(xs, ys).unwrap();
let y = interp.eval(1.5);
```

## Hermite Interpolation

Cubic Hermite interpolation with user-supplied derivatives at each knot. Guarantees C¹ continuity.

```rust
use numeris::interp::HermiteInterp;

let xs = [0.0_f64, 1.0, 2.0, 3.0];
let ys = [0.0_f64, 1.0, 0.0, 1.0];
// Derivatives at each knot (e.g., from finite differences or analytic formula)
let ds = [1.0_f64, 0.0, -1.0, 0.0];

let interp = HermiteInterp::new(xs, ys, ds).unwrap();

let y  = interp.eval(1.5);
let dy = interp.eval_deriv(1.5);  // first derivative
```

!!! tip "When to use Hermite"
    Use Hermite when you have physical knowledge of derivatives — e.g., velocity data from an accelerometer alongside position data. It avoids Runge's phenomenon better than high-degree Lagrange polynomials.

## Barycentric Lagrange Interpolation

Global polynomial interpolation through all N knots. O(N²) setup (barycentric weights), O(N) per evaluation. C^(N-1) continuity.

```rust
use numeris::interp::LagrangeInterp;

let xs = [0.0_f64, 0.5, 1.0, 1.5, 2.0];
let ys = [0.0_f64, 0.479, 0.841, 0.997, 0.909];  // ≈ sin(x)

let interp = LagrangeInterp::new(xs, ys).unwrap();

let y  = interp.eval(0.75);
let dy = interp.eval_deriv(0.75);  // first derivative
```

!!! warning "Runge's phenomenon"
    High-degree Lagrange polynomials on uniformly-spaced nodes can oscillate wildly near the boundaries. Prefer cubic splines for smooth data with many knots, or use Chebyshev nodes for Lagrange when possible.

## Natural Cubic Spline

Piecewise cubic with C² continuity (continuous second derivative). Natural boundary conditions (`S''(x₀) = S''(xₙ) = 0`). Solved via Thomas algorithm (tridiagonal, O(N)).

```rust
use numeris::interp::CubicSpline;

let xs = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
let ys = [0.0_f64, 1.0, 0.0, 1.0, 0.0];

let spline = CubicSpline::new(xs, ys).unwrap();

let y  = spline.eval(1.5);       // smooth interpolation
let dy = spline.eval_deriv(1.5); // first derivative
```

Dynamic variant:

```rust
use numeris::interp::DynCubicSpline;

let xs = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
let ys = vec![0.0_f64, 1.0, 0.0, 1.0, 0.0];
let spline = DynCubicSpline::new(xs, ys).unwrap();
let y = spline.eval(2.7);
```

## Bilinear Interpolation (2D)

For data on a rectangular grid. Interpolates z = f(x, y) within each cell using the four corner values. Out-of-bounds queries extrapolate using the nearest boundary cell.

Input is **row-major**: `zs[iy][ix]` is the value at `(xs[ix], ys[iy])`, matching the natural "row = fixed y" convention. Internally stored column-major.

```rust
use numeris::interp::BilinearInterp;

// 3×2 grid: z = x + y
let xs = [0.0_f64, 1.0, 2.0];
let ys = [0.0, 1.0];

// Row-major: each inner array is a row at fixed y
let zs = [
    [0.0, 1.0, 2.0],  // y = 0: z = x + 0
    [1.0, 2.0, 3.0],  // y = 1: z = x + 1
];

let interp = BilinearInterp::new(xs, ys, zs).unwrap();
assert!((interp.eval(0.5, 0.5) - 1.0).abs() < 1e-14);  // 0.5 + 0.5 = 1.0
assert!((interp.eval(1.5, 0.0) - 1.5).abs() < 1e-14);  // 1.5 + 0.0 = 1.5
```

Dynamic variant accepts `Vec` data:

```rust
use numeris::interp::DynBilinearInterp;

let xs = vec![0.0_f64, 1.0, 2.0];
let ys = vec![0.0, 1.0];
let zs = vec![
    vec![0.0, 1.0, 2.0],  // y = 0
    vec![1.0, 2.0, 3.0],  // y = 1
];
let interp = DynBilinearInterp::new(xs, ys, zs).unwrap();
assert!((interp.eval(0.5, 0.5) - 1.0).abs() < 1e-14);
```

The dynamic variant also supports pre-flattened column-major data via `DynBilinearInterp::from_slice()`.

## Error Handling

```rust
use numeris::interp::InterpError;

match CubicSpline::new(xs, ys) {
    Ok(spline) => { /* use spline */ }
    Err(InterpError::NotSorted)         => { /* xs must be strictly increasing */ }
    Err(InterpError::LengthMismatch)    => { /* xs.len() != ys.len() */ }
    Err(InterpError::TooFewPoints)       => { /* need at least 2 knots */ }
}
```

## Method Comparison

--8<-- "includes/plot_interp.html"

```rust
use numeris::interp::{LinearInterp, HermiteInterp, LagrangeInterp, CubicSpline};

let xs = [0.0_f64, 1.0, 2.0, 3.0];
let ys = [0.0_f64, 0.841, 0.909, 0.141];  // ≈ sin(x)
let ds = [1.0_f64, 0.540, -0.416, -0.990]; // ≈ cos(x) (derivatives)

let linear  = LinearInterp::new(xs, ys).unwrap();
let hermite = HermiteInterp::new(xs, ys, ds).unwrap();
let lagrange = LagrangeInterp::new(xs, ys).unwrap();
let spline  = CubicSpline::new(xs, ys).unwrap();

let x_query = 1.5;
println!("linear   = {:.6}", linear.eval(x_query));   // piecewise linear
println!("hermite  = {:.6}", hermite.eval(x_query));  // smooth, uses derivatives
println!("lagrange = {:.6}", lagrange.eval(x_query)); // global polynomial
println!("spline   = {:.6}", spline.eval(x_query));   // C² cubic spline
// All should be close to sin(1.5) ≈ 0.997
```
