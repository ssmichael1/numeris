# Statistics

Ten probability distributions, each implementing `ContinuousDistribution<T>` or `DiscreteDistribution<T>` traits.

Requires the `stats` Cargo feature (implies `special`):

```toml
numeris = { version = "0.2", features = ["stats"] }
```

All distributions work with `f32` and `f64`, are no-std compatible, and have no heap allocation.

## Distribution Traits

```rust
pub trait ContinuousDistribution<T> {
    fn pdf(&self, x: T) -> T;       // probability density function
    fn cdf(&self, x: T) -> T;       // cumulative distribution function
    fn mean(&self) -> T;
    fn variance(&self) -> T;
    fn std_dev(&self) -> T;
}

pub trait DiscreteDistribution<T> {
    fn pmf(&self, k: u64) -> T;     // probability mass function P(X = k)
    fn cdf(&self, k: u64) -> T;     // cumulative P(X ≤ k)
    fn mean(&self) -> T;
    fn variance(&self) -> T;
    fn std_dev(&self) -> T;
}
```

## Continuous Distributions

### Normal (Gaussian)

`Normal<T> { mean: T, std_dev: T }`

```rust
use numeris::stats::Normal;

let n = Normal::new(0.0_f64, 1.0).unwrap();  // standard normal N(0,1)
let p = n.pdf(0.0);      // 1/√(2π) ≈ 0.3989
let c = n.cdf(1.645);    // ≈ 0.95 (95th percentile)

let n2 = Normal::new(100.0_f64, 15.0).unwrap(); // IQ distribution
let prob_above_130 = 1.0 - n2.cdf(130.0);       // P(IQ > 130) ≈ 2.3%
```

### Uniform

`Uniform<T> { a: T, b: T }` — continuous uniform on [a, b].

```rust
use numeris::stats::Uniform;

let u = Uniform::new(0.0_f64, 1.0).unwrap();
let p = u.pdf(0.5);      // 1.0 (constant density)
let c = u.cdf(0.3);      // 0.3
assert_eq!(u.mean(), 0.5);
assert!((u.variance() - 1.0/12.0).abs() < 1e-10);
```

### Exponential

`Exponential<T> { rate: T }` — rate λ (mean = 1/λ).

```rust
use numeris::stats::Exponential;

let e = Exponential::new(2.0_f64).unwrap();  // λ=2, mean=0.5
let p = e.pdf(1.0);      // 2·e⁻² ≈ 0.2707
let c = e.cdf(1.0);      // 1 - e⁻² ≈ 0.8647
assert!((e.mean() - 0.5).abs() < 1e-12);
```

### Gamma

`Gamma<T> { shape: T, rate: T }` — shape α, rate β (mean = α/β).

The chi-squared and exponential distributions are special cases.

```rust
use numeris::stats::Gamma;

let g = Gamma::new(2.0_f64, 1.0).unwrap();  // Gamma(α=2, β=1)
let p = g.pdf(1.0);   // 1·e⁻¹ ≈ 0.3679
let c = g.cdf(2.0);   // uses regularized incomplete gamma P(2, 2)
assert!((g.mean() - 2.0).abs() < 1e-10);
assert!((g.variance() - 2.0).abs() < 1e-10);
```

### Beta

`Beta<T> { alpha: T, beta: T }` — shape parameters α, β. Support: [0, 1].

```rust
use numeris::stats::Beta;

let b = Beta::new(2.0_f64, 5.0).unwrap();
let p = b.pdf(0.3);    // uses regularized incomplete beta
let c = b.cdf(0.5);
assert!((b.mean() - 2.0/7.0).abs() < 1e-10);
```

### Chi-Squared

`ChiSquared<T> { k: T }` — k degrees of freedom. Special case of Gamma(k/2, 1/2).

```rust
use numeris::stats::ChiSquared;

let chi = ChiSquared::new(3.0_f64).unwrap();  // χ²(3)
let p_val = 1.0 - chi.cdf(7.815);    // p-value for χ²=7.815 with 3 df ≈ 0.05
assert_eq!(chi.mean(), 3.0);
assert_eq!(chi.variance(), 6.0);
```

### Student's t

`StudentT<T> { nu: T }` — ν degrees of freedom.

```rust
use numeris::stats::StudentT;

let t = StudentT::new(10.0_f64).unwrap();  // t(10)
let c = t.cdf(2.228);   // ≈ 0.975 (95% two-sided CI critical value)
let p = t.pdf(0.0);     // peak of distribution
assert_eq!(t.mean(), 0.0);
assert!((t.variance() - 10.0/8.0).abs() < 1e-10);  // ν/(ν-2)
```

## Discrete Distributions

### Bernoulli

`Bernoulli<T> { p: T }` — single trial with success probability p.

```rust
use numeris::stats::Bernoulli;

let b = Bernoulli::new(0.3_f64).unwrap();
assert!((b.pmf(1) - 0.3).abs() < 1e-12);
assert!((b.pmf(0) - 0.7).abs() < 1e-12);
assert!((b.mean() - 0.3).abs() < 1e-12);
assert!((b.variance() - 0.21).abs() < 1e-12);
```

### Binomial

`Binomial<T> { n: u64, p: T }` — n independent Bernoulli trials.

```rust
use numeris::stats::Binomial;

let b = Binomial::new(10, 0.5_f64).unwrap();    // B(10, 0.5)
let p5 = b.pmf(5);    // P(X=5) ≈ 0.2461
let c7  = b.cdf(7);   // P(X≤7) ≈ 0.9453
assert!((b.mean() - 5.0).abs() < 1e-10);
assert!((b.variance() - 2.5).abs() < 1e-10);
```

### Poisson

`Poisson<T> { lambda: T }` — number of events in a fixed interval.

```rust
use numeris::stats::Poisson;

let pois = Poisson::new(3.0_f64).unwrap();  // Poisson(λ=3)
let p3 = pois.pmf(3);    // P(X=3) = e⁻³·3³/3! ≈ 0.2240
let c5 = pois.cdf(5);    // P(X≤5) ≈ 0.9161
assert_eq!(pois.mean(), 3.0);
assert_eq!(pois.variance(), 3.0);
```

## Distribution Summary Table

| Distribution | Type | Parameters | Mean | Variance |
|---|---|---|---|---|
| `Normal` | Continuous | μ, σ | μ | σ² |
| `Uniform` | Continuous | a, b | (a+b)/2 | (b-a)²/12 |
| `Exponential` | Continuous | λ | 1/λ | 1/λ² |
| `Gamma` | Continuous | α, β | α/β | α/β² |
| `Beta` | Continuous | α, β | α/(α+β) | αβ/((α+β)²(α+β+1)) |
| `ChiSquared` | Continuous | k | k | 2k |
| `StudentT` | Continuous | ν | 0 (ν>1) | ν/(ν-2) (ν>2) |
| `Bernoulli` | Discrete | p | p | p(1-p) |
| `Binomial` | Discrete | n, p | np | np(1-p) |
| `Poisson` | Discrete | λ | λ | λ |

## Error Handling

```rust
use numeris::stats::StatsError;

match Normal::new(0.0_f64, -1.0) {  // std_dev must be > 0
    Err(StatsError::InvalidParameter) => { /* bad parameter */ }
    Ok(n) => { /* use n */ }
}
```
