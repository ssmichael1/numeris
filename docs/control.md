# Digital Control

Biquad-cascade IIR filters and a discrete-time PID controller. All components are no-std compatible, use no complex arithmetic at runtime, and work with both `f32` and `f64`.

Requires the `control` Cargo feature:

```toml
numeris = { version = "0.2", features = ["control"] }
```

## IIR Filter Design

numeris designs IIR filters by:

1. Computing analog prototype poles (Butterworth or Chebyshev Type I)
2. Applying a bilinear transform (Tustin's method) with frequency pre-warping
3. Pairing poles into second-order sections (biquads)

### Butterworth

Maximally flat passband, monotone rolloff.

```rust
use numeris::control::{butterworth_lowpass, butterworth_highpass, BiquadCascade};

// 4th-order lowpass at 1 kHz, 8 kHz sample rate
// → 2 biquad sections (N/2 = 2 for even order)
let mut lpf: BiquadCascade<f64, 2> = butterworth_lowpass(4, 1000.0, 8000.0).unwrap();

// Process one sample
let y = lpf.tick(1.0);

// Process a block
let input  = [1.0_f64, 0.5, -0.3, 0.8, 0.2];
let mut output = [0.0_f64; 5];
lpf.reset();
lpf.process(&input, &mut output);

// In-place processing
let mut buf = [1.0_f64, 0.5, -0.3];
lpf.reset();
lpf.process_inplace(&mut buf);

// Highpass version
let mut hpf: BiquadCascade<f64, 2> = butterworth_highpass(4, 2000.0, 8000.0).unwrap();
```

### Chebyshev Type I

Steeper rolloff than Butterworth at the cost of passband ripple.

```rust
use numeris::control::{chebyshev1_lowpass, chebyshev1_highpass, BiquadCascade};

// 4th-order Chebyshev lowpass, 1 dB passband ripple
let mut cheb: BiquadCascade<f64, 2> = chebyshev1_lowpass(4, 1.0, 1000.0, 8000.0).unwrap();

let y = cheb.tick(1.0);

// Highpass version
let mut cheb_hp: BiquadCascade<f64, 2> = chebyshev1_highpass(4, 1.0, 2000.0, 8000.0).unwrap();
```

### Design Function Signatures

```rust
// Butterworth
fn butterworth_lowpass<T, const N: usize>(
    order: usize,           // filter order (1..=2N, even fills N sections, odd uses degenerate last)
    cutoff_hz: T,           // -3 dB frequency
    sample_hz: T,           // sample rate
) -> Result<BiquadCascade<T, N>, ControlError>

fn butterworth_highpass<T, const N: usize>(
    order: usize,
    cutoff_hz: T,
    sample_hz: T,
) -> Result<BiquadCascade<T, N>, ControlError>

// Chebyshev Type I
fn chebyshev1_lowpass<T, const N: usize>(
    order: usize,
    ripple_db: T,           // passband ripple in dB (> 0)
    cutoff_hz: T,
    sample_hz: T,
) -> Result<BiquadCascade<T, N>, ControlError>

fn chebyshev1_highpass<T, const N: usize>(
    order: usize,
    ripple_db: T,
    cutoff_hz: T,
    sample_hz: T,
) -> Result<BiquadCascade<T, N>, ControlError>
```

### Comparison

| Design | Passband | Stopband | Notes |
|---|---|---|---|
| Butterworth | Flat | Monotone | General purpose |
| Chebyshev Type I | Ripple ≤ N dB | Steeper | Better rolloff, slight passband distortion |

## Biquad Sections

A `Biquad<T>` implements a single second-order IIR section in **Direct Form II Transposed** (DFII-T), which is numerically better conditioned than Direct Form I:

```
y[n] = b0*x[n] + s1[n-1]
s1[n] = b1*x[n] - a1*y[n] + s2[n-1]
s2[n] = b2*x[n] - a2*y[n]
```

Coefficients follow the convention `H(z) = (b0 + b1 z⁻¹ + b2 z⁻²) / (1 + a1 z⁻¹ + a2 z⁻²)`.

```rust
use numeris::control::Biquad;

// Construct manually (e.g., from external design tool)
let bq = Biquad::new(
    1.0_f64, 2.0, 1.0,   // b0, b1, b2
    1.0, -1.8, 0.81,     // a0 (must be 1.0), a1, a2
);

let mut state = bq.initial_state();
let y = bq.tick(1.0, &mut state);
```

`BiquadCascade<T, N>` chains `N` biquad sections in series.

## PID Controller

Discrete-time PID with:

- **Trapezoidal integration** (bilinear, no integrator windup from step changes)
- **Derivative on measurement** (avoids derivative kick on setpoint changes)
- **Optional derivative LPF** (reduces noise amplification)
- **Anti-windup** via back-calculation (clamps integrator when output saturates)
- **Output limits** (hard clamp)

```rust
use numeris::control::Pid;

// Kp=2.0, Ki=0.5, Kd=0.1, sample period dt=0.001 s (1 kHz)
let mut pid = Pid::new(2.0_f64, 0.5, 0.1, 0.001)
    .with_output_limits(-10.0, 10.0)           // clamp output to ±10
    .with_derivative_filter(0.01);              // LPF time constant τ=10ms

let setpoint    = 1.0_f64;
let mut process = 0.0_f64;

for _ in 0..1000 {
    let u = pid.tick(setpoint, process);
    // Simple first-order plant: τ_plant = 0.1 s
    process += 0.001 * (-process + u);
}

assert!((process - setpoint).abs() < 0.01);
```

### API

```rust
// Constructor
fn Pid::new(kp: T, ki: T, kd: T, dt: T) -> Pid<T>

// Builders
fn with_output_limits(self, min: T, max: T) -> Pid<T>
fn with_derivative_filter(self, tau: T)     -> Pid<T>   // τ = filter time constant

// Runtime
fn tick(&mut self, setpoint: T, measurement: T) -> T   // compute output u[k]
fn reset(&mut self)                                      // clear integrator and derivative state
```

### Difference Equations

Let `e[k] = setpoint[k] - measurement[k]`. With trapezoidal integration and derivative on measurement:

```
I[k] = I[k-1] + Ki * dt/2 * (e[k] + e[k-1])     (bilinear integration)
D[k] = -Kd/τ * (measurement[k] - measurement[k-1]) / dt + (1 - dt/τ) * D[k-1]  (filtered derivative)
u[k] = clamp(Kp*e[k] + I[k] + D[k])
```

Anti-windup: if `u` is clamped, the integrator is back-corrected: `I[k] -= Kb * (u_unclamped - u_clamped)`.

## Error Handling

```rust
use numeris::control::ControlError;

match butterworth_lowpass::<f64, 2>(4, 1000.0, 8000.0) {
    Ok(filter) => { /* use filter */ }
    Err(ControlError::InvalidOrder)     => { /* order too large for N sections */ }
    Err(ControlError::InvalidFrequency) => { /* cutoff >= nyquist */ }
    Err(ControlError::InvalidRipple)    => { /* ripple_db <= 0 */ }
}
```
