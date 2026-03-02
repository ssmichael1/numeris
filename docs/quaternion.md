# Quaternion

Unit quaternion 3D rotations with scalar-first convention `[w, x, y, z]`.

`Quaternion<T>` is available in the crate root with no feature flag.

## Convention

numeris uses the **scalar-first** (Hamilton) convention:

```
q = w + xi + yj + zk    where w = cos(θ/2), [x,y,z] = sin(θ/2)·axis
```

- `q * p` composes rotation `q` applied *after* rotation `p`
- `q * v` rotates vector `v` by the rotation represented by `q`

## Construction

```rust
use numeris::{Quaternion, Vector};

// Identity (no rotation)
let id = Quaternion::<f64>::identity();

// From axis-angle
let q = Quaternion::from_axis_angle(
    &Vector::from_array([0.0_f64, 0.0, 1.0]),  // z-axis
    std::f64::consts::FRAC_PI_2,               // 90°
);

// From Euler angles (roll, pitch, yaw in radians — ZYX convention)
let q_euler = Quaternion::from_euler(0.0_f64, 0.0, std::f64::consts::FRAC_PI_4);

// From rotation matrix (3×3 orthogonal)
use numeris::Matrix3;
let rot = Matrix3::<f64>::eye();  // identity rotation
let q_mat = Quaternion::from_rotation_matrix(&rot);

// Elementary rotations
let qx = Quaternion::rotx(std::f64::consts::PI / 6.0); // 30° around x
let qy = Quaternion::roty(std::f64::consts::FRAC_PI_4); // 45° around y
let qz = Quaternion::rotz(std::f64::consts::FRAC_PI_2); // 90° around z

// Direct construction (auto-normalized)
let q_raw = Quaternion::new(1.0_f64, 0.0, 0.0, 0.0);  // identity
```

## Vector Rotation

```rust
use numeris::{Quaternion, Vector};

// 90° rotation around z-axis
let q = Quaternion::from_axis_angle(
    &Vector::from_array([0.0_f64, 0.0, 1.0]),
    std::f64::consts::FRAC_PI_2,
);

let v = Vector::from_array([1.0_f64, 0.0, 0.0]);
let rotated = q * v;   // ≈ [0, 1, 0]

assert!((rotated[0] - 0.0).abs() < 1e-14);
assert!((rotated[1] - 1.0).abs() < 1e-14);
assert!((rotated[2] - 0.0).abs() < 1e-14);
```

## Composition

```rust
use numeris::{Quaternion, Vector};

let q1 = Quaternion::rotx(std::f64::consts::FRAC_PI_2);  // 90° x
let q2 = Quaternion::rotz(std::f64::consts::FRAC_PI_2);  // 90° z

// Apply q1 first, then q2
let combined = q2 * q1;

// Equivalently:
let v = Vector::from_array([1.0_f64, 0.0, 0.0]);
let r1 = q2 * (q1 * v);   // step by step
let r2 = combined * v;     // combined rotation
// r1 ≈ r2
```

## Inverse and Conjugate

For unit quaternions, conjugate = inverse:

```rust
let q = Quaternion::from_axis_angle(
    &Vector::from_array([0.0_f64, 0.0, 1.0]),
    1.0,
);

let q_conj = q.conjugate();   // q* = [w, -x, -y, -z]
let q_inv  = q.inverse();     // same as conjugate for unit quaternions

// q * q^{-1} = identity
let id = q * q_inv;
assert!((id.w() - 1.0).abs() < 1e-14);
```

## Interpolation (SLERP)

Spherical linear interpolation — constant angular velocity, smooth path on SO(3).

```rust
use numeris::Quaternion;

let q0 = Quaternion::<f64>::identity();
let q1 = Quaternion::rotz(std::f64::consts::FRAC_PI_2);

// t=0 → q0, t=1 → q1, t=0.5 → halfway (45°)
let q_half = q0.slerp(&q1, 0.5);

// Use for smooth animation or attitude interpolation
for i in 0..=10 {
    let t = i as f64 / 10.0;
    let q = q0.slerp(&q1, t);
    // q represents i*9° rotation around z-axis
}
```

## Conversion

```rust
use numeris::Quaternion;

let q = Quaternion::from_axis_angle(
    &numeris::Vector::from_array([0.0_f64, 0.0, 1.0]),
    1.2,
);

// To rotation matrix (3×3 orthogonal)
let rot = q.to_rotation_matrix();   // Matrix3<f64>

// To axis-angle (axis is unit vector, angle in radians)
let (axis, angle) = q.to_axis_angle();

// To Euler angles (ZYX: roll, pitch, yaw)
let (roll, pitch, yaw) = q.to_euler();

// Components
let w = q.w();
let x = q.x();
let y = q.y();
let z = q.z();

// Normalize (in case of accumulated numerical drift)
let q_norm = q.normalize();
```

## Operations

```rust
let q = Quaternion::rotz(1.0_f64);

// Hamilton product (composition)
let q2 = q * q;         // 2 radians around z

// Scalar operations
let q_scaled = q * 2.0; // NOT a unit quaternion — use normalize() after

// Norm (should be ≈ 1.0 for properly constructed quaternions)
let n = q.norm();
assert!((n - 1.0).abs() < 1e-14);
```

## Attitude Determination Example

```rust
use numeris::{Quaternion, Vector};

// Represent spacecraft attitude as quaternion (body frame relative to ECI)
let q_body_to_eci = Quaternion::from_euler(
    0.1_f64,   // roll  10°
    -0.05,     // pitch -5°
    1.57,      // yaw   90°
);

// Transform a vector from body to ECI frame
let boresight_body = Vector::from_array([1.0_f64, 0.0, 0.0]);
let boresight_eci  = q_body_to_eci * boresight_body;

// Attitude error between two frames
let q_target = Quaternion::identity();
let q_error  = q_target * q_body_to_eci.inverse();
let (axis, angle) = q_error.to_axis_angle();
// |angle| is the pointing error magnitude
```
