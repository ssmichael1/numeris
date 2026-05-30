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

## Euler Angle Convention

numeris uses **ZYX intrinsic** (Tait-Bryan) Euler angles, the standard aerospace convention. `from_euler(roll, pitch, yaw)` applies rotations in this order:

1. **Yaw** ψ — rotate about the fixed **Z** axis (heading)
2. **Pitch** θ — rotate about the new **Y'** axis (nose up/down)
3. **Roll** φ — rotate about the final **X''** axis (bank)

The equivalent rotation matrix is **R = R_x(φ) · R_y(θ) · R_z(ψ)** (rightmost applied first). The quaternion composition follows the same order: **q = q_x(φ) ⊗ q_y(θ) ⊗ q_z(ψ)**.

`to_euler()` returns `(roll, pitch, yaw)` in radians. Pitch is clamped to ±π/2 at gimbal lock (pitch = ±90°), where roll and yaw become degenerate.

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

`q * v` embeds `v` as a pure quaternion $(0, \mathbf{v})$ and forms the
conjugation $q\,\mathbf{v}\,q^{-1}$. For a unit quaternion the inverse equals the
conjugate ($q^{-1} = q^{*} = (w,\,-x,\,-y,\,-z)$), so

$$
\mathbf{v}' = q\,\mathbf{v}\,q^{-1} = q\,\mathbf{v}\,q^{*}.
$$

numeris evaluates this with the algebraically equivalent, allocation-free form
(writing $q = (w, \mathbf{u})$ with vector part $\mathbf{u} = [x, y, z]$):

$$
\mathbf{v}' = \mathbf{v} + 2w\,(\mathbf{u} \times \mathbf{v})
             + 2\,\mathbf{u} \times (\mathbf{u} \times \mathbf{v}).
$$

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

**Hamilton product.** With $q = (w_1, \mathbf{u}_1)$ and $p = (w_2, \mathbf{u}_2)$
(vector parts $\mathbf{u} = [x, y, z]$), the product `q * p` is

$$
q \otimes p = \bigl(\,
  w_1 w_2 - \mathbf{u}_1 \cdot \mathbf{u}_2,\;\;
  w_1 \mathbf{u}_2 + w_2 \mathbf{u}_1 + \mathbf{u}_1 \times \mathbf{u}_2
\,\bigr).
$$

Componentwise, in scalar-first $[w, x, y, z]$ storage:

$$
\begin{aligned}
w &= w_1 w_2 - x_1 x_2 - y_1 y_2 - z_1 z_2 \\
x &= w_1 x_2 + x_1 w_2 + y_1 z_2 - z_1 y_2 \\
y &= w_1 y_2 - x_1 z_2 + y_1 w_2 + z_1 x_2 \\
z &= w_1 z_2 + x_1 y_2 - y_1 x_2 + z_1 w_2
\end{aligned}
$$

**Order of operations.** `q_total = q2 * q1` applies `q1` first, then `q2`,
because the conjugation nests right-to-left:

$$
(q_2 q_1)\,\mathbf{v}\,(q_2 q_1)^{-1}
  = q_2 \bigl( q_1\,\mathbf{v}\,q_1^{-1} \bigr) q_2^{-1}.
$$

The product is associative but **not** commutative
($q_2 \otimes q_1 \neq q_1 \otimes q_2$ in general) — the cross term
$\mathbf{u}_1 \times \mathbf{u}_2$ changes sign when the operands are swapped.

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

**Direction cosine matrix.** For a unit quaternion $q = [w, x, y, z]$,
`to_rotation_matrix()` returns the $R$ that performs the same rotation as `q * v`
(that is, $R\,\mathbf{v} = q\,\mathbf{v}\,q^{-1}$):

$$
R = \begin{bmatrix}
1 - 2(y^2 + z^2) & 2(xy - wz)       & 2(xz + wy)       \\
2(xy + wz)       & 1 - 2(x^2 + z^2) & 2(yz - wx)       \\
2(xz - wy)       & 2(yz + wx)       & 1 - 2(x^2 + y^2)
\end{bmatrix}.
$$

$R$ is orthogonal with $\det R = +1$; its rows are the direction cosines of the
rotated axes. `from_rotation_matrix` inverts this (Shepperd's method, branching on
the largest of the trace and diagonal entries for numerical robustness). Since
$R\,\mathbf{v}$ and `q * v` realize the same rotation, `q2 * q1` corresponds to the
matrix product $R(q_2)\,R(q_1)$.

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
