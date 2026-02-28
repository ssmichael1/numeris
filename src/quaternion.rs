use core::ops::{Mul, Neg};

use crate::matrix::vector::Vector3;
use crate::traits::FloatScalar;
use crate::Matrix;

/// Unit quaternion for 3D rotations.
///
/// Scalar-first convention: `[w, x, y, z]` where `w` is the scalar part
/// and `(x, y, z)` is the vector part.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion<T> {
    pub w: T,
    pub x: T,
    pub y: T,
    pub z: T,
}

// ── Constructors ─────────────────────────────────────────────────────

impl<T: FloatScalar> Quaternion<T> {
    /// Create a quaternion from components.
    #[inline]
    pub fn new(w: T, x: T, y: T, z: T) -> Self {
        Self { w, x, y, z }
    }

    /// Identity quaternion (no rotation).
    #[inline]
    pub fn identity() -> Self {
        Self {
            w: T::one(),
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
        }
    }

    /// Create from an axis (must be unit length) and angle in radians.
    #[inline]
    pub fn from_axis_angle(axis: Vector3<T>, angle: T) -> Self {
        let half = angle / (T::one() + T::one());
        let (s, c) = half.sin_cos();
        Self {
            w: c,
            x: axis[0] * s,
            y: axis[1] * s,
            z: axis[2] * s,
        }
    }

    /// Create from a 3×3 rotation matrix using Shepperd's method.
    ///
    /// Numerically stable for all rotation angles.
    pub fn from_rotation_matrix(m: &Matrix<T, 3, 3>) -> Self {
        let trace = m[(0, 0)] + m[(1, 1)] + m[(2, 2)];
        let one = T::one();
        let quarter = one / (one + one + one + one);
        let half = one / (one + one);

        if trace > T::zero() {
            let s = (trace + one).sqrt();
            let w = s * half;
            let k = half / s;
            Self {
                w,
                x: (m[(2, 1)] - m[(1, 2)]) * k,
                y: (m[(0, 2)] - m[(2, 0)]) * k,
                z: (m[(1, 0)] - m[(0, 1)]) * k,
            }
        } else if m[(0, 0)] >= m[(1, 1)] && m[(0, 0)] >= m[(2, 2)] {
            let s = ((one + m[(0, 0)] - m[(1, 1)] - m[(2, 2)]) * quarter).sqrt();
            let k = quarter / s;
            Self {
                w: (m[(2, 1)] - m[(1, 2)]) * k,
                x: s,
                y: (m[(0, 1)] + m[(1, 0)]) * k,
                z: (m[(0, 2)] + m[(2, 0)]) * k,
            }
        } else if m[(1, 1)] >= m[(2, 2)] {
            let s = ((one - m[(0, 0)] + m[(1, 1)] - m[(2, 2)]) * quarter).sqrt();
            let k = quarter / s;
            Self {
                w: (m[(0, 2)] - m[(2, 0)]) * k,
                x: (m[(0, 1)] + m[(1, 0)]) * k,
                y: s,
                z: (m[(1, 2)] + m[(2, 1)]) * k,
            }
        } else {
            let s = ((one - m[(0, 0)] - m[(1, 1)] + m[(2, 2)]) * quarter).sqrt();
            let k = quarter / s;
            Self {
                w: (m[(1, 0)] - m[(0, 1)]) * k,
                x: (m[(0, 2)] + m[(2, 0)]) * k,
                y: (m[(1, 2)] + m[(2, 1)]) * k,
                z: s,
            }
        }
    }

    /// Create from Euler angles (ZYX intrinsic convention).
    ///
    /// Arguments: roll (X), pitch (Y), yaw (Z), all in radians.
    pub fn from_euler(roll: T, pitch: T, yaw: T) -> Self {
        let half = T::one() / (T::one() + T::one());
        let (sr, cr) = (roll * half).sin_cos();
        let (sp, cp) = (pitch * half).sin_cos();
        let (sy, cy) = (yaw * half).sin_cos();

        Self {
            w: cr * cp * cy + sr * sp * sy,
            x: sr * cp * cy - cr * sp * sy,
            y: cr * sp * cy + sr * cp * sy,
            z: cr * cp * sy - sr * sp * cy,
        }
    }
}

// ── Core operations ──────────────────────────────────────────────────

impl<T: FloatScalar> Quaternion<T> {
    /// Conjugate: `(w, -x, -y, -z)`.
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self {
            w: self.w,
            x: T::zero() - self.x,
            y: T::zero() - self.y,
            z: T::zero() - self.z,
        }
    }

    /// Squared norm: `w² + x² + y² + z²`.
    #[inline]
    pub fn norm_squared(&self) -> T {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Norm (magnitude).
    #[inline]
    pub fn norm(&self) -> T {
        self.norm_squared().sqrt()
    }

    /// Normalize to unit quaternion.
    #[inline]
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        let inv = T::one() / n;
        Self {
            w: self.w * inv,
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
        }
    }

    /// Inverse: `conjugate / norm²`.
    ///
    /// For unit quaternions this equals the conjugate.
    #[inline]
    pub fn inverse(&self) -> Self {
        let inv_n2 = T::one() / self.norm_squared();
        Self {
            w: self.w * inv_n2,
            x: (T::zero() - self.x) * inv_n2,
            y: (T::zero() - self.y) * inv_n2,
            z: (T::zero() - self.z) * inv_n2,
        }
    }

    /// Dot product of two quaternions.
    #[inline]
    pub fn dot(&self, rhs: &Self) -> T {
        self.w * rhs.w + self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

// ── Conversions ──────────────────────────────────────────────────────

impl<T: FloatScalar> Quaternion<T> {
    /// Convert to a 3×3 rotation matrix.
    pub fn to_rotation_matrix(&self) -> Matrix<T, 3, 3> {
        let two = T::one() + T::one();
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);

        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let wx = w * x;
        let wy = w * y;
        let wz = w * z;

        Matrix::new([
            [
                T::one() - two * (yy + zz),
                two * (xy - wz),
                two * (xz + wy),
            ],
            [
                two * (xy + wz),
                T::one() - two * (xx + zz),
                two * (yz - wx),
            ],
            [
                two * (xz - wy),
                two * (yz + wx),
                T::one() - two * (xx + yy),
            ],
        ])
    }

    /// Convert to axis-angle representation.
    ///
    /// Returns `(axis, angle)` where `axis` is a unit vector and `angle` is in radians.
    /// For identity rotation, returns `([1,0,0], 0)`.
    pub fn to_axis_angle(&self) -> (Vector3<T>, T) {
        let two = T::one() + T::one();
        let n = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        let eps = T::epsilon();

        if n < eps {
            (
                Vector3::from_array([T::one(), T::zero(), T::zero()]),
                T::zero(),
            )
        } else {
            let inv_n = T::one() / n;
            let angle = two * n.atan2(self.w);
            (
                Vector3::from_array([self.x * inv_n, self.y * inv_n, self.z * inv_n]),
                angle,
            )
        }
    }

    /// Convert to Euler angles (ZYX intrinsic convention).
    ///
    /// Returns `(roll, pitch, yaw)` in radians.
    pub fn to_euler(&self) -> (T, T, T) {
        let two = T::one() + T::one();
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);

        // Roll (X-axis rotation)
        let sinr_cosp = two * (w * x + y * z);
        let cosr_cosp = T::one() - two * (x * x + y * y);
        let roll = sinr_cosp.atan2(cosr_cosp);

        // Pitch (Y-axis rotation) — clamp to avoid NaN at gimbal lock
        let sinp = two * (w * y - z * x);
        let pitch = if sinp.abs() >= T::one() {
            // Use copysign-like behavior for ±π/2
            let half_pi = T::one().atan2(T::zero()); // π/2
            if sinp > T::zero() {
                half_pi
            } else {
                T::zero() - half_pi
            }
        } else {
            sinp.asin()
        };

        // Yaw (Z-axis rotation)
        let siny_cosp = two * (w * z + x * y);
        let cosy_cosp = T::one() - two * (y * y + z * z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        (roll, pitch, yaw)
    }
}

// ── Interpolation ────────────────────────────────────────────────────

impl<T: FloatScalar> Quaternion<T> {
    /// Spherical linear interpolation between `self` and `other`.
    ///
    /// `t = 0` returns `self`, `t = 1` returns `other`.
    /// Automatically takes the short path on the 4-sphere.
    pub fn slerp(&self, other: &Self, t: T) -> Self {
        let mut dot = self.dot(other);

        // If dot is negative, negate one to take the short path
        let other = if dot < T::zero() {
            dot = T::zero() - dot;
            -*other
        } else {
            *other
        };

        let eps = T::from(1e-6).unwrap();
        if dot > T::one() - eps {
            // Very close — fall back to normalized lerp
            Self {
                w: self.w + (other.w - self.w) * t,
                x: self.x + (other.x - self.x) * t,
                y: self.y + (other.y - self.y) * t,
                z: self.z + (other.z - self.z) * t,
            }
            .normalize()
        } else {
            let theta = dot.acos();
            let sin_theta = theta.sin();
            let a = ((T::one() - t) * theta).sin() / sin_theta;
            let b = (t * theta).sin() / sin_theta;
            Self {
                w: self.w * a + other.w * b,
                x: self.x * a + other.x * b,
                y: self.y * a + other.y * b,
                z: self.z * a + other.z * b,
            }
        }
    }
}

// ── Kinematics ───────────────────────────────────────────────────────

impl<T: FloatScalar> Quaternion<T> {
    /// Create a small rotation quaternion from angular velocity and time step.
    ///
    /// `omega` is the angular velocity vector (rad/s), `dt` is the time step.
    /// Uses the exact axis-angle formula: angle = |omega| * dt, axis = omega / |omega|.
    pub fn from_angular_velocity(omega: &Vector3<T>, dt: T) -> Self {
        let angle = omega.norm() * dt;
        let eps = T::epsilon();

        if angle < eps {
            Self::identity()
        } else {
            let axis = omega.normalize();
            Self::from_axis_angle(axis, angle)
        }
    }
}

// ── Operators ────────────────────────────────────────────────────────

// Hamilton product: q1 * q2
impl<T: FloatScalar> Mul for Quaternion<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}

// Reference variants for Hamilton product
impl<T: FloatScalar> Mul<Quaternion<T>> for &Quaternion<T> {
    type Output = Quaternion<T>;
    #[inline]
    fn mul(self, rhs: Quaternion<T>) -> Quaternion<T> {
        (*self).mul(rhs)
    }
}

impl<T: FloatScalar> Mul<&Quaternion<T>> for Quaternion<T> {
    type Output = Quaternion<T>;
    #[inline]
    fn mul(self, rhs: &Quaternion<T>) -> Quaternion<T> {
        self.mul(*rhs)
    }
}

impl<T: FloatScalar> Mul<&Quaternion<T>> for &Quaternion<T> {
    type Output = Quaternion<T>;
    #[inline]
    fn mul(self, rhs: &Quaternion<T>) -> Quaternion<T> {
        (*self).mul(*rhs)
    }
}

// Rotate a vector: q * v computes q v q⁻¹
// Uses the efficient formula: v' = v + 2w(u × v) + 2(u × (u × v))
impl<T: FloatScalar> Mul<Vector3<T>> for Quaternion<T> {
    type Output = Vector3<T>;

    #[inline]
    fn mul(self, v: Vector3<T>) -> Vector3<T> {
        let u = Vector3::from_array([self.x, self.y, self.z]);
        let two = T::one() + T::one();
        let uv = u.cross(&v);
        let uuv = u.cross(&uv);
        v + uv * (two * self.w) + uuv * two
    }
}

impl<T: FloatScalar> Mul<Vector3<T>> for &Quaternion<T> {
    type Output = Vector3<T>;
    #[inline]
    fn mul(self, v: Vector3<T>) -> Vector3<T> {
        (*self).mul(v)
    }
}

impl<T: FloatScalar> Mul<&Vector3<T>> for Quaternion<T> {
    type Output = Vector3<T>;
    #[inline]
    fn mul(self, v: &Vector3<T>) -> Vector3<T> {
        self.mul(*v)
    }
}

impl<T: FloatScalar> Mul<&Vector3<T>> for &Quaternion<T> {
    type Output = Vector3<T>;
    #[inline]
    fn mul(self, v: &Vector3<T>) -> Vector3<T> {
        (*self).mul(*v)
    }
}

// Negation
impl<T: FloatScalar> Neg for Quaternion<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            w: T::zero() - self.w,
            x: T::zero() - self.x,
            y: T::zero() - self.y,
            z: T::zero() - self.z,
        }
    }
}

impl<T: FloatScalar> Neg for &Quaternion<T> {
    type Output = Quaternion<T>;

    #[inline]
    fn neg(self) -> Quaternion<T> {
        (*self).neg()
    }
}

// ── Display ──────────────────────────────────────────────────────────

impl<T: core::fmt::Display> core::fmt::Display for Quaternion<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "({} + {}i + {}j + {}k)", self.w, self.x, self.y, self.z)
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    const EPS: f64 = 1e-12;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn quat_approx_eq(a: &Quaternion<f64>, b: &Quaternion<f64>) -> bool {
        // Quaternions q and -q represent the same rotation
        let direct =
            approx_eq(a.w, b.w) && approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z);
        let negated = approx_eq(a.w, -b.w)
            && approx_eq(a.x, -b.x)
            && approx_eq(a.y, -b.y)
            && approx_eq(a.z, -b.z);
        direct || negated
    }

    // ── Constructors ─────────────────────────────────────────────

    #[test]
    fn identity() {
        let q = Quaternion::<f64>::identity();
        assert_eq!(q.w, 1.0);
        assert_eq!(q.x, 0.0);
        assert_eq!(q.y, 0.0);
        assert_eq!(q.z, 0.0);
    }

    #[test]
    fn from_axis_angle_z_90() {
        let axis = Vector3::from_array([0.0, 0.0, 1.0]);
        let q = Quaternion::from_axis_angle(axis, FRAC_PI_2);
        assert!(approx_eq(q.norm(), 1.0));
        assert!(approx_eq(q.w, (FRAC_PI_4).cos()));
        assert!(approx_eq(q.z, (FRAC_PI_4).sin()));
    }

    #[test]
    fn from_axis_angle_identity() {
        let axis = Vector3::from_array([1.0, 0.0, 0.0]);
        let q = Quaternion::from_axis_angle(axis, 0.0);
        assert!(quat_approx_eq(&q, &Quaternion::identity()));
    }

    #[test]
    fn from_euler_zero() {
        let q = Quaternion::from_euler(0.0, 0.0, 0.0);
        assert!(quat_approx_eq(&q, &Quaternion::identity()));
    }

    // ── Core operations ──────────────────────────────────────────

    #[test]
    fn conjugate() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let c = q.conjugate();
        assert_eq!(c.w, 1.0);
        assert_eq!(c.x, -2.0);
        assert_eq!(c.y, -3.0);
        assert_eq!(c.z, -4.0);
    }

    #[test]
    fn norm_and_normalize() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        assert!(approx_eq(q.norm_squared(), 30.0));
        assert!(approx_eq(q.norm(), 30.0_f64.sqrt()));

        let u = q.normalize();
        assert!(approx_eq(u.norm(), 1.0));
    }

    #[test]
    fn inverse_unit() {
        let axis = Vector3::from_array([0.0, 1.0, 0.0]);
        let q = Quaternion::from_axis_angle(axis, 1.0);
        let qi = q.inverse();
        let product = q * qi;
        assert!(quat_approx_eq(&product, &Quaternion::identity()));
    }

    #[test]
    fn inverse_non_unit() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let qi = q.inverse();
        let product = q * qi;
        assert!(quat_approx_eq(&product, &Quaternion::identity()));
    }

    #[test]
    fn dot_product() {
        let a = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let b = Quaternion::new(5.0, 6.0, 7.0, 8.0);
        assert!(approx_eq(a.dot(&b), 70.0)); // 5+12+21+32
    }

    // ── Hamilton product ─────────────────────────────────────────

    #[test]
    fn hamilton_product_identity() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0).normalize();
        let id = Quaternion::identity();
        assert!(quat_approx_eq(&(q * id), &q));
        assert!(quat_approx_eq(&(id * q), &q));
    }

    #[test]
    fn hamilton_product_associative() {
        let a = Quaternion::from_axis_angle(Vector3::from_array([1.0, 0.0, 0.0]), 0.3);
        let b = Quaternion::from_axis_angle(Vector3::from_array([0.0, 1.0, 0.0]), 0.5);
        let c = Quaternion::from_axis_angle(Vector3::from_array([0.0, 0.0, 1.0]), 0.7);

        let ab_c = (a * b) * c;
        let a_bc = a * (b * c);
        assert!(quat_approx_eq(&ab_c, &a_bc));
    }

    #[test]
    fn hamilton_product_ref_variants() {
        let a = Quaternion::from_axis_angle(Vector3::from_array([1.0, 0.0, 0.0]), 0.5);
        let b = Quaternion::from_axis_angle(Vector3::from_array([0.0, 1.0, 0.0]), 0.7);
        let expected = a * b;

        assert!(quat_approx_eq(&(&a * b), &expected));
        assert!(quat_approx_eq(&(a * &b), &expected));
        assert!(quat_approx_eq(&(&a * &b), &expected));
    }

    // ── Vector rotation ──────────────────────────────────────────

    #[test]
    fn rotate_vector_identity() {
        let q = Quaternion::<f64>::identity();
        let v = Vector3::from_array([1.0, 2.0, 3.0]);
        let r = q * v;
        assert!(approx_eq(r[0], 1.0));
        assert!(approx_eq(r[1], 2.0));
        assert!(approx_eq(r[2], 3.0));
    }

    #[test]
    fn rotate_vector_90_about_z() {
        let q = Quaternion::from_axis_angle(Vector3::from_array([0.0, 0.0, 1.0]), FRAC_PI_2);
        let v = Vector3::from_array([1.0, 0.0, 0.0]);
        let r = q * v;
        assert!(approx_eq(r[0], 0.0));
        assert!(approx_eq(r[1], 1.0));
        assert!(approx_eq(r[2], 0.0));
    }

    #[test]
    fn rotate_vector_180_about_y() {
        let q = Quaternion::from_axis_angle(Vector3::from_array([0.0, 1.0, 0.0]), PI);
        let v = Vector3::from_array([1.0, 0.0, 0.0]);
        let r = q * v;
        assert!(approx_eq(r[0], -1.0));
        assert!(approx_eq(r[1], 0.0));
        assert!(approx_eq(r[2], 0.0));
    }

    #[test]
    fn rotate_vector_ref_variants() {
        let q = Quaternion::from_axis_angle(Vector3::from_array([0.0, 0.0, 1.0]), FRAC_PI_2);
        let v = Vector3::from_array([1.0, 0.0, 0.0]);
        let expected = q * v;

        let r1 = &q * v;
        let r2 = q * &v;
        let r3 = &q * &v;
        for r in [r1, r2, r3] {
            assert!(approx_eq(r[0], expected[0]));
            assert!(approx_eq(r[1], expected[1]));
            assert!(approx_eq(r[2], expected[2]));
        }
    }

    // ── Rotation matrix roundtrip ────────────────────────────────

    #[test]
    fn to_rotation_matrix_identity() {
        let q = Quaternion::<f64>::identity();
        let m = q.to_rotation_matrix();
        let eye: Matrix<f64, 3, 3> = Matrix::eye();
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq(m[(i, j)], eye[(i, j)]));
            }
        }
    }

    #[test]
    fn rotation_matrix_roundtrip() {
        let q = Quaternion::from_axis_angle(
            Vector3::from_array([1.0, 1.0, 1.0]).normalize(),
            1.23,
        );
        let m = q.to_rotation_matrix();
        let q2 = Quaternion::from_rotation_matrix(&m);
        assert!(quat_approx_eq(&q, &q2));
    }

    #[test]
    fn rotation_matrix_matches_vector_rotation() {
        let q = Quaternion::from_axis_angle(
            Vector3::from_array([0.0, 1.0, 0.0]),
            0.8,
        );
        let v = Vector3::from_array([1.0, 2.0, 3.0]);
        let r_quat = q * v;
        let m = q.to_rotation_matrix();
        let r_mat = m.vecmul(&v);

        assert!(approx_eq(r_quat[0], r_mat[0]));
        assert!(approx_eq(r_quat[1], r_mat[1]));
        assert!(approx_eq(r_quat[2], r_mat[2]));
    }

    #[test]
    fn from_rotation_matrix_all_branches() {
        // Test each branch of Shepperd's method by using rotations
        // that make different diagonal elements dominant

        // Branch: trace > 0 (small rotation)
        let q1 = Quaternion::from_axis_angle(
            Vector3::from_array([1.0, 0.0, 0.0]),
            0.1,
        );
        let m1 = q1.to_rotation_matrix();
        let q1r = Quaternion::from_rotation_matrix(&m1);
        assert!(quat_approx_eq(&q1, &q1r));

        // Branch: m[0,0] dominant (180° about X)
        let q2 = Quaternion::from_axis_angle(
            Vector3::from_array([1.0, 0.0, 0.0]),
            PI - 0.01,
        );
        let m2 = q2.to_rotation_matrix();
        let q2r = Quaternion::from_rotation_matrix(&m2);
        assert!(quat_approx_eq(&q2, &q2r));

        // Branch: m[1,1] dominant (180° about Y)
        let q3 = Quaternion::from_axis_angle(
            Vector3::from_array([0.0, 1.0, 0.0]),
            PI - 0.01,
        );
        let m3 = q3.to_rotation_matrix();
        let q3r = Quaternion::from_rotation_matrix(&m3);
        assert!(quat_approx_eq(&q3, &q3r));

        // Branch: m[2,2] dominant (180° about Z)
        let q4 = Quaternion::from_axis_angle(
            Vector3::from_array([0.0, 0.0, 1.0]),
            PI - 0.01,
        );
        let m4 = q4.to_rotation_matrix();
        let q4r = Quaternion::from_rotation_matrix(&m4);
        assert!(quat_approx_eq(&q4, &q4r));
    }

    // ── Axis-angle roundtrip ─────────────────────────────────────

    #[test]
    fn axis_angle_roundtrip() {
        let axis = Vector3::from_array([0.0, 1.0, 0.0]);
        let angle = 1.5;
        let q = Quaternion::from_axis_angle(axis, angle);
        let (axis2, angle2) = q.to_axis_angle();

        assert!(approx_eq(angle, angle2));
        assert!(approx_eq(axis[0], axis2[0]));
        assert!(approx_eq(axis[1], axis2[1]));
        assert!(approx_eq(axis[2], axis2[2]));
    }

    #[test]
    fn axis_angle_identity() {
        let q = Quaternion::<f64>::identity();
        let (_, angle) = q.to_axis_angle();
        assert!(approx_eq(angle, 0.0));
    }

    // ── Euler roundtrip ──────────────────────────────────────────

    #[test]
    fn euler_roundtrip() {
        let roll = 0.3;
        let pitch = 0.2;
        let yaw = 0.7;
        let q = Quaternion::from_euler(roll, pitch, yaw);
        let (r2, p2, y2) = q.to_euler();

        assert!(approx_eq(roll, r2));
        assert!(approx_eq(pitch, p2));
        assert!(approx_eq(yaw, y2));
    }

    #[test]
    fn euler_gimbal_lock() {
        // Pitch = ±90° is the gimbal lock singularity
        let q = Quaternion::from_euler(0.0, FRAC_PI_2, 0.0);
        let (_, pitch, _) = q.to_euler();
        assert!(approx_eq(pitch, FRAC_PI_2));
    }

    // ── Negation ─────────────────────────────────────────────────

    #[test]
    fn negation() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let n = -q;
        assert_eq!(n.w, -1.0);
        assert_eq!(n.x, -2.0);
        assert_eq!(n.y, -3.0);
        assert_eq!(n.z, -4.0);
    }

    #[test]
    fn ref_negation() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(-&q, -q);
    }

    #[test]
    fn negate_same_rotation() {
        // q and -q represent the same rotation
        let q = Quaternion::from_axis_angle(Vector3::from_array([0.0, 0.0, 1.0]), 1.0);
        let v = Vector3::from_array([1.0, 0.0, 0.0]);
        let r1 = q * v;
        let r2 = (-q) * v;
        assert!(approx_eq(r1[0], r2[0]));
        assert!(approx_eq(r1[1], r2[1]));
        assert!(approx_eq(r1[2], r2[2]));
    }

    // ── Slerp ────────────────────────────────────────────────────

    #[test]
    fn slerp_endpoints() {
        let a = Quaternion::from_axis_angle(Vector3::from_array([1.0, 0.0, 0.0]), 0.0);
        let b = Quaternion::from_axis_angle(Vector3::from_array([0.0, 0.0, 1.0]), 1.0);

        let s0 = a.slerp(&b, 0.0);
        let s1 = a.slerp(&b, 1.0);
        assert!(quat_approx_eq(&s0, &a));
        assert!(quat_approx_eq(&s1, &b));
    }

    #[test]
    fn slerp_midpoint() {
        let a = Quaternion::<f64>::identity();
        let b = Quaternion::from_axis_angle(Vector3::from_array([0.0, 0.0, 1.0]), FRAC_PI_2);
        let mid = a.slerp(&b, 0.5);
        // Midpoint should be a 45° rotation about Z
        let expected =
            Quaternion::from_axis_angle(Vector3::from_array([0.0, 0.0, 1.0]), FRAC_PI_4);
        assert!(quat_approx_eq(&mid, &expected));
    }

    #[test]
    fn slerp_unit_output() {
        let a = Quaternion::from_axis_angle(Vector3::from_array([1.0, 0.0, 0.0]), 0.3);
        let b = Quaternion::from_axis_angle(Vector3::from_array([0.0, 1.0, 0.0]), 1.5);
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let s = a.slerp(&b, t);
            assert!(approx_eq(s.norm(), 1.0));
        }
    }

    // ── Kinematics ───────────────────────────────────────────────

    #[test]
    fn angular_velocity_zero() {
        let omega = Vector3::from_array([0.0, 0.0, 0.0]);
        let q = Quaternion::from_angular_velocity(&omega, 1.0);
        assert!(quat_approx_eq(&q, &Quaternion::identity()));
    }

    #[test]
    fn angular_velocity_consistency() {
        // Rotating at 1 rad/s about Z for 0.5 s should give 0.5 rad rotation
        let omega = Vector3::from_array([0.0, 0.0, 1.0]);
        let q = Quaternion::from_angular_velocity(&omega, 0.5);
        let expected = Quaternion::from_axis_angle(Vector3::from_array([0.0, 0.0, 1.0]), 0.5);
        assert!(quat_approx_eq(&q, &expected));
    }

    // ── Composition ──────────────────────────────────────────────

    #[test]
    fn composition_matches_sequential_rotation() {
        let q1 = Quaternion::from_axis_angle(Vector3::from_array([1.0, 0.0, 0.0]), FRAC_PI_2);
        let q2 = Quaternion::from_axis_angle(Vector3::from_array([0.0, 1.0, 0.0]), FRAC_PI_2);

        let v = Vector3::from_array([1.0, 0.0, 0.0]);

        // Apply q1 then q2 via composition
        let composed = q2 * q1;
        let r1 = composed * v;

        // Apply sequentially
        let r2 = q2 * (q1 * v);

        assert!(approx_eq(r1[0], r2[0]));
        assert!(approx_eq(r1[1], r2[1]));
        assert!(approx_eq(r1[2], r2[2]));
    }

    // ── Display ──────────────────────────────────────────────────

    #[test]
    fn display() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let s = format!("{}", q);
        assert_eq!(s, "(1 + 2i + 3j + 4k)");
    }

    // ── f32 ──────────────────────────────────────────────────────

    #[test]
    fn f32_basic() {
        let q = Quaternion::from_axis_angle(
            Vector3::from_array([0.0_f32, 0.0, 1.0]),
            core::f32::consts::FRAC_PI_2,
        );
        let v = Vector3::from_array([1.0_f32, 0.0, 0.0]);
        let r = q * v;
        assert!((r[0]).abs() < 1e-6);
        assert!((r[1] - 1.0).abs() < 1e-6);
    }
}
