use super::Quaternion;
use crate::matrix::Vector3;
use crate::rowmat;

impl<T> Quaternion<T>
where
    T: num_traits::Float + crate::matrix::MatrixElem,
{
    pub fn new(w: T, x: T, y: T, z: T) -> Self {
        Self { w, x, y, z }
    }

    /// Quaternion representing rotation of `angle` radians around the x-axis
    ///
    /// # Arguments
    ///
    /// * `angle` - The angle of rotation in radians.
    ///
    /// # Returns
    ///
    /// A quaternion representing right-handed rotation about X axis of vector
    /// by `angle` radians
    pub fn rotx(angle: T) -> Self {
        let half_angle = angle / T::from(2.0).unwrap();
        Self {
            w: half_angle.cos(),
            x: half_angle.sin(),
            y: T::zero(),
            z: T::zero(),
        }
    }

    /// Quaternion representing rotation of `angle` radians around the y-axis
    ///
    /// # Arguments
    ///
    /// * `angle` - The angle of rotation in radians.
    ///
    /// # Returns
    ///
    /// A quaternion representing right-handed rotation about Y axis of vector
    /// by `angle` radians
    pub fn roty(angle: T) -> Self {
        let half_angle = angle / T::from(2.0).unwrap();
        Self {
            w: half_angle.cos(),
            x: T::zero(),
            y: half_angle.sin(),
            z: T::zero(),
        }
    }

    /// Quaternion representing rotation of `angle` radians around the z-axis
    ///
    /// # Arguments
    ///
    /// * `angle` - The angle of rotation in radians.
    ///
    /// # Returns
    ///
    /// A quaternion representing right-handed rotation about Z axis of vector
    /// by `angle` radians
    pub fn rotz(angle: T) -> Self {
        let half_angle = angle / T::from(2.0).unwrap();
        Self {
            w: half_angle.cos(),
            x: T::zero(),
            y: T::zero(),
            z: half_angle.sin(),
        }
    }

    /// Unit quaternion: represents no rotation
    ///
    /// # Returns
    ///
    /// A quaternion representing no rotation.
    pub fn identity() -> Self {
        Self {
            w: T::one(),
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
        }
    }

    /// Conjugate of the quaternion
    ///
    /// # Returns
    ///
    /// The conjugate of the quaternion.
    pub fn conjugate(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Alias for `conjugate`
    ///
    /// # Returns
    ///
    /// The conjugate of the quaternion.
    pub fn conj(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Norm squared of the quaternion
    ///
    /// # Returns
    ///
    /// The squared norm of the quaternion
    pub fn norm_squared(&self) -> T {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Norm of the quaternion
    ///
    /// # Returns
    ///
    /// The norm of the quaternion
    pub fn norm(&self) -> T {
        self.norm_squared().sqrt()
    }

    /// Axis of rotation represented by the quaternion
    ///
    /// # Returns
    ///
    /// A vector representing the axis of rotation.
    pub fn axis(&self) -> Vector3<T> {
        rowmat![[self.x, self.y, self.z]]
            .normalized()
            .unwrap_or(rowmat![T::zero(), T::zero(), T::zero()])
    }

    /// Angle of rotation in radians
    ///
    /// # Returns
    ///
    /// The angle of rotation in radians.
    pub fn angle(&self) -> T {
        T::from(2.0).unwrap() * (self.w / self.norm()).acos()
    }

    /// Represent rotation as an axis-angle pair
    ///
    /// # Note:
    ///
    /// The quaternion represents a right-handed rotation of a vector
    /// about axis by the angle in radians
    ///
    /// # Returns
    ///
    /// A tuple containing the axis of rotation as a `Vector3<T>` and the angle of rotation in radians.
    ///
    pub fn as_axis_angle(&self) -> (Vector3<T>, T) {
        let axis = self.axis();
        let angle = self.angle();
        (axis, angle)
    }

    /// Create a quaternion representing a right-handed rotation of a
    /// vector about the given axis by the given angle in radians
    ///
    /// # Arguments:
    ///
    /// * `axis` - The axis of rotation as a `Vector3<T>`.
    /// * `angle` - The angle of rotation in radians.
    ///
    /// # Returns
    ///
    /// A quaternion representing the rotation.
    ///
    /// # Note:
    ///
    /// * If axis norm is zero, it is taken to be x axis
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let angle = std::f64::consts::PI / 2.0;
    /// // Rotate xhat by 90 degrees around z axis
    /// let rotated = Quaternion::from_axis_angle(Vector3d::zhat(), angle) * Vector3d::xhat();
    /// ```
    pub fn from_axis_angle(axis: Vector3<T>, angle: T) -> Self {
        let axis = axis
            .normalized()
            .unwrap_or(rowmat![T::one(), T::zero(), T::zero()]);
        let half_angle = angle / T::from(2.0).unwrap();
        Self {
            w: half_angle.cos(),
            x: axis[0] * half_angle.sin(),
            y: axis[1] * half_angle.sin(),
            z: axis[2] * half_angle.sin(),
        }
    }

    /// Conversion from a 3x3 rotation matrix
    ///
    /// # Arguments
    ///
    /// * `matrix` - A 3x3 rotation matrix.
    ///
    /// # Returns
    ///
    /// A quaternion representing the equivalent rotation as the input matrix
    ///
    /// # Example
    ///
    /// ```rust
    /// use numeris::prelude::*;
    /// let q = Quaternion::rotz(std::f64::consts::PI / 2.0);
    /// let m = q.as_dcm();
    /// let r1 = m * Vector3d::xhat();
    /// let r2 = q * Vector3d::xhat();
    /// for idx in 0..3 {
    ///     approx::assert_abs_diff_eq!(r1[idx], r2[idx], epsilon = 1e-10);
    /// }
    ///
    /// ```
    pub fn from_dcm(m: &crate::matrix::Matrix3<T>) -> Self {
        let trace = m[(0, 0)] + m[(1, 1)] + m[(2, 2)];
        let max = m[(0, 0)].max(m[(1, 1)].max(m[(2, 2)])).max(trace);
        let qmax = T::from(0.5).unwrap() * (T::one() - trace + T::from(2.0).unwrap() * max).sqrt();
        let qmax4 = qmax * T::from(4.0).unwrap();
        let mut q = {
            if m[(0, 0)] > m[(1, 1)] && m[(0, 0)] > m[(2, 2)] && m[(0, 0)] > trace {
                Self {
                    w: (-m[(2, 1)] - m[(1, 2)]) / qmax4,
                    x: qmax,
                    y: (m[(0, 1)] + m[(1, 0)]) / qmax4,
                    z: (m[(0, 2)] + m[(2, 0)]) / qmax4,
                }
            } else if m[(1, 1)] > m[(2, 2)] && m[(1, 1)] > trace {
                Self {
                    w: (m[(0, 2)] - m[(2, 0)]) / qmax4,
                    x: (m[(0, 1)] + m[(1, 0)]) / qmax4,
                    y: qmax,
                    z: (m[(1, 2)] + m[(2, 1)]) / qmax4,
                }
            } else if m[(2, 2)] > trace {
                Self {
                    w: (m[(1, 0)] - m[(0, 1)]) / qmax4,
                    x: (m[(0, 2)] + m[(2, 0)]) / qmax4,
                    y: (m[(1, 2)] + m[(2, 1)]) / qmax4,
                    z: qmax,
                }
            } else {
                Self {
                    w: qmax,
                    x: (m[(2, 1)] - m[(1, 2)]) / qmax4,
                    y: (m[(0, 2)] - m[(2, 0)]) / qmax4,
                    z: (m[(1, 0)] - m[(0, 1)]) / qmax4,
                }
            }
        };
        // Pick w to be positive
        if q.w < T::zero() {
            q = Self {
                w: -q.w,
                x: -q.x,
                y: -q.y,
                z: -q.z,
            };
        }
        q = q.normalized().unwrap_or(Self::identity());
        q
    }

    pub fn as_dcm(&self) -> crate::matrix::Matrix3<T> {
        let (axis, angle) = self.as_axis_angle();
        let cos = angle.cos();
        let msin = -angle.sin();
        let one_minus_cos = T::one() - cos;

        crate::matrix::Matrix3::new([
            [
                cos + axis[0] * axis[0] * one_minus_cos,
                axis[0] * axis[1] * one_minus_cos - axis[2] * msin,
                axis[0] * axis[2] * one_minus_cos + axis[1] * msin,
            ],
            [
                axis[1] * axis[0] * one_minus_cos + axis[2] * msin,
                cos + axis[1] * axis[1] * one_minus_cos,
                axis[1] * axis[2] * one_minus_cos - axis[0] * msin,
            ],
            [
                axis[2] * axis[0] * one_minus_cos - axis[1] * msin,
                axis[2] * axis[1] * one_minus_cos + axis[0] * msin,
                cos + axis[2] * axis[2] * one_minus_cos,
            ],
        ])
    }

    /// Normalized quaternion
    ///
    /// # Returns
    ///
    /// The normalized quaternion, or `None` if the quaternion is zero.
    pub fn normalized(&self) -> Option<Self> {
        let norm = self.norm();
        if norm == T::zero() {
            return None;
        }
        Some(Self {
            w: self.w / norm,
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        })
    }
}

impl<T> Default for Quaternion<T>
where
    T: num_traits::Float + crate::matrix::MatrixElem,
{
    fn default() -> Self {
        Self::identity()
    }
}

impl<T> std::fmt::Display for Quaternion<T>
where
    T: num_traits::Float + std::fmt::Display + crate::matrix::MatrixElem,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (axis, angle) = self.as_axis_angle();
        write!(
            f,
            "Quaternion(Axis: |{:.3}, {:.3}, {:.3}|, Angle: {:.3} rad)",
            axis[0], axis[1], axis[2], angle
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_quaternion_norm() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(q.norm(), 5.477225575051661);
    }

    #[test]
    fn test_quaternion_conjugate() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(q.conjugate(), Quaternion::new(1.0, -2.0, -3.0, -4.0));
    }

    #[test]
    fn test_dcm() {
        let m =
            Matrix3::<f64>::from_row_major([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        let q = Quaternion::from_dcm(&m);

        for r in 0..3 {
            for c in 0..3 {
                approx::assert_abs_diff_eq!(m[(r, c)], q.as_dcm()[(r, c)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_quaternion_display() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        println!("{}", q);
    }

    #[test]
    fn test_quaternion_as_dcm() {
        let q = Quaternion::rotz(std::f64::consts::PI / 2.0);
        let dcm = q.as_dcm();
        let v1 = q * Vector3d::xhat();
        let v2 = dcm * Vector3d::xhat();
        for idx in 0..3 {
            approx::assert_abs_diff_eq!(v1[idx], v2[idx], epsilon = 1e-10);
        }
    }
}
