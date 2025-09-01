use super::Quaternion;
use crate::matrix::MatrixElem;
use crate::matrix::Vector3;
use crate::rowmat;

/// Multiply two quaternions
impl<T> std::ops::Mul for Quaternion<T>
where
    T: num_traits::Float + MatrixElem,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }
}

/// Multiply Quaternion by reference Quaternion
impl<T> std::ops::Mul<&Quaternion<T>> for Quaternion<T>
where
    T: num_traits::Float + MatrixElem,
{
    type Output = Self;

    fn mul(self, other: &Self) -> Self::Output {
        Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }
}

/// Multiply reference quaternion by quaternion
impl<T> std::ops::Mul<Quaternion<T>> for &Quaternion<T>
where
    T: num_traits::Float + MatrixElem,
{
    type Output = Quaternion<T>;

    fn mul(self, other: Quaternion<T>) -> Self::Output {
        Quaternion {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }
}

/// Multiply two reference quaternions
impl<T> std::ops::Mul<&Quaternion<T>> for &Quaternion<T>
where
    T: num_traits::Float + MatrixElem,
{
    type Output = Quaternion<T>;

    fn mul(self, other: &Quaternion<T>) -> Self::Output {
        Quaternion {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }
}

/// Multiply (rotate) a vector by a quaternion
impl<T> std::ops::Mul<Vector3<T>> for Quaternion<T>
where
    T: num_traits::Float + MatrixElem,
{
    type Output = Vector3<T>;

    fn mul(self, vec: Vector3<T>) -> Self::Output {
        let q_vec = Quaternion {
            w: T::zero(),
            x: vec[0],
            y: vec[1],
            z: vec[2],
        };
        let res = self * q_vec * self.conjugate();
        rowmat![res.x, res.y, res.z]
    }
}

/// Multiply (rotate) a vector by a quaternion reference
impl<T> std::ops::Mul<Vector3<T>> for &Quaternion<T>
where
    T: num_traits::Float + MatrixElem,
{
    type Output = Vector3<T>;

    fn mul(self, vec: Vector3<T>) -> Self::Output {
        let q_vec = Quaternion {
            w: T::zero(),
            x: vec[0],
            y: vec[1],
            z: vec[2],
        };
        let res = self * q_vec * self.conjugate();
        rowmat![res.x, res.y, res.z]
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::matrix::Vector3d;
    use std::f64::consts::PI;

    #[test]
    fn test_quaternion_multiplication() {
        let q1 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let q2 = Quaternion::new(5.0, 6.0, 7.0, 8.0);
        let result = q1 * q2;
        assert_eq!(result, Quaternion::new(-60.0, 12.0, 30.0, 24.0));
    }

    #[test]
    fn test_quaternion_rotation() {
        let q = Quaternion::rotx(PI / 2.0);
        let rotated = q * Vector3d::zhat();
        for idx in 0..3 {
            approx::assert_abs_diff_eq!(rotated[idx], -Vector3d::yhat()[idx], epsilon = 1e-10);
        }
        let q = Quaternion::rotz(PI / 2.0);
        let rotated = q * Vector3d::xhat();
        for idx in 0..3 {
            approx::assert_abs_diff_eq!(rotated[idx], Vector3d::yhat()[idx], epsilon = 1e-10);
        }
    }
}
