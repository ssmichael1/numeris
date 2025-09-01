use super::{Matrix, MatrixElem, Vector3};
use crate::rowmat;

impl<const ROWS: usize, T> Matrix<ROWS, 1, T>
where
    T: MatrixElem + num_traits::Float,
{
    /// Vector dot product with another vector
    pub fn dot(&self, other: Self) -> T {
        self.data[0]
            .iter()
            .zip(other.data[0].iter())
            .map(|(a, b)| *a * *b)
            .sum()
    }

    /// Norm squared of vector
    /// Computes the sum of squares of the elements
    pub fn norm_squared(&self) -> T {
        self.data[0].iter().map(|row| *row * *row).sum()
    }

    /// Norm of vector
    /// Computes the square root of the sum of squares of the elements
    pub fn norm(&self) -> T {
        self.norm_squared().sqrt()
    }

    /// Normalized vector
    ///
    /// # Returns
    ///
    /// The normalized vector, or `None` if the vector is zero.
    pub fn normalized(&self) -> Option<Self> {
        let norm = self.norm();
        if norm == T::zero() {
            return None; // Cannot normalize zero vector
        }
        let mut result = *self;
        result.data[0].iter_mut().for_each(|elem| *elem /= norm);
        Some(result)
    }
}

impl<const ROWS: usize, T> std::ops::Index<usize> for Matrix<ROWS, 1, T>
where
    T: MatrixElem,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= ROWS {
            panic!("Index out of bounds");
        }
        &self.data[0][index]
    }
}

impl<const ROWS: usize, T> From<[T; ROWS]> for Matrix<ROWS, 1, T>
where
    T: MatrixElem,
{
    fn from(arr: [T; ROWS]) -> Self {
        Self { data: [arr] }
    }
}

impl<const ROWS: usize, T> std::ops::IndexMut<usize> for Matrix<ROWS, 1, T>
where
    T: MatrixElem,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= ROWS {
            panic!("Index out of bounds");
        }
        &mut self.data[0][index]
    }
}

/// Implementation for 3-element vectors
impl<T> Vector3<T>
where
    T: MatrixElem + num_traits::Float,
{
    /// Cross product of two 3-element vectors
    ///
    /// # Returns:
    ///
    /// The cross product of the two vectors.
    pub fn cross(&self, other: &Self) -> Self {
        let x = self[1] * other[2] - self[2] * other[1];
        let y = self[2] * other[0] - self[0] * other[2];
        let z = self[0] * other[1] - self[1] * other[0];
        rowmat![[x, y, z]]
    }

    // xhat unit vector
    pub fn xhat() -> Self {
        rowmat![[T::one(), T::zero(), T::zero()]]
    }

    // yhat unit vector
    pub fn yhat() -> Self {
        rowmat![[T::zero(), T::one(), T::zero()]]
    }

    // zhat unit vector
    pub fn zhat() -> Self {
        rowmat![[T::zero(), T::zero(), T::one()]]
    }
}

#[cfg(test)]
mod tests {

    use crate::matrix::Vector3d;
    use crate::rowmat;

    #[test]
    fn test_norm_squared() {
        let vec = crate::rowmat!([1.0, 2.0, 3.0]);
        assert_eq!(vec.norm_squared(), 14.0);
    }

    #[test]
    fn test_norm() {
        //let vec = crate::vector!([3.0, 4.0]);
        //assert_eq!(vec.norm(), 5.0);
    }

    #[test]
    fn test_cross() {
        let zhat = Vector3d::xhat().cross(&Vector3d::yhat());
        assert_eq!(zhat, Vector3d::zhat());
        let yhat = Vector3d::zhat().cross(&Vector3d::xhat());
        assert_eq!(yhat, Vector3d::yhat());
        let xhat = Vector3d::yhat().cross(&Vector3d::zhat());
        assert_eq!(xhat, Vector3d::xhat());
    }

    #[test]
    fn test_dot() {
        let a = Vector3d::xhat();
        let b = Vector3d::yhat();
        assert_eq!(a.dot(b), 0.0);

        let a = rowmat![[1.0, 2.0, 3.0]];
        let b = rowmat![[4.0, 5.0, 6.0]];
        assert_eq!(a.dot(b), 32.0);
    }
}
