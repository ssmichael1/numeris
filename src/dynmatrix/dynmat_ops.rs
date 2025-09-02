use super::{DynMatrix, DynMatrixError, DynMatrixResult};
use crate::matrix::MatrixElem;

impl<T> std::ops::Add<DynMatrix<T>> for DynMatrix<T>
where
    T: MatrixElem,
{
    type Output = DynMatrixResult<DynMatrix<T>>;

    fn add(self, other: DynMatrix<T>) -> Self::Output {
        if self.shape() != other.shape() {
            return Err(DynMatrixError::DimensionMismatch);
        }
        let mut result = DynMatrix::<T>::zeros(self.rows(), self.cols());
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] + other.data[i];
        }
        Ok(result)
    }
}

impl<T> std::ops::Add<T> for DynMatrix<T>
where
    T: MatrixElem + std::ops::Add<Output = T> + Copy,
{
    type Output = DynMatrix<T>;

    fn add(self, other: T) -> Self::Output {
        let mut result = DynMatrix::<T>::zeros(self.rows(), self.cols());
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] + other;
        }
        result
    }
}

impl<T> std::ops::AddAssign<T> for DynMatrix<T>
where
    T: MatrixElem + std::ops::AddAssign,
{
    fn add_assign(&mut self, other: T) {
        for i in 0..self.data.len() {
            self.data[i] += other;
        }
    }
}

impl<T> std::ops::Mul<T> for DynMatrix<T>
where
    T: MatrixElem + std::ops::Mul<Output = T> + Copy,
{
    type Output = DynMatrix<T>;

    fn mul(self, other: T) -> Self::Output {
        let mut result = DynMatrix::<T>::zeros(self.rows(), self.cols());
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] * other;
        }
        result
    }
}

impl<T> std::ops::Div<T> for DynMatrix<T>
where
    T: MatrixElem + std::ops::Div<Output = T> + Copy,
{
    type Output = DynMatrix<T>;

    fn div(self, other: T) -> Self::Output {
        let mut result = DynMatrix::<T>::zeros(self.rows(), self.cols());
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] / other;
        }
        result
    }
}

impl<T> std::ops::MulAssign<T> for DynMatrix<T>
where
    T: MatrixElem + std::ops::MulAssign,
{
    fn mul_assign(&mut self, other: T) {
        for i in 0..self.data.len() {
            self.data[i] *= other;
        }
    }
}

impl<T> std::ops::DivAssign<T> for DynMatrix<T>
where
    T: MatrixElem + std::ops::DivAssign,
{
    fn div_assign(&mut self, other: T) {
        for i in 0..self.data.len() {
            self.data[i] /= other;
        }
    }
}

impl<T> std::ops::Neg for DynMatrix<T>
where
    T: MatrixElem + std::ops::Neg<Output = T>,
{
    type Output = DynMatrix<T>;

    fn neg(self) -> Self::Output {
        let mut result = DynMatrix::<T>::zeros(self.rows(), self.cols());
        for i in 0..self.data.len() {
            result.data[i] = -self.data[i];
        }
        result
    }
}

impl<T> std::ops::Sub<T> for DynMatrix<T>
where
    T: MatrixElem + std::ops::Sub<Output = T> + Copy,
{
    type Output = DynMatrix<T>;

    fn sub(self, other: T) -> Self::Output {
        let mut result = DynMatrix::<T>::zeros(self.rows(), self.cols());
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] - other;
        }
        result
    }
}

impl<T> std::ops::SubAssign<T> for DynMatrix<T>
where
    T: MatrixElem + std::ops::SubAssign,
{
    fn sub_assign(&mut self, other: T) {
        for i in 0..self.data.len() {
            self.data[i] -= other;
        }
    }
}

impl<T> std::ops::Sub<DynMatrix<T>> for DynMatrix<T>
where
    T: MatrixElem,
{
    type Output = DynMatrixResult<DynMatrix<T>>;

    fn sub(self, other: DynMatrix<T>) -> Self::Output {
        if self.shape() != other.shape() {
            return Err(DynMatrixError::DimensionMismatch);
        }
        let mut result = DynMatrix::<T>::zeros(self.rows(), self.cols());
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] - other.data[i];
        }
        Ok(result)
    }
}

impl<T> std::ops::Index<(usize, usize)> for DynMatrix<T>
where
    T: MatrixElem,
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        if row >= self.rows() || col >= self.cols() {
            panic!("Index out of bounds");
        }
        &self.data[row * self.cols() + col]
    }
}

impl<T> std::ops::IndexMut<(usize, usize)> for DynMatrix<T>
where
    T: MatrixElem,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        let rows = self.rows();
        let cols = self.cols();
        if row >= rows || col >= cols {
            panic!("Index out of bounds");
        }
        &mut self.data[row * cols + col]
    }
}

impl<T> std::ops::Mul<DynMatrix<T>> for &DynMatrix<T>
where
    T: MatrixElem,
{
    type Output = DynMatrixResult<DynMatrix<T>>;

    fn mul(self, other: DynMatrix<T>) -> Self::Output {
        if self.cols() != other.rows() {
            return Err(DynMatrixError::DimensionMismatch);
        }
        let mut result = DynMatrix::<T>::zeros(self.rows(), other.cols());
        for i in 0..self.rows() {
            for j in 0..other.cols() {
                let mut sum = T::zero();
                for k in 0..self.cols() {
                    sum += self.data[i * self.cols() + k] * other.data[k * other.cols() + j];
                }
                result.data[i * other.cols() + j] = sum;
            }
        }
        Ok(result)
    }
}

impl<T> std::ops::Mul<&DynMatrix<T>> for &DynMatrix<T>
where
    T: MatrixElem,
{
    type Output = DynMatrixResult<DynMatrix<T>>;

    fn mul(self, other: &DynMatrix<T>) -> Self::Output {
        if self.cols() != other.rows() {
            return Err(DynMatrixError::DimensionMismatch);
        }
        let mut result = DynMatrix::<T>::zeros(self.rows(), other.cols());
        for i in 0..self.rows() {
            for j in 0..other.cols() {
                let mut sum = T::zero();
                for k in 0..self.cols() {
                    sum += self.data[i * self.cols() + k] * other.data[k * other.cols() + j];
                }
                result.data[i * other.cols() + j] = sum;
            }
        }
        Ok(result)
    }
}

impl<T> std::ops::Mul<DynMatrix<T>> for DynMatrix<T>
where
    T: MatrixElem,
{
    type Output = DynMatrixResult<DynMatrix<T>>;

    fn mul(self, other: DynMatrix<T>) -> Self::Output {
        if self.cols() != other.rows() {
            return Err(DynMatrixError::DimensionMismatch);
        }
        let mut result = DynMatrix::<T>::zeros(self.rows(), other.cols());
        for i in 0..self.rows() {
            for j in 0..other.cols() {
                let mut sum = T::zero();
                for k in 0..self.cols() {
                    sum += self.data[i * self.cols() + k] * other.data[k * other.cols() + j];
                }
                result.data[i * other.cols() + j] = sum;
            }
        }
        Ok(result)
    }
}
