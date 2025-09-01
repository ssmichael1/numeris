use super::*;

impl<T> DynArray<T>
where
    T: ArrayElem + std::ops::Add<Output = T> + Copy,
{
    /// Adds two arrays element-wise.
    ///
    /// # Errors
    /// Returns an error if the arrays have different shapes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tiny_matrix::prelude::*;
    /// let a = DynArray::from_vec(vec![1, 2, 3]);
    /// let b = DynArray::from_vec(vec![4, 5, 6]);
    /// let c = a.add_element_wise(&b);
    /// ```
    pub fn add_element_wise(&self, other: &Self) -> DynArrayResult<DynArray<T>> {
        if self.shape_ != other.shape_ {
            return Err(DynArrayError::DimMismatch);
        }
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a + *b)
            .collect();
        Ok(DynArray {
            data,
            shape_: self.shape_.clone(),
        })
    }
}

impl<T> DynArray<T>
where
    T: ArrayElem + std::ops::Sub<Output = T> + Copy,
{
    /// Subtracts two arrays element-wise.
    pub fn sub_element_wise(&self, other: &Self) -> DynArrayResult<DynArray<T>> {
        if self.shape_ != other.shape_ {
            return Err(DynArrayError::DimMismatch);
        }
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a - *b)
            .collect();
        Ok(DynArray {
            data,
            shape_: self.shape_.clone(),
        })
    }
}

impl<T> std::ops::Sub for DynArray<T>
where
    T: ArrayElem + std::ops::Sub<Output = T> + Copy,
{
    type Output = DynArrayResult<DynArray<T>>;

    fn sub(self, other: Self) -> Self::Output {
        self.sub_element_wise(&other)
    }
}

impl<T> std::ops::Add for DynArray<T>
where
    T: ArrayElem + std::ops::Add<Output = T> + Copy,
{
    type Output = DynArrayResult<DynArray<T>>;

    fn add(self, other: Self) -> Self::Output {
        self.add_element_wise(&other)
    }
}

impl<T> std::ops::Add<T> for DynArray<T>
where
    T: ArrayElem + std::ops::Add<Output = T> + Copy,
{
    type Output = DynArray<T>;

    fn add(self, other: T) -> Self::Output {
        self.map(|val| *val + other)
    }
}

impl<T> std::ops::Mul<T> for DynArray<T>
where
    T: ArrayElem + std::ops::Mul<Output = T> + Copy,
{
    type Output = DynArray<T>;

    fn mul(self, other: T) -> Self::Output {
        self.map(|val| *val * other)
    }
}

impl<T> std::ops::Div<T> for DynArray<T>
where
    T: ArrayElem + std::ops::Div<Output = T> + Copy,
{
    type Output = DynArray<T>;

    fn div(self, other: T) -> Self::Output {
        self.map(|val| *val / other)
    }
}

impl<T> std::ops::Sub<T> for DynArray<T>
where
    T: ArrayElem + std::ops::Sub<Output = T> + Copy,
{
    type Output = DynArray<T>;

    fn sub(self, other: T) -> Self::Output {
        self.map(|val| *val - other)
    }
}

impl<T> std::ops::Neg for DynArray<T>
where
    T: ArrayElem + std::ops::Neg<Output = T> + Copy,
{
    type Output = DynArray<T>;

    fn neg(self) -> Self::Output {
        self.map(|val| -(*val))
    }
}

impl<T> DynArray<T>
where
    T: ArrayElem + std::ops::Div<Output = T> + Copy,
{
    pub fn div_element_wise(&self, other: &Self) -> DynArrayResult<DynArray<T>> {
        if self.shape_ != other.shape_ {
            return Err(DynArrayError::DimMismatch);
        }
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a / *b)
            .collect();
        Ok(DynArray {
            data,
            shape_: self.shape_.clone(),
        })
    }
}

impl<T> std::ops::Div<DynArray<T>> for DynArray<T>
where
    T: ArrayElem + std::ops::Div<Output = T> + Copy,
{
    type Output = DynArrayResult<DynArray<T>>;

    fn div(self, other: Self) -> Self::Output {
        self.div_element_wise(&other)
    }
}

impl<T> DynArray<T>
where
    T: ArrayElem + std::ops::Mul<Output = T> + Copy,
{
    /// Multiplies two arrays element-wise.
    pub fn mul_element_wise(&self, other: &Self) -> DynArrayResult<DynArray<T>> {
        if self.shape_ != other.shape_ {
            return Err(DynArrayError::DimMismatch);
        }
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a * *b)
            .collect();
        Ok(DynArray {
            data,
            shape_: self.shape_.clone(),
        })
    }
}

impl<T> std::ops::Mul for DynArray<T>
where
    T: ArrayElem + std::ops::Mul<Output = T> + Copy,
{
    type Output = DynArrayResult<DynArray<T>>;

    fn mul(self, other: Self) -> Self::Output {
        self.mul_element_wise(&other)
    }
}

impl<T> std::ops::AddAssign<DynArray<T>> for DynArray<T>
where
    T: ArrayElem + std::ops::Add<Output = T> + Copy,
{
    fn add_assign(&mut self, other: DynArray<T>) {
        if self.shape_ != other.shape_ {
            panic!("Dimension mismatch in AddAssign");
        }
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a = *a + *b;
        }
    }
}

impl<T> std::ops::AddAssign<T> for DynArray<T>
where
    T: ArrayElem + std::ops::Add<Output = T> + Copy,
{
    fn add_assign(&mut self, other: T) {
        for a in self.data.iter_mut() {
            *a = *a + other;
        }
    }
}

impl<T> std::ops::MulAssign<T> for DynArray<T>
where
    T: ArrayElem + std::ops::Mul<Output = T> + Copy,
{
    fn mul_assign(&mut self, other: T) {
        for a in self.data.iter_mut() {
            *a = *a * other;
        }
    }
}

impl<T> std::ops::DivAssign<T> for DynArray<T>
where
    T: ArrayElem + std::ops::Div<Output = T> + Copy,
{
    fn div_assign(&mut self, other: T) {
        for a in self.data.iter_mut() {
            *a = *a / other;
        }
    }
}

impl<T> std::ops::SubAssign<DynArray<T>> for DynArray<T>
where
    T: ArrayElem + std::ops::Sub<Output = T> + Copy,
{
    fn sub_assign(&mut self, other: DynArray<T>) {
        if self.shape_ != other.shape_ {
            panic!("Dimension mismatch in SubAssign");
        }
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a = *a - *b;
        }
    }
}

impl<T> std::ops::SubAssign<T> for DynArray<T>
where
    T: ArrayElem + std::ops::Sub<Output = T> + Copy,
{
    fn sub_assign(&mut self, other: T) {
        for a in self.data.iter_mut() {
            *a = *a - other;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_dynarray_addition() {
        let arr1 = DynArray::uniform(1, &[2, 3]);
        let arr2 = DynArray::uniform(2, &[2, 3]);
        let result = arr1 + arr2;
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(*result.at_ref(&[0, 0]).unwrap(), 3);
    }
}
