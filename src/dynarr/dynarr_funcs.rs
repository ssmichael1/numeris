use super::*;

/// Functions on arrays with floating-point elements
impl<T> DynArray<T>
where
    T: ArrayElem + num_traits::Float,
{
    /// Sum of squares of all elements in the array
    pub fn norm_squared(&self) -> T {
        self.data.iter().fold(T::zero(), |acc, &x| acc + x * x)
    }

    /// Euclidean norm (magnitude) of the array
    pub fn norm(&self) -> T {
        self.norm_squared().sqrt()
    }

    /// Cosine of all elements in the array
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1.0, &[2, 3]);
    /// let cos_array = array.cos();
    /// ```
    pub fn cos(&self) -> DynArray<T> {
        self.map(|x| x.cos())
    }

    /// Sine of all elements in the array
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1.0, &[2, 3]);
    /// let sin_array = array.sin();
    /// ```
    pub fn sin(&self) -> DynArray<T> {
        self.map(|x| x.sin())
    }

    /// Tangent of all elements in the array
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1.0, &[2, 3]);
    /// let tan_array = array.tan();
    /// ```
    pub fn tan(&self) -> DynArray<T> {
        self.map(|x| x.tan())
    }

    /// Exponent of all elements in the array
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1.0, &[2, 3]);
    /// let exp_array = array.exp();
    /// ```
    pub fn exp(&self) -> DynArray<T> {
        self.map(|x| x.exp())
    }

    /// Natural logarithm of all elements in the array
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1.0, &[2, 3]);
    /// let log_array = array.log();
    /// ```
    pub fn log(&self) -> DynArray<T> {
        self.map(|x| x.ln())
    }

    /// Square root of all elements in the array
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1.0, &[2, 3]);
    /// let sqrt_array = array.sqrt();
    /// ```
    pub fn sqrt(&self) -> DynArray<T> {
        self.map(|x| x.sqrt())
    }

    /// Absolute value of all elements in the array
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(-1.0, &[2, 3]);
    /// let abs_array = array.abs();
    /// ```
    pub fn abs(&self) -> DynArray<T> {
        self.map(|x| x.abs())
    }

    /// Element-wise arc cosine of all elements in the array
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1.0, &[2, 3]);
    /// let acos_array = array.acos();
    /// ```
    pub fn acos(&self) -> DynArray<T> {
        self.map(|x| x.acos())
    }

    /// Element-wise arc sine of all elements in the array
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1.0, &[2, 3]);
    /// let asin_array = array.asin();
    /// ```
    pub fn asin(&self) -> DynArray<T> {
        self.map(|x| x.asin())
    }

    /// Element-wise arc-tangent of all elements in the array
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1.0, &[2, 3]);
    /// let atan_array = array.atan();
    /// ```
    pub fn atan(&self) -> DynArray<T> {
        self.map(|x| x.atan())
    }

    /// Element-wise "ceiling" of all elements in the array
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1.5, &[2, 3]);
    /// let ceil_array = array.ceil();
    /// ```
    pub fn ceil(&self) -> DynArray<T> {
        self.map(|x| x.ceil())
    }

    /// Element-wise "floor" of all elements in the array
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1.5, &[2, 3]);
    /// let floor_array = array.floor();
    /// assert!(floor_array == DynArray::uniform(1.0, &[2, 3]));
    /// ```
    pub fn floor(&self) -> DynArray<T> {
        self.map(|x| x.floor())
    }

    /// Element-wise hyperbolic sine of all elements in the array
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1.0, &[2, 3]);
    /// let sinh_array = array.sinh();
    /// ```
    pub fn sinh(&self) -> DynArray<T> {
        self.map(|x| x.sinh())
    }

    /// Element-wise hyperbolic cosine of all elements in the array
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1.0, &[2, 3]);
    /// let cosh_array = array.cosh();
    /// ```
    pub fn cosh(&self) -> DynArray<T> {
        self.map(|x| x.cosh())
    }

    /// Element-wise Quadrant-Aware Arctangent of all elements in the array
    ///
    /// # Notes:
    /// "Self" is y, "other" is x and arctan is y/x
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array1 = DynArray::uniform(1.0, &[2, 3]);
    /// let array2 = DynArray::uniform(2.0, &[2, 3]);
    /// let result = array1.atan2(&array2);
    /// ```
    pub fn atan2(&self, other: &DynArray<T>) -> DynArrayResult<DynArray<T>> {
        if self.shape() != other.shape() {
            return Err(DynArrayError::ShapeMismatch);
        }
        Ok(self.zip_map(other, |y: &T, x: &T| y.atan2(*x)))
    }

    /// Element-wise sign of all elements in the array
    ///
    /// # Returns
    ///
    /// A new array with the sign of each element. (1.0 = positive, -1.0 = negative)
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(-1.0, &[2, 3]);
    /// let sign_array = array.signum();
    /// ```
    pub fn signum(&self) -> DynArray<T> {
        self.map(|x| x.signum())
    }

    /// Element-wise convert from radians to degrees
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(90.0, &[2, 3]);
    /// let degrees_array = array.to_degrees();
    /// ```
    pub fn to_degrees(&self) -> DynArray<T> {
        self.map(|x| x.to_degrees())
    }

    /// Element-wise convert from degrees to radians
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(90.0, &[2, 3]);
    /// let radians_array = array.to_radians();
    /// ```
    pub fn to_radians(&self) -> DynArray<T> {
        self.map(|x| x.to_radians())
    }

    /// Element-wise log10 of array
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(10.0, &[2, 3]);
    /// let log_array = array.log10();
    /// ```
    pub fn log10(&self) -> DynArray<T> {
        self.map(|x| x.log10())
    }
}
