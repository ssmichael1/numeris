use super::{ArrayElem, DynArray, DynArrayError, DynArrayResult};

impl<T> std::ops::Index<&[usize]> for DynArray<T>
where
    T: ArrayElem,
{
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        &self.data[self.flat_index_unchecked(index)]
    }
}

impl<T> std::ops::IndexMut<&[usize]> for DynArray<T>
where
    T: ArrayElem,
{
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        let flat_index = self.flat_index_unchecked(index);
        &mut self.data[flat_index]
    }
}

impl<T> DynArray<T>
where
    T: ArrayElem + num_traits::Zero,
{
    pub fn zeros(shape: &[usize]) -> Self {
        let total_size = shape.iter().cloned().reduce(|a, b| a * b).unwrap_or(0);
        let data = vec![T::zero(); total_size];
        Self {
            data,
            shape_: shape.to_vec(),
        }
    }
}

impl<T> DynArray<T>
where
    T: ArrayElem + num_traits::One,
{
    pub fn ones(shape: &[usize]) -> Self {
        let total_size = shape.iter().cloned().reduce(|a, b| a * b).unwrap_or(0);
        let data = vec![T::one(); total_size];
        Self {
            data,
            shape_: shape.to_vec(),
        }
    }
}

impl<T> DynArray<T>
where
    T: ArrayElem,
{
    /// Creates a new uniform array with the given value and shape.
    ///
    /// # Example:
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1, &[2, 3]);
    /// assert_eq!(array.at(&[1, 2]).unwrap(), 1);
    /// ```
    pub fn uniform(value: T, shape: &[usize]) -> Self {
        let total_size = shape.iter().cloned().reduce(|a, b| a * b).unwrap_or(0);
        let data = vec![value; total_size];
        Self {
            data,
            shape_: shape.to_vec(),
        }
    }

    /// Returns an iterator over the indices of the elements that match the predicate.
    ///
    /// # Example:
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let mut array = DynArray::uniform(0, &[2, 3]);
    /// array[&[0,0]] = 1;
    /// let indices: Vec<_> = array.iter_where(|&x| x == 1).collect();
    /// assert_eq!(indices, vec![vec![0, 0]]);
    /// ```
    pub fn iter_where<'a, F>(&'a self, predicate: F) -> impl Iterator<Item = Vec<usize>> + 'a
    where
        F: Fn(&T) -> bool + 'a, // Tie F's lifetime to &self
    {
        self.data
            .iter()
            .enumerate()
            .filter(move |(_idx, val)| predicate(val))
            .map(move |(idx, _val)| self.flat_to_index(idx))
    }

    /// Create a 1-dimensional array from a vector,
    /// consuming the vector
    ///
    /// # Example:
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let vec = vec![1, 2, 3];
    /// let array = DynArray::from_vec(vec);
    /// assert_eq!(array.shape(), &[3]);
    /// ```
    pub fn from_vec(vec: Vec<T>) -> Self {
        let shape = vec![vec.len()];
        Self {
            data: vec,
            shape_: shape,
        }
    }

    fn flat_index_unchecked(&self, index: &[usize]) -> usize {
        let mut flat_index = 0;
        let mut stride = 1;
        for (i, &dim_size) in self.shape_.iter().rev().enumerate() {
            let idx = index[self.shape_.len() - 1 - i];
            flat_index += idx * stride;
            stride *= dim_size;
        }
        flat_index
    }

    // Convert flat to multi-dimensional index
    fn flat_to_index(&self, flat_index: usize) -> Vec<usize> {
        let mut index = Vec::new();
        let mut remaining = flat_index;
        for &dim_size in &self.shape_ {
            index.push(remaining % dim_size);
            remaining /= dim_size;
        }
        index.reverse();
        index
    }

    /// Flat index from multi-dimensional index
    fn flat_index(&self, index: &[usize]) -> DynArrayResult<usize> {
        if index.len() != self.shape_.len() {
            return Err(DynArrayError::DimMismatch);
        }
        let mut flat_index = 0;
        let mut stride = 1;
        for (i, &dim_size) in self.shape_.iter().rev().enumerate() {
            let idx = index[self.shape_.len() - 1 - i];
            if idx >= dim_size {
                return Err(DynArrayError::OutOfBounds);
            }
            flat_index += idx * stride;
            stride *= dim_size;
        }
        Ok(flat_index)
    }

    /// Returns value at given index, with bounds checking
    ///
    /// # Returns:
    ///
    /// The value at the given index, or an error if the index is out of bounds.
    ///
    /// # Example:
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1, &[2, 3]);
    /// assert_eq!(array.at(&[1, 2]).unwrap(), 1);
    /// ```
    pub fn at(&self, index: &[usize]) -> DynArrayResult<T> {
        Ok(self.data[self.flat_index(index)?])
    }

    /// Returns value at given index by reference, with bounds checking
    ///
    /// # Returns:
    ///
    /// The value at the given index, or an error if the index is out of bounds.
    ///
    /// # Example:
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1, &[2, 3]);
    /// assert_eq!(*array.at_ref(&[1, 2]).unwrap(), 1);
    /// ```
    pub fn at_ref(&self, index: &[usize]) -> DynArrayResult<&T> {
        Ok(&self.data[self.flat_index(index)?])
    }

    /// Returns mutable reference to value at given index, with bounds checking
    ///
    /// # Returns:
    ///
    /// A mutable reference to the value at the given index, or an error if the index is out of bounds.
    ///
    /// # Example:
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let mut array = DynArray::uniform(1, &[2, 3]);
    /// *array.at_mut(&[1, 2]).unwrap() = 4;
    /// assert_eq!(*array.at_ref(&[1, 2]).unwrap(), 4);
    /// ```
    pub fn at_mut(&mut self, index: &[usize]) -> DynArrayResult<&mut T> {
        let flat_index = self.flat_index(index)?;
        Ok(&mut self.data[flat_index])
    }

    /// Returns the shape of the array.
    ///
    /// Returns:
    ///
    /// The shape of the array.
    ///
    /// Example:
    ///
    /// ```rust
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1, &[2, 3]);
    /// assert_eq!(array.shape(), &[2, 3]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape_
    }

    /// Create an array from an iterator
    ///
    /// # Errors
    /// Returns an error if the number of elements does not match the shape.    
    ///
    /// # Example
    /// ```rust
    /// use numeris::prelude::*;
    /// let array = DynArray::from_iter(0..6, &[2, 3]).unwrap();
    /// assert_eq!(array.shape(), &[2, 3]);
    /// ```
    pub fn from_iter<I>(iter: I, shape: &[usize]) -> DynArrayResult<Self>
    where
        I: IntoIterator<Item = T>,
    {
        let data: Vec<T> = iter.into_iter().collect();
        let shape_ = shape.to_vec();
        if data.len() != shape_.iter().cloned().product::<usize>() {
            return Err(DynArrayError::ShapeMismatch);
        }
        Ok(DynArray { data, shape_ })
    }

    /// Returns the number of dimensions of the array.
    ///
    /// Example:
    ///
    /// ```rust
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1, &[2, 3]);
    /// assert_eq!(array.ndims(), 2);
    /// ```
    pub fn ndims(&self) -> usize {
        self.shape_.len()
    }

    /// Return the total number of elements in the array.
    ///
    /// Example:
    ///
    /// ```rust
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1, &[2, 3]);
    /// assert_eq!(array.size(), 6);
    /// ```
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Map a function over elements of the array, with array index as argument.
    ///
    /// # Example:
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1, &[2, 3]);
    /// let mapped = array.map_with_index(|idx, val| (*val + idx[0]) as f32);
    /// assert_eq!(*mapped.at_ref(&[1, 2]).unwrap(), 3.0);
    /// ```
    pub fn map_with_index<F, U>(&self, f: F) -> DynArray<U>
    where
        F: Fn(&[usize], &T) -> U,
        U: ArrayElem,
    {
        let data = self
            .data
            .iter()
            .enumerate()
            .map(|(i, elem)| f(&self.flat_to_index(i), elem))
            .collect();
        DynArray {
            data,
            shape_: self.shape_.clone(),
        }
    }

    /// Element-wise zip map of two arrays
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array1 = DynArray::uniform(1.0, &[2, 3]);
    /// let array2 = DynArray::uniform(2.0, &[2, 3]);
    /// let result = array1.zip_map(&array2, |x, y| x + y);
    /// ```
    pub fn zip_map<U, F>(&self, other: &DynArray<U>, f: F) -> DynArray<T>
    where
        F: Fn(&T, &U) -> T,
        U: ArrayElem,
    {
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(x, y)| f(x, y))
            .collect();
        DynArray {
            data,
            shape_: self.shape_.clone(),
        }
    }

    /// Maps a function over the elements of the array.
    ///
    /// # Example:
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::uniform(1, &[2, 3]);
    /// let mapped = array.map(|x| *x + 1);
    /// assert_eq!(*mapped.at_ref(&[1, 2]).unwrap(), 2);
    /// ```
    pub fn map<F, U>(&self, f: F) -> DynArray<U>
    where
        F: Fn(&T) -> U,
        U: ArrayElem,
    {
        let data = self.data.iter().map(f).collect();
        DynArray {
            data,
            shape_: self.shape_.clone(),
        }
    }

    /// In-place removal of singleton dimensions
    ///
    /// This function removes dimensions of size 1 from the shape of the array.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let mut array = DynArray::uniform(1, &[1, 2, 1]);
    /// array.squeeze();
    /// assert_eq!(array.shape(), &[2]);
    /// ```
    pub fn squeeze(&mut self) {
        self.shape_.retain(|&dim| dim != 1);
        if self.shape_.is_empty() {
            self.shape_.push(1);
        }
    }

    /// Consume self and represent data as a dynamic matrix
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    /// let array = DynArray::from_iter(0..6, &[2, 3]).unwrap();
    /// let m = array.as_dynmatrix().unwrap();
    /// assert_eq!(m.shape(), (2, 3));
    /// ```
    pub fn as_dynmatrix(self) -> DynArrayResult<crate::prelude::DynMatrix<T>>
    where
        T: ArrayElem + crate::matrix::MatrixElem,
    {
        if self.shape_.len() != 2 {
            return Err(DynArrayError::DimMismatch);
        }
        Ok(crate::prelude::DynMatrix {
            data: self.data,
            rows_: self.shape_[0],
            cols_: self.shape_[1],
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_dynarray() {
        let arr = DynArray::uniform(1, &[2, 3]);
        assert_eq!(*arr.at_ref(&[1, 2]).unwrap(), 1);
    }

    #[test]
    fn test_squeeze() {
        let mut array = DynArray::uniform(1, &[1, 2, 1]);
        array.squeeze();
        assert_eq!(array.shape(), &[2]);
    }
}
