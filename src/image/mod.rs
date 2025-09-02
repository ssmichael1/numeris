mod pixeltype;
pub mod rgb;

use std::ops::{Index, IndexMut};

pub use pixeltype::PixelType;

pub struct Image<T>
where
    T: PixelType,
{
    pub(crate) width_: usize,
    pub(crate) height_: usize,
    pub(crate) data: Vec<T>,
}

impl<T> Image<T>
where
    T: PixelType,
{
    /// Returns the width of the image.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    ///
    /// let img = Image::<i32>::ones(100, 200);
    /// assert_eq!(img.width(), 100);
    /// ```
    pub fn width(&self) -> usize {
        self.width_
    }

    /// Returns the height of the image.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    ///
    /// let img = Image::<i32>::ones(100, 200);
    /// assert_eq!(img.height(), 200);
    /// ```
    pub fn height(&self) -> usize {
        self.height_
    }

    /// Returns the shape of the image as (width, height).
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    ///
    /// let img = Image::<i32>::ones(100, 200);
    /// assert_eq!(img.shape(), (100, 200));
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        (self.width_, self.height_)
    }

    /// Returns the total number of pixels in the image.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    ///
    /// let img = Image::<i32>::ones(100, 200);
    /// assert_eq!(img.size(), 100 * 200);
    /// ```
    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &Vec<T> {
        &self.data
    }
    /// Applies a function to each pixel in the image, returning a new image.
    ///
    /// This function does not modify the original image.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    ///
    /// let img = Image::<i32>::ones(100, 100);
    /// let mapped = img.map(|v| v + 1);
    /// ```
    pub fn map<F, U>(&self, f: F) -> Image<U>
    where
        F: Fn(&T) -> U,
        U: PixelType + Clone,
    {
        let data = self.data.iter().map(f).collect();
        Image {
            width_: self.width_,
            height_: self.height_,
            data,
        }
    }

    /// Converts 2D coordinates (x, y) to a 1D index.
    /// Note: not range checked since it is used only internally
    fn index_to_flat(&self, x: usize, y: usize) -> Option<usize> {
        if x < self.width_ && y < self.height_ {
            Some(y * self.width_ + x)
        } else {
            None
        }
    }

    fn flat_to_index(&self, flat: usize) -> (usize, usize) {
        let x = flat % self.width_;
        let y = flat / self.width_;
        (x, y)
    }

    /// Converts a 1D index to 2D coordinates (x, y).
    pub fn at(&self, x: usize, y: usize) -> Option<&T> {
        self.index_to_flat(x, y).and_then(|idx| self.data.get(idx))
    }
    pub fn at_mut(&mut self, x: usize, y: usize) -> Option<&mut T> {
        self.index_to_flat(x, y)
            .and_then(move |idx| self.data.get_mut(idx))
    }

    /// Applies a function to each pixel in the image, returning a new image.
    ///
    /// This function does not modify the original image.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    ///
    /// let img = Image::<i32>::ones(100, 100);
    /// let mapped = img.map(|v| v + 1);
    /// ```
    pub fn map_with_index<F, U>(&self, f: F) -> Image<U>
    where
        F: Fn(&T, usize, usize) -> U,
        U: PixelType + Clone,
    {
        let data = self
            .data
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let (x, y) = self.flat_to_index(i);
                f(v, x, y)
            })
            .collect();

        Image {
            width_: self.width_,
            height_: self.height_,
            data,
        }
    }

    /// Applies a function to each pixel in the image, modifying the original image.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    ///
    /// let mut img = Image::<i32>::ones(100, 100);
    /// img.map_in_place(|v| v + 1);
    /// ```
    pub fn map_in_place<F>(&mut self, f: F)
    where
        F: Fn(&T) -> T,
    {
        self.data.iter_mut().for_each(|v| *v = f(v));
    }
}

impl<T> Image<T>
where
    T: PixelType + num_traits::PrimInt,
{
    /// Clamps the pixel values of the image to the specified range.
    ///
    /// This function modifies the image in place, clamping all pixel values
    /// to be within the [min, max] range.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    ///
    /// let mut img = Image::<i32>::ones(100, 100);
    /// img.clamp_in_place(0, 255);
    /// ```
    pub fn clamp_in_place(&mut self, min: T, max: T) {
        self.data.iter_mut().for_each(|v| {
            if *v < min {
                *v = min;
            } else if *v > max {
                *v = max;
            }
        });
    }
}

impl<T> Image<T>
where
    T: PixelType + num_traits::Zero,
{
    pub fn zeros(width: usize, height: usize) -> Self {
        let data = vec![T::zero(); width * height];
        Self {
            width_: width,
            height_: height,
            data,
        }
    }
}

impl<T> Image<T>
where
    T: PixelType + num_traits::One,
{
    pub fn ones(width: usize, height: usize) -> Self {
        let data = vec![T::one(); width * height];
        Self {
            width_: width,
            height_: height,
            data,
        }
    }
}

impl<T> Index<(usize, usize)> for Image<T>
where
    T: PixelType,
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.at(index.0, index.1).unwrap()
    }
}

impl<T> Index<usize> for Image<T>
where
    T: PixelType,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for Image<T>
where
    T: PixelType,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T> IndexMut<(usize, usize)> for Image<T>
where
    T: PixelType,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.at_mut(index.0, index.1).unwrap()
    }
}
