mod pixeltype;

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
    T: PixelType + Clone,
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

    /// Converts a 1D index to 2D coordinates (x, y).
    /// Note: not range checked since it is used only internally
    fn flat_to_index(&self, flat: usize) -> (usize, usize) {
        let x = flat % self.width_;
        let y = flat / self.width_;
        (x, y)
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
    T: PixelType + num_traits::Zero + Clone,
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
    T: PixelType + num_traits::One + Clone,
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
