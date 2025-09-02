use super::{Image, PixelType};

impl<T> Image<T> where T: PixelType + num_traits::PrimInt
{

    /// Applies gamma correction to the image in place.
    ///
    /// This function modifies the original image, adjusting the pixel values
    /// according to the specified gamma curve.
    ///
    /// # Arguments
    ///
    /// * `gamma` - The gamma value to use for correction.
    /// * `maxval` - An optional maximum value to use for normalization.
    ///   If not passed in, maximum value for the data type is used
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    ///
    /// let mut img = Image::<i32>::ones(100, 100);
    /// img.gamma_correction_in_place(2.2, None);
    /// ```
    pub fn gamma_correction_in_place(&mut self, gamma: f32, maxval: Option<T>)
    {
        self.data.iter_mut().for_each(|v| {
            let mv = maxval.unwrap_or_else(|| T::max_value());
            let normalized = v.to_f32().unwrap() / mv.to_f32().unwrap();
            let gamma_corrected = normalized.powf(gamma);
            *v = T::from(gamma_corrected * mv.to_f32().unwrap()).unwrap();
        });
    }

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

    /// Applies gamma correction to the image, returning a new image.
    ///
    /// This function does not modify the original image.
    ///
    /// # Arguments
    ///
    /// * `gamma` - The gamma value to use for correction.
    /// * `maxval` - An optional maximum value to use for normalization.
    ///   If not passed in, maximum value for the data type is used
    ///
    /// # Returns
    ///
    /// A new image with the gamma correction applied.
    ///
    /// $p_{new} = \frac{p}{p_{max}}^{\gamma} \cdot p_{max}
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    ///
    /// let img = Image::<i32>::ones(100, 100);
    /// let new_img = img.gamma_correction(2.2, None);
    /// ```
    pub fn gamma_correction(&self, gamma: f32, maxval: Option<T>) -> Image<T>
    where
        T: PixelType + num_traits::PrimInt,
    {
        let mut new_image = self.clone();
        new_image.gamma_correction_in_place(gamma, maxval);
        new_image
    }

    /// Clamps the pixel values of the image to the specified range.
    ///
    /// This function does not modify the original image.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::prelude::*;
    ///
    /// let img = Image::<i32>::ones(100, 100);
    /// let new_img = img.clamp(0, 255);
    /// ```
    pub fn clamp(&self, min: T, max: T) -> Image<T>
    where
        T: PixelType + num_traits::PrimInt,
    {
        let mut new_image = self.clone();
        new_image.clamp_in_place(min, max);
        new_image
    }

}