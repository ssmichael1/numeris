use super::{Image, PixelType};


 /// A boxcar filter convolves image with a box (uniform) 2D filter of ksize dimensions on each side
/// 
/// # Notes
///
/// * This convolution operation is also known as a moving average filter
/// * The convlution is done in-place
///
/// # Example
/// ```
/// use numeris::prelude::*;
/// let mut img = Image::<f32>::ones(100, 100);
/// boxcar(&mut img, 5);
/// ```
/// 
pub fn boxcar<T>(image: &mut Image<T>, ksize: usize)
where
    T: PixelType
        + Copy
        + Default
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::DivAssign
        + std::iter::Sum
        + num_traits::Zero
        + num_traits::FromPrimitive,
{
    let (width, height) = (image.width(), image.height());
    let mut temp = Image::<T>::zeros(width, height);
    let half = (ksize / 2) as isize;
    let ksize_t = num_traits::cast::FromPrimitive::from_usize(ksize).expect("ksize must be convertible to T");

    // Horizontal pass: moving sum for each row
    for y in 0..height {
        let mut sum = T::zero();
        // Initialize sum for the first window
        for dx in 0..ksize.min(width) {
            sum += image[(dx, y)];
        }
        for x in 0..width {
            let mut avg = sum;
            avg /= ksize_t;
            temp[(x, y)] = avg;
            // Slide window: remove left, add right
            let left = x as isize - half;
            let right = x as isize + half + 1;
            if left >= 0 && right < width as isize {
                sum -= image[(left as usize, y)];
                sum += image[(right as usize, y)];
            } else if left >= 0 {
                sum -= image[(left as usize, y)];
            } else if right < width as isize {
                sum += image[(right as usize, y)];
            }
        }
    }

    // Vertical pass: moving sum for each column
    for x in 0..width {
        let mut sum = T::zero();
        for dy in 0..ksize.min(height) {
            sum += temp[(x, dy)];
        }
        for y in 0..height {
            let mut avg = sum;
            avg /= ksize_t;
            image[(x, y)] = avg;
            let top = y as isize - half;
            let bottom = y as isize + half + 1;
            if top >= 0 && bottom < height as isize {
                sum -= temp[(x, top as usize)];
                sum += temp[(x, bottom as usize)];
            } else if top >= 0 {
                sum -= temp[(x, top as usize)];
            } else if bottom < height as isize {
                sum += temp[(x, bottom as usize)];
            }
        }
    }
}


pub fn gaussian_blur<const KSIZE: usize>(image: &mut Image<f32>, sigma: f32)
{
    // Generate 1D Gaussian kernel
    let half = (KSIZE / 2) as isize;
    let sigma2 = 2.0 * sigma * sigma;
    let mut kernel = [0.0f32; KSIZE];
    let mut sum = 0.0f32;
    for (i, k) in kernel.iter_mut().enumerate() {
        let x = i as isize - half;
        let val = (-((x * x) as f32) / sigma2).exp();
        *k = val;
        sum += val;
    }
    for k in &mut kernel {
        *k /= sum;
    }

    let (width, height) = (image.width(), image.height());
    let mut temp = Image::<f32>::zeros(width, height);

    // Horizontal pass
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f32;
            for (k, &weight) in kernel.iter().enumerate() {
                let dx = k as isize - half;
                let xx = x as isize + dx;
                if xx >= 0 && xx < width as isize {
                    acc += image[(xx as usize, y)] * weight;
                }
            }
            temp[(x, y)] = acc;
        }
    }

    // Vertical pass (write back to image)
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f32;
            for (k, &weight) in kernel.iter().enumerate() {
                let dy = k as isize - half;
                let yy = y as isize + dy;
                if yy >= 0 && yy < height as isize {
                    acc += temp[(x, yy as usize)] * weight;
                }
            }
            image[(x, y)] = acc;
        }
    }
}
