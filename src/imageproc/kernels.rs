use alloc::vec;
use alloc::vec::Vec;

use crate::matrix::Matrix;
use crate::traits::FloatScalar;

use super::ImageError;

/// Generate a 1D Gaussian kernel with standard deviation `sigma`, truncated at
/// `truncate * sigma` on each side.
///
/// The returned kernel has length `2 * ceil(truncate * sigma) + 1` (odd), is
/// symmetric, and sums to 1 (normalized so the DC gain is unity). A typical
/// choice is `truncate = 3.0` or `4.0`; smaller values give a shorter kernel
/// at the cost of a larger truncation error.
///
/// # Errors
///
/// Returns [`ImageError::InvalidParameter`] if `sigma <= 0` or not finite, or
/// if `truncate <= 0`.
///
/// # Example
///
/// ```
/// use numeris::imageproc::gaussian_kernel_1d;
///
/// let k = gaussian_kernel_1d::<f64>(1.0, 3.0).unwrap();
/// assert_eq!(k.len(), 7); // 2*ceil(3*1)+1
/// let sum: f64 = k.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-12);
/// ```
pub fn gaussian_kernel_1d<T: FloatScalar>(sigma: T, truncate: T) -> Result<Vec<T>, ImageError> {
    if !sigma.is_finite() || sigma <= T::zero() {
        return Err(ImageError::InvalidParameter);
    }
    if !truncate.is_finite() || truncate <= T::zero() {
        return Err(ImageError::InvalidParameter);
    }

    let radius_f = (truncate * sigma).ceil();
    let radius: usize = radius_f.to_usize().ok_or(ImageError::InvalidParameter)?;
    let len = 2 * radius + 1;

    let two = T::one() + T::one();
    let inv_two_sigma_sq = T::one() / (two * sigma * sigma);

    let mut kernel = Vec::with_capacity(len);
    let mut sum = T::zero();
    for i in 0..len {
        let x = T::from(i as isize - radius as isize).unwrap();
        let v = (-(x * x) * inv_two_sigma_sq).exp();
        kernel.push(v);
        sum = sum + v;
    }
    // Normalize so the DC gain is exactly 1.
    let inv_sum = T::one() / sum;
    for v in kernel.iter_mut() {
        *v = *v * inv_sum;
    }
    Ok(kernel)
}

/// Generate a 1D box (uniform) kernel of length `n`.
///
/// Each element is `1/n`, so applying the kernel produces a moving average.
/// `n` must be odd and nonzero.
///
/// # Errors
///
/// Returns [`ImageError::InvalidKernelSize`] if `n == 0` or `n` is even.
pub fn box_kernel_1d<T: FloatScalar>(n: usize) -> Result<Vec<T>, ImageError> {
    if n == 0 || n % 2 == 0 {
        return Err(ImageError::InvalidKernelSize);
    }
    let v = T::one() / T::from(n).unwrap();
    Ok(vec![v; n])
}

/// 3×3 Sobel kernel for the horizontal gradient (`∂/∂x`), pre-oriented so that
/// [`convolve2d`](super::convolve2d) (correlation) produces a positive
/// response on transitions from dark to bright from left to right:
///
/// ```text
/// [-1  0  1]
/// [-2  0  2]
/// [-1  0  1]
/// ```
pub fn sobel_x_3x3<T: FloatScalar>() -> Matrix<T, 3, 3> {
    let m1 = -T::one();
    let z = T::zero();
    let p1 = T::one();
    let m2 = m1 + m1;
    let p2 = p1 + p1;
    Matrix::new([[m1, z, p1], [m2, z, p2], [m1, z, p1]])
}

/// 3×3 Sobel kernel for the vertical gradient (`∂/∂y`), pre-oriented so that
/// [`convolve2d`](super::convolve2d) (correlation) produces a positive
/// response on transitions from dark to bright from top to bottom:
///
/// ```text
/// [-1 -2 -1]
/// [ 0  0  0]
/// [ 1  2  1]
/// ```
pub fn sobel_y_3x3<T: FloatScalar>() -> Matrix<T, 3, 3> {
    let m1 = -T::one();
    let z = T::zero();
    let p1 = T::one();
    let m2 = m1 + m1;
    let p2 = p1 + p1;
    Matrix::new([[m1, m2, m1], [z, z, z], [p1, p2, p1]])
}

/// 3×3 Scharr kernel for the horizontal gradient (`∂/∂x`). Scharr weights
/// give better rotational symmetry than Sobel at the cost of a larger peak
/// response (normalizer = 1/32 for a balanced magnitude).
///
/// ```text
/// [ -3  0   3]
/// [-10  0  10]
/// [ -3  0   3]
/// ```
pub fn scharr_x_3x3<T: FloatScalar>() -> Matrix<T, 3, 3> {
    let three = T::from(3.0_f64).unwrap();
    let ten = T::from(10.0_f64).unwrap();
    let z = T::zero();
    Matrix::new([[-three, z, three], [-ten, z, ten], [-three, z, three]])
}

/// 3×3 Scharr kernel for the vertical gradient (`∂/∂y`).
///
/// ```text
/// [-3 -10 -3]
/// [ 0   0  0]
/// [ 3  10  3]
/// ```
pub fn scharr_y_3x3<T: FloatScalar>() -> Matrix<T, 3, 3> {
    let three = T::from(3.0_f64).unwrap();
    let ten = T::from(10.0_f64).unwrap();
    let z = T::zero();
    Matrix::new([[-three, -ten, -three], [z, z, z], [three, ten, three]])
}

/// 3×3 Laplacian kernel (4-neighbour variant, second-order isotropic
/// discrete difference operator).
///
/// ```text
/// [ 0  1  0]
/// [ 1 -4  1]
/// [ 0  1  0]
/// ```
pub fn laplacian_3x3<T: FloatScalar>() -> Matrix<T, 3, 3> {
    let z = T::zero();
    let one = T::one();
    let m4 = -(one + one + one + one);
    Matrix::new([[z, one, z], [one, m4, one], [z, one, z]])
}

/// 3×3 Laplacian kernel (8-neighbour variant; includes diagonal neighbours).
///
/// ```text
/// [ 1  1  1]
/// [ 1 -8  1]
/// [ 1  1  1]
/// ```
pub fn laplacian_3x3_diag<T: FloatScalar>() -> Matrix<T, 3, 3> {
    let one = T::one();
    let two = one + one;
    let four = two + two;
    let m8 = -(four + four);
    Matrix::new([[one, one, one], [one, m8, one], [one, one, one]])
}
