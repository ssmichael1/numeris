use crate::matrix::vector::Vector;
use crate::traits::FloatScalar;
use crate::Matrix;

/// Approximate the Jacobian of `f: R^N → R^M` using forward finite differences.
///
/// Uses step size `h_j = sqrt(ε) * max(|x_j|, 1)` for each component,
/// requiring `N + 1` function evaluations (one base evaluation + N perturbed).
///
/// # Example
///
/// ```
/// use numeris::optim::finite_difference_jacobian;
/// use numeris::Vector;
///
/// // f(x) = [x0^2, x0*x1], Jacobian = [[2*x0, 0], [x1, x0]]
/// let x = Vector::from_array([3.0_f64, 4.0]);
/// let j = finite_difference_jacobian(|x: &Vector<f64, 2>| {
///     Vector::from_array([x[0] * x[0], x[0] * x[1]])
/// }, &x);
/// assert!((j[(0, 0)] - 6.0).abs() < 1e-6);
/// assert!((j[(0, 1)] - 0.0).abs() < 1e-6);
/// assert!((j[(1, 0)] - 4.0).abs() < 1e-6);
/// assert!((j[(1, 1)] - 3.0).abs() < 1e-6);
/// ```
pub fn finite_difference_jacobian<T: FloatScalar, const M: usize, const N: usize>(
    mut f: impl FnMut(&Vector<T, N>) -> Vector<T, M>,
    x: &Vector<T, N>,
) -> Matrix<T, M, N> {
    let sqrt_eps = T::epsilon().sqrt();
    let f0 = f(x);
    let mut jac = Matrix::<T, M, N>::zeros();

    for j in 0..N {
        let h = sqrt_eps * x[j].abs().max(T::one());
        let mut x_pert = *x;
        x_pert[j] = x_pert[j] + h;
        let f_pert = f(&x_pert);

        for i in 0..M {
            jac[(i, j)] = (f_pert[i] - f0[i]) / h;
        }
    }

    jac
}

/// Approximate the gradient of `f: R^N → R` using forward finite differences.
///
/// Uses step size `h_j = sqrt(ε) * max(|x_j|, 1)` for each component,
/// requiring `N + 1` function evaluations.
///
/// # Example
///
/// ```
/// use numeris::optim::finite_difference_gradient;
/// use numeris::Vector;
///
/// // f(x) = x0^2 + 2*x1^2, grad = [2*x0, 4*x1]
/// let x = Vector::from_array([3.0_f64, 4.0]);
/// let g = finite_difference_gradient(|x: &Vector<f64, 2>| {
///     x[0] * x[0] + x[1] * x[1] * 2.0
/// }, &x);
/// assert!((g[0] - 6.0).abs() < 1e-5);
/// assert!((g[1] - 16.0).abs() < 1e-5);
/// ```
pub fn finite_difference_gradient<T: FloatScalar, const N: usize>(
    mut f: impl FnMut(&Vector<T, N>) -> T,
    x: &Vector<T, N>,
) -> Vector<T, N> {
    let sqrt_eps = T::epsilon().sqrt();
    let f0 = f(x);
    let mut grad = Vector::<T, N>::zeros();

    for j in 0..N {
        let h = sqrt_eps * x[j].abs().max(T::one());
        let mut x_pert = *x;
        x_pert[j] = x_pert[j] + h;
        grad[j] = (f(&x_pert) - f0) / h;
    }

    grad
}
