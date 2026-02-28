use crate::matrix::vector::Vector;
use crate::traits::FloatScalar;

use super::OptimError;

/// Backtracking line search satisfying the Armijo (sufficient decrease) condition.
///
/// Starts with `alpha = 1` and contracts by factor `rho` until:
/// `f(x + α·p) ≤ f(x) + c1·α·(∇f · p)`
///
/// Returns `(alpha, f_new, evals)`.
pub(crate) fn backtracking_armijo<T: FloatScalar, const N: usize>(
    f_at_x: T,
    grad_dot_p: T,
    x: &Vector<T, N>,
    p: &Vector<T, N>,
    f: &mut impl FnMut(&Vector<T, N>) -> T,
    c1: T,
    rho: T,
    max_iter: usize,
) -> Result<(T, T, usize), OptimError> {
    let mut alpha = T::one();
    let mut evals = 0;

    for _ in 0..max_iter {
        let x_new = *x + *p * alpha;
        let f_new = f(&x_new);
        evals += 1;

        if f_new <= f_at_x + c1 * alpha * grad_dot_p {
            return Ok((alpha, f_new, evals));
        }

        alpha = alpha * rho;
    }

    Err(OptimError::LineSearchFailed)
}
