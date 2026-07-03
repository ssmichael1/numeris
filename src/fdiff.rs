//! Shared forward-difference Jacobian kernel.
//!
//! The `optim`, `estimate`, and `ode` modules each need a forward-difference
//! Jacobian with the same step-size policy `h_j = √ε · max(|x_j|, 1)`. This
//! module is the single source of truth for that policy so the three cannot
//! drift apart. Compiled only when one of those features is enabled.

use crate::matrix::vector::Vector;
use crate::traits::FloatScalar;
use crate::Matrix;

/// Forward-difference step for component value `x_j`: `h_j = √ε · max(|x_j|, 1)`.
///
/// The single definition of the crate's finite-difference step-size policy,
/// shared by every forward-difference routine (fixed and dynamic, sequential and
/// parallel) so the policy cannot drift between them. `√ε` is loop-invariant, so
/// callers can still call this per component without a per-iteration `sqrt`
/// surviving optimization.
#[inline]
pub(crate) fn fd_step<T: FloatScalar>(xj: T) -> T {
    T::epsilon().sqrt() * xj.abs().max(T::one())
}

/// Forward-difference Jacobian of `f: Rᴺ → Rᴹ` given a precomputed base value
/// `f0 = f(x)`.
///
/// Column `j` is `(f(x + h_j·e_j) − f0) / h_j` with `h_j = √ε · max(|x_j|, 1)`,
/// costing `N` evaluations of `eval`. The caller passes `f0` (rather than this
/// kernel recomputing it) so hot loops that already hold `f(x)` — such as the
/// Rosenbrock step loop — pay no extra evaluation.
pub(crate) fn forward_diff_jacobian<T: FloatScalar, const M: usize, const N: usize>(
    x: &Vector<T, N>,
    f0: &Vector<T, M>,
    mut eval: impl FnMut(&Vector<T, N>) -> Vector<T, M>,
) -> Matrix<T, M, N> {
    let mut jac = Matrix::<T, M, N>::zeros();

    for j in 0..N {
        let h = fd_step(x[j]);
        let mut x_pert = *x;
        x_pert[j] = x_pert[j] + h;
        let f_pert = eval(&x_pert);

        for i in 0..M {
            jac[(i, j)] = (f_pert[i] - f0[i]) / h;
        }
    }

    jac
}
