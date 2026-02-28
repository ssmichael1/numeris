use crate::traits::FloatScalar;
use crate::matrix::vector::Vector;

/// Single step of the classic 4th-order Runge-Kutta method.
///
/// Advances `y` from `t` to `t + h` using `f(t, y) -> dy/dt`.
///
/// ```
/// use numeris::ode::rk4_step;
/// use numeris::Vector;
///
/// // dy/dt = -y (exponential decay)
/// let y = Vector::from_array([1.0_f64]);
/// let y1 = rk4_step(0.0, &y, 0.01, |_t, y| y * (-1.0));
/// assert!((y1[0] - (-0.01_f64).exp()).abs() < 1e-10);
/// ```
pub fn rk4_step<T: FloatScalar, const S: usize>(
    t: T,
    y: &Vector<T, S>,
    h: T,
    mut f: impl FnMut(T, &Vector<T, S>) -> Vector<T, S>,
) -> Vector<T, S> {
    let half = T::from(0.5).unwrap();
    let sixth = T::from(1.0 / 6.0).unwrap();
    let third = T::from(1.0 / 3.0).unwrap();

    let k1 = f(t, y);
    let k2 = f(t + h * half, &(*y + k1 * h * half));
    let k3 = f(t + h * half, &(*y + k2 * h * half));
    let k4 = f(t + h, &(*y + k3 * h));

    *y + (k1 * sixth + k2 * third + k3 * third + k4 * sixth) * h
}

/// Integrate an ODE using fixed-step 4th-order Runge-Kutta.
///
/// Returns the final state at `tf`. The step size `dt` is used directly
/// (positive for forward, negative for backward).
///
/// ```
/// use numeris::ode::rk4;
/// use numeris::Vector;
///
/// // Harmonic oscillator: y'' = -y  →  [y, y']
/// let y0 = Vector::from_array([1.0_f64, 0.0]);
/// let yf = rk4(0.0, std::f64::consts::TAU, 0.001, &y0,
///     |_t, y| Vector::from_array([y[1], -y[0]]),
/// );
/// assert!((yf[0] - 1.0).abs() < 1e-8);
/// assert!((yf[1]).abs() < 1e-8);
/// ```
pub fn rk4<T: FloatScalar, const S: usize>(
    t0: T,
    tf: T,
    dt: T,
    y0: &Vector<T, S>,
    mut f: impl FnMut(T, &Vector<T, S>) -> Vector<T, S>,
) -> Vector<T, S> {
    let mut t = t0;
    let mut y = *y0;
    let tdir = if tf > t0 { T::one() } else { -T::one() };
    let mut h = dt.abs() * tdir;

    loop {
        // Clamp last step
        if (tdir > T::zero() && t + h > tf) || (tdir < T::zero() && t + h < tf) {
            h = tf - t;
        }

        y = rk4_step(t, &y, h, &mut f);
        t = t + h;

        if (tdir > T::zero() && t >= tf) || (tdir < T::zero() && t <= tf) {
            break;
        }
    }

    y
}
