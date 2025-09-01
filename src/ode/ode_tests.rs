//! Test for ODE solvers
//!
//! Tests are for a simple 1D harmonic oscillator
//! which can be easily checked for correctness against the analytical solution.
//!
//! The tests are run for all adaptive solvers with and without interpolation.
//!

use super::ODEResult;
use super::RKAdaptive;
use super::RKAdaptiveSettings;

/// ODE state for 1-dimensional harmonic oscillator
/// y[0] = position
/// y[1] = velocity
type State = crate::matrix::Vector2d;

/// Harmonic oscillator ODE y'' = -y
fn ydot(_t: f64, y: &State) -> ODEResult<State> {
    Ok([y[1], -y[0]].into())
}

/// Test harmonic oscillator
fn harmonic_oscillator<const N: usize, const NI: usize, F>(_integrator: F)
where
    F: RKAdaptive<N, NI>,
{
    use std::f64::consts::PI;

    let y0 = [1.0, 0.0].into();

    let settings = RKAdaptiveSettings {
        dense_output: false,
        abserror: 1e-12,
        relerror: 1e-12,
        ..RKAdaptiveSettings::default()
    };

    let res = F::integrate(0.0, 2.0 * PI, &y0, ydot, &settings).unwrap();
    assert!((res.y[0] - 1.0).abs() < 1e-11);
    assert!((res.y[1]).abs() < 1e-11);
}

/// Test harmonic oscillator with interpolation
fn harmonic_oscillator_interp<const N: usize, const NI: usize, F>(_integrator: F)
where
    F: RKAdaptive<N, NI>,
{
    use std::f64::consts::PI;

    let y0 = [1.0, 0.0].into();

    let settings = RKAdaptiveSettings {
        dense_output: true,
        abserror: 1e-12,
        relerror: 1e-12,
        ..RKAdaptiveSettings::default()
    };

    let res = F::integrate(0.0, PI, &y0, ydot, &settings).unwrap();

    let testcount = 100;
    (0..testcount).for_each(|idx| {
        let x = idx as f64 * PI / testcount as f64;
        let interp = F::interpolate(x, &res).unwrap();
        assert!((interp[0] - x.cos()).abs() < 1e-10);
        assert!((interp[1] + x.sin()).abs() < 1e-10);
    });
}

/// Test harmonic oscillator with all integrators
#[test]
fn test_harmonic_oscillator() {
    harmonic_oscillator(super::solvers::RKF45 {});
    harmonic_oscillator(super::solvers::RKTS54 {});
    harmonic_oscillator(super::solvers::RKV65 {});
    harmonic_oscillator(super::solvers::RKV87 {});
    harmonic_oscillator(super::solvers::RKV98 {});
    harmonic_oscillator(super::solvers::RKV98NoInterp {});
}

/// Test harmonic oscillator with all integrators with interpolation
#[test]
fn test_harmonic_oscillator_interp() {
    harmonic_oscillator_interp(super::solvers::RKTS54 {});
    harmonic_oscillator_interp(super::solvers::RKV65 {});
    harmonic_oscillator_interp(super::solvers::RKV87 {});
    harmonic_oscillator_interp(super::solvers::RKV98 {});
}

#[test]
fn simple_standalone() {
    use crate::matrix::Vector2d;
    use crate::ode::solvers::RKF45;
    use std::f64::consts::PI;

    let settings = RKAdaptiveSettings::default();
    let y0: Vector2d = [0.0, -1.0].into();
    let result = RKF45::integrate(
        0.0,
        2.0 * PI,
        &y0,
        // Harmonic oscillator ODE
        |_t, y: &Vector2d| Ok([y[1], -y[0]].into()),
        &settings,
    );
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!((result.y[1] + 1.0).abs() < 1e-6);
    assert!((result.y[0] - 0.0).abs() < 1e-8);
}
