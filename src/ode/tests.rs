use super::*;
use crate::Vector;

const PI: f64 = core::f64::consts::PI;
const TAU: f64 = 2.0 * PI;

fn ydot(_t: f64, y: &Vector<f64, 2>) -> Vector<f64, 2> {
    Vector::from_array([y[1], -y[0]])
}

// ── Fixed-step RK4 (works in no-std) ───────────────────────────────

#[test]
fn rk4_step_exponential_decay() {
    let y = Vector::from_array([1.0_f64]);
    let y1 = rk4_step(0.0, &y, 0.01, |_t, y| *y * (-1.0));
    assert!((y1[0] - (-0.01_f64).exp()).abs() < 1e-10);
}

#[test]
fn rk4_harmonic_oscillator() {
    let y0 = Vector::from_array([1.0_f64, 0.0]);
    let yf = rk4(0.0, TAU, 0.001, &y0, ydot);
    assert!((yf[0] - 1.0).abs() < 1e-8);
    assert!(yf[1].abs() < 1e-8);
}

#[test]
fn rk4_backward() {
    let y0 = Vector::from_array([1.0_f64, 0.0]);
    let yf = rk4(0.0, -TAU, 0.001, &y0, ydot);
    assert!((yf[0] - 1.0).abs() < 1e-8);
    assert!(yf[1].abs() < 1e-8);
}

// ── Adaptive + dense output tests (require std for Vec) ─────────────

#[cfg(feature = "std")]
mod adaptive_tests {
    use super::*;

    fn tight_settings() -> AdaptiveSettings<f64> {
        AdaptiveSettings {
            abs_tol: 1e-12,
            rel_tol: 1e-12,
            ..AdaptiveSettings::default()
        }
    }

    fn test_harmonic<const N: usize, const NI: usize, S: RKAdaptive<N, NI>>() {
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let settings = tight_settings();
        let sol = S::integrate(0.0, TAU, &y0, ydot, &settings).unwrap();
        assert!((sol.y[0] - 1.0).abs() < 1e-10);
        assert!(sol.y[1].abs() < 1e-10);
    }

    #[test]
    fn harmonic_rkf45() {
        test_harmonic::<6, 1, RKF45>();
    }

    #[test]
    fn harmonic_rkts54() {
        test_harmonic::<7, 4, RKTS54>();
    }

    #[test]
    fn harmonic_rkv65() {
        test_harmonic::<10, 6, RKV65>();
    }

    #[test]
    fn harmonic_rkv87() {
        test_harmonic::<17, 7, RKV87>();
    }

    #[test]
    fn harmonic_rkv98() {
        test_harmonic::<21, 8, RKV98>();
    }

    #[test]
    fn harmonic_rkv98_nointerp() {
        test_harmonic::<16, 1, RKV98NoInterp>();
    }

    #[test]
    fn harmonic_rkv98_efficient() {
        test_harmonic::<26, 9, RKV98Efficient>();
    }

    #[test]
    fn backward_rkts54() {
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let settings = tight_settings();
        let sol = RKTS54::integrate(0.0, -TAU, &y0, ydot, &settings).unwrap();
        assert!((sol.y[0] - 1.0).abs() < 1e-10);
        assert!(sol.y[1].abs() < 1e-10);
    }

    #[test]
    fn backward_rkv87() {
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let settings = tight_settings();
        let sol = RKV87::integrate(0.0, -TAU, &y0, ydot, &settings).unwrap();
        assert!((sol.y[0] - 1.0).abs() < 1e-10);
        assert!(sol.y[1].abs() < 1e-10);
    }

    #[test]
    fn fsal_saves_evaluations() {
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let settings = tight_settings();

        let sol_fsal = RKTS54::integrate(0.0, TAU, &y0, ydot, &settings).unwrap();
        let sol_nonfsal = RKF45::integrate(0.0, TAU, &y0, ydot, &settings).unwrap();

        // RKTS54 (FSAL, 7 stages) should use fewer evals than 7*accepted
        assert!(sol_fsal.evals < sol_fsal.accepted * 7);

        assert!((sol_fsal.y[0] - 1.0).abs() < 1e-10);
        assert!((sol_nonfsal.y[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rkv98_variants_agree() {
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let settings = tight_settings();

        let sol_robust = RKV98::integrate(0.0, TAU, &y0, ydot, &settings).unwrap();
        let sol_nointerp = RKV98NoInterp::integrate(0.0, TAU, &y0, ydot, &settings).unwrap();
        let sol_efficient = RKV98Efficient::integrate(0.0, TAU, &y0, ydot, &settings).unwrap();

        assert!((sol_robust.y[0] - 1.0).abs() < 1e-11);
        assert!((sol_nointerp.y[0] - 1.0).abs() < 1e-11);
        assert!((sol_efficient.y[0] - 1.0).abs() < 1e-11);
    }

    #[test]
    fn tighter_tolerance_improves_accuracy() {
        let y0 = Vector::from_array([1.0_f64, 0.0]);

        let tols = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12];
        let mut prev_err = f64::MAX;

        for &tol in &tols {
            let settings = AdaptiveSettings {
                abs_tol: tol,
                rel_tol: tol,
                ..AdaptiveSettings::default()
            };
            let sol = RKTS54::integrate(0.0, TAU, &y0, ydot, &settings).unwrap();
            let err = (sol.y[0] - 1.0).abs() + sol.y[1].abs();
            assert!(err < prev_err, "tol={tol}: error {err} not smaller than previous {prev_err}");
            prev_err = err;
        }
    }

    #[test]
    fn max_steps_exceeded() {
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let settings = AdaptiveSettings {
            max_steps: 5,
            ..tight_settings()
        };
        let result = RKTS54::integrate(0.0, TAU, &y0, ydot, &settings);
        assert!(matches!(result, Err(OdeError::MaxStepsExceeded)));
    }

    #[test]
    fn harmonic_f32() {
        let y0 = Vector::from_array([1.0_f32, 0.0]);
        let settings = AdaptiveSettings::<f32> {
            abs_tol: 1e-6,
            rel_tol: 1e-6,
            ..AdaptiveSettings::default()
        };
        let sol = RKTS54::integrate(
            0.0_f32,
            core::f32::consts::TAU,
            &y0,
            |_t, y| Vector::from_array([y[1], -y[0]]),
            &settings,
        )
        .unwrap();
        assert!((sol.y[0] - 1.0).abs() < 1e-4);
        assert!(sol.y[1].abs() < 1e-4);
    }

    // ── Dense output / interpolation ────────────────────────────────

    fn test_interp<const N: usize, const NI: usize, S: RKAdaptive<N, NI>>() {
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let settings = AdaptiveSettings {
            abs_tol: 1e-12,
            rel_tol: 1e-12,
            dense_output: true,
            ..AdaptiveSettings::default()
        };
        let sol = S::integrate(0.0, PI, &y0, ydot, &settings).unwrap();

        for i in 0..=100 {
            let t = PI * (i as f64) / 100.0;
            let y_interp = S::interpolate(t, &sol).unwrap();
            assert!((y_interp[0] - t.cos()).abs() < 1e-9);
            assert!((y_interp[1] - (-t.sin())).abs() < 1e-9);
        }
    }

    #[test]
    fn interp_rkts54() {
        test_interp::<7, 4, RKTS54>();
    }

    #[test]
    fn interp_rkv65() {
        test_interp::<10, 6, RKV65>();
    }

    #[test]
    fn interp_rkv87() {
        test_interp::<17, 7, RKV87>();
    }

    #[test]
    fn interp_rkv98() {
        test_interp::<21, 8, RKV98>();
    }

    #[test]
    fn interp_rkv98_efficient() {
        test_interp::<26, 9, RKV98Efficient>();
    }

    #[test]
    fn interp_nointerp_returns_error() {
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let settings = AdaptiveSettings {
            dense_output: true,
            ..tight_settings()
        };
        let sol = RKV98NoInterp::integrate(0.0, PI, &y0, ydot, &settings).unwrap();
        assert_eq!(RKV98NoInterp::interpolate(0.5, &sol).unwrap_err(), OdeError::InterpNotImplemented);
    }

    #[test]
    fn interp_out_of_bounds() {
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let settings = AdaptiveSettings {
            dense_output: true,
            ..tight_settings()
        };
        let sol = RKTS54::integrate(0.0, PI, &y0, ydot, &settings).unwrap();
        assert_eq!(RKTS54::interpolate(PI + 1.0, &sol).unwrap_err(), OdeError::InterpOutOfBounds);
    }

    #[test]
    fn interp_no_dense() {
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let settings = tight_settings();
        let sol = RKTS54::integrate(0.0, PI, &y0, ydot, &settings).unwrap();
        assert_eq!(RKTS54::interpolate(0.5, &sol).unwrap_err(), OdeError::NoDenseOutput);
    }

    #[test]
    fn interp_backward() {
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let settings = AdaptiveSettings {
            dense_output: true,
            ..tight_settings()
        };
        let sol = RKTS54::integrate(0.0, -PI, &y0, ydot, &settings).unwrap();

        for i in 0..=10 {
            let t = -PI * (i as f64) / 10.0;
            let y_interp = RKTS54::interpolate(t, &sol).unwrap();
            assert!((y_interp[0] - t.cos()).abs() < 1e-9);
        }
    }
}
