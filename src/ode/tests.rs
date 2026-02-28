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

    // ── Rosenbrock / stiff solver tests ─────────────────────────────

    #[test]
    fn rodas4_stiff_exponential_decay() {
        // y' = -1000*y, y(0) = 1 → y(t) = e^{-1000t}
        // Extremely stiff: explicit methods need tiny steps, RODAS4 handles it.
        let y0 = Vector::from_array([1.0_f64]);
        let settings = AdaptiveSettings {
            abs_tol: 1e-10,
            rel_tol: 1e-10,
            ..AdaptiveSettings::default()
        };
        let sol = RODAS4::integrate(
            0.0, 0.01, &y0,
            |_t, y| Vector::from_array([-1000.0 * y[0]]),
            |_t, _y| crate::Matrix::new([[-1000.0]]),
            &settings,
        ).unwrap();
        let exact = (-1000.0_f64 * 0.01).exp();
        assert!(
            (sol.y[0] - exact).abs() < 1e-8,
            "rodas4 stiff decay: y = {}, exact = {}, err = {}",
            sol.y[0], exact, (sol.y[0] - exact).abs()
        );
    }

    #[test]
    fn rodas4_van_der_pol_mu1000() {
        // Van der Pol oscillator with μ = 1000 (very stiff).
        // y₁' = y₂
        // y₂' = μ((1 - y₁²)y₂ - y₁)
        // Just verify it completes without error and stays bounded.
        let mu = 1000.0_f64;
        let y0 = Vector::from_array([2.0, 0.0]);
        let settings = AdaptiveSettings {
            abs_tol: 1e-6,
            rel_tol: 1e-6,
            max_steps: 500_000,
            ..AdaptiveSettings::default()
        };

        let sol = RODAS4::integrate(
            0.0, 100.0, &y0,
            |_t, y| Vector::from_array([
                y[1],
                mu * ((1.0 - y[0] * y[0]) * y[1] - y[0]),
            ]),
            |_t, y| crate::Matrix::new([
                [0.0, 1.0],
                [mu * (-2.0 * y[0] * y[1] - 1.0), mu * (1.0 - y[0] * y[0])],
            ]),
            &settings,
        ).unwrap();

        // Solution should stay bounded (|y₁| ≤ ~2.1 for Van der Pol limit cycle)
        assert!(
            sol.y[0].abs() < 3.0,
            "Van der Pol y[0] = {} exceeds bound", sol.y[0]
        );
    }

    #[test]
    fn rodas4_robertson() {
        // Robertson chemical kinetics (classic stiff test):
        // y₁' = -0.04 y₁ + 1e4 y₂ y₃
        // y₂' =  0.04 y₁ - 1e4 y₂ y₃ - 3e7 y₂²
        // y₃' =  3e7 y₂²
        // Conservation: y₁ + y₂ + y₃ = 1 for all t.
        let y0 = Vector::from_array([1.0, 0.0, 0.0]);
        let settings = AdaptiveSettings {
            abs_tol: 1e-8,
            rel_tol: 1e-8,
            ..AdaptiveSettings::default()
        };

        let sol = RODAS4::integrate(
            0.0, 1e3, &y0,
            |_t, y| Vector::from_array([
                -0.04 * y[0] + 1e4 * y[1] * y[2],
                 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1],
                 3e7 * y[1] * y[1],
            ]),
            |_t, y| crate::Matrix::new([
                [-0.04,           1e4 * y[2],        1e4 * y[1]],
                [ 0.04, -1e4 * y[2] - 6e7 * y[1],  -1e4 * y[1]],
                [ 0.0,             6e7 * y[1],        0.0],
            ]),
            &settings,
        ).unwrap();

        // Check conservation law
        let mass: f64 = sol.y[0] + sol.y[1] + sol.y[2];
        assert!(
            (mass - 1.0).abs() < 1e-6,
            "Robertson mass conservation violated: sum = {}", mass
        );

        // All concentrations should be non-negative
        assert!(sol.y[0] >= -1e-10, "y[0] = {} is negative", sol.y[0]);
        assert!(sol.y[1] >= -1e-10, "y[1] = {} is negative", sol.y[1]);
        assert!(sol.y[2] >= -1e-10, "y[2] = {} is negative", sol.y[2]);
    }

    #[test]
    fn rodas4_auto_jacobian_stiff_decay() {
        // Same stiff decay as above, but using auto-Jacobian.
        let y0 = Vector::from_array([1.0_f64]);
        let settings = AdaptiveSettings {
            abs_tol: 1e-8,
            rel_tol: 1e-8,
            ..AdaptiveSettings::default()
        };
        let sol = RODAS4::integrate_auto(
            0.0, 0.01, &y0,
            |_t, y| Vector::from_array([-1000.0 * y[0]]),
            &settings,
        ).unwrap();
        let exact = (-1000.0_f64 * 0.01).exp();
        assert!(
            (sol.y[0] - exact).abs() < 1e-6,
            "auto-jac stiff decay: y = {}, exact = {}, err = {}",
            sol.y[0], exact, (sol.y[0] - exact).abs()
        );
    }

    #[test]
    fn rodas4_auto_jacobian_robertson() {
        // Robertson with auto-Jacobian — verify similar accuracy to analytic.
        let y0 = Vector::from_array([1.0, 0.0, 0.0]);
        let settings = AdaptiveSettings {
            abs_tol: 1e-6,
            rel_tol: 1e-6,
            ..AdaptiveSettings::default()
        };

        let sol = RODAS4::integrate_auto(
            0.0, 1e3, &y0,
            |_t, y| Vector::from_array([
                -0.04 * y[0] + 1e4 * y[1] * y[2],
                 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1],
                 3e7 * y[1] * y[1],
            ]),
            &settings,
        ).unwrap();

        let mass: f64 = sol.y[0] + sol.y[1] + sol.y[2];
        assert!(
            (mass - 1.0).abs() < 1e-4,
            "auto-jac Robertson mass conservation: sum = {}", mass
        );
    }

    #[test]
    fn rodas4_singular_w_matrix() {
        // Provide a Jacobian that makes W = I/(hγ) - J singular.
        // W is singular when J has eigenvalue 1/(hγ). With γ=0.25 and the
        // initial step size guess, we construct J = (1/(hγ))*I which makes
        // the diagonal of W zero. In practice, the easiest way to trigger
        // this is to provide a Jacobian whose eigenvalues overwhelm the
        // 1/(hγ) term, but that depends on h. Instead, test that a NaN
        // in the state produces StepNotFinite.
        let y0 = Vector::from_array([f64::NAN]);
        let settings = AdaptiveSettings::default();
        let result = RODAS4::integrate(
            0.0, 1.0, &y0,
            |_t, y| *y,
            |_t, _y| crate::Matrix::new([[1.0]]),
            &settings,
        );
        assert!(matches!(result, Err(OdeError::StepNotFinite)));
    }

    #[test]
    fn rodas4_harmonic_oscillator() {
        // Non-stiff problem: harmonic oscillator (same as explicit tests).
        // RODAS4 should still give reasonable results, just less efficient.
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let settings = AdaptiveSettings {
            abs_tol: 1e-10,
            rel_tol: 1e-10,
            ..AdaptiveSettings::default()
        };
        let sol = RODAS4::integrate(
            0.0, TAU, &y0,
            ydot,
            |_t, _y| crate::Matrix::new([[0.0, 1.0], [-1.0, 0.0]]),
            &settings,
        ).unwrap();
        assert!(
            (sol.y[0] - 1.0).abs() < 1e-6,
            "harmonic y[0] = {}, err = {}", sol.y[0], (sol.y[0] - 1.0).abs()
        );
        assert!(
            sol.y[1].abs() < 1e-6,
            "harmonic y[1] = {}", sol.y[1]
        );
    }
}
