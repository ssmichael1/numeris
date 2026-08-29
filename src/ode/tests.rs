use super::*;
use crate::{Matrix, Vector};

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
            assert!(
                err < prev_err,
                "tol={tol}: error {err} not smaller than previous {prev_err}"
            );
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

    fn test_interp_batch<const N: usize, const NI: usize, S: RKAdaptive<N, NI>>() {
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let settings = AdaptiveSettings {
            abs_tol: 1e-12,
            rel_tol: 1e-12,
            dense_output: true,
            ..AdaptiveSettings::default()
        };
        let sol = S::integrate(0.0, PI, &y0, ydot, &settings).unwrap();

        // Build sorted array of 101 interpolation times from 0 to PI
        let times: Vec<f64> = (0..=100).map(|i| PI * (i as f64) / 100.0).collect();

        let batch = S::interpolate_batch(&times, &sol).unwrap();
        assert_eq!(batch.len(), 101);

        for (i, y_batch) in batch.iter().enumerate() {
            let t = times[i];
            // Check against analytic solution
            assert!(
                (y_batch[0] - t.cos()).abs() < 1e-9,
                "batch cos mismatch at t={t}: {} vs {}",
                y_batch[0],
                t.cos()
            );
            assert!(
                (y_batch[1] - (-t.sin())).abs() < 1e-9,
                "batch sin mismatch at t={t}: {} vs {}",
                y_batch[1],
                -t.sin()
            );

            // Check that batch matches individual interpolate
            let y_single = S::interpolate(t, &sol).unwrap();
            assert!(
                (y_batch[0] - y_single[0]).abs() < 1e-15,
                "batch vs single mismatch at t={t}"
            );
            assert!(
                (y_batch[1] - y_single[1]).abs() < 1e-15,
                "batch vs single mismatch at t={t}"
            );
        }
    }

    #[test]
    fn interp_batch_rkts54() {
        test_interp_batch::<7, 4, RKTS54>();
    }

    #[test]
    fn interp_batch_rkv65() {
        test_interp_batch::<10, 6, RKV65>();
    }

    #[test]
    fn interp_batch_rkv87() {
        test_interp_batch::<17, 7, RKV87>();
    }

    #[test]
    fn interp_batch_rkv98() {
        test_interp_batch::<21, 8, RKV98>();
    }

    #[test]
    fn interp_batch_rkv98_efficient() {
        test_interp_batch::<26, 9, RKV98Efficient>();
    }

    #[test]
    fn interp_nointerp_returns_error() {
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let settings = AdaptiveSettings {
            dense_output: true,
            ..tight_settings()
        };
        let sol = RKV98NoInterp::integrate(0.0, PI, &y0, ydot, &settings).unwrap();
        assert_eq!(
            RKV98NoInterp::interpolate(0.5, &sol).unwrap_err(),
            OdeError::InterpNotImplemented
        );
    }

    #[test]
    fn interp_out_of_bounds() {
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let settings = AdaptiveSettings {
            dense_output: true,
            ..tight_settings()
        };
        let sol = RKTS54::integrate(0.0, PI, &y0, ydot, &settings).unwrap();
        assert_eq!(
            RKTS54::interpolate(PI + 1.0, &sol).unwrap_err(),
            OdeError::InterpOutOfBounds
        );
    }

    #[test]
    fn interp_no_dense() {
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let settings = tight_settings();
        let sol = RKTS54::integrate(0.0, PI, &y0, ydot, &settings).unwrap();
        assert_eq!(
            RKTS54::interpolate(0.5, &sol).unwrap_err(),
            OdeError::NoDenseOutput
        );
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
            0.0,
            0.01,
            &y0,
            |_t, y| Vector::from_array([-1000.0 * y[0]]),
            |_t, _y| crate::Matrix::new([[-1000.0]]),
            &settings,
        )
        .unwrap();
        let exact = (-1000.0_f64 * 0.01).exp();
        assert!(
            (sol.y[0] - exact).abs() < 1e-8,
            "rodas4 stiff decay: y = {}, exact = {}, err = {}",
            sol.y[0],
            exact,
            (sol.y[0] - exact).abs()
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
            0.0,
            100.0,
            &y0,
            |_t, y| Vector::from_array([y[1], mu * ((1.0 - y[0] * y[0]) * y[1] - y[0])]),
            |_t, y| {
                crate::Matrix::new([
                    [0.0, 1.0],
                    [mu * (-2.0 * y[0] * y[1] - 1.0), mu * (1.0 - y[0] * y[0])],
                ])
            },
            &settings,
        )
        .unwrap();

        // Solution should stay bounded (|y₁| ≤ ~2.1 for Van der Pol limit cycle)
        assert!(
            sol.y[0].abs() < 3.0,
            "Van der Pol y[0] = {} exceeds bound",
            sol.y[0]
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

        #[rustfmt::skip]
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
            "Robertson mass conservation violated: sum = {}",
            mass
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
            0.0,
            0.01,
            &y0,
            |_t, y| Vector::from_array([-1000.0 * y[0]]),
            &settings,
        )
        .unwrap();
        let exact = (-1000.0_f64 * 0.01).exp();
        assert!(
            (sol.y[0] - exact).abs() < 1e-6,
            "auto-jac stiff decay: y = {}, exact = {}, err = {}",
            sol.y[0],
            exact,
            (sol.y[0] - exact).abs()
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
            0.0,
            1e3,
            &y0,
            |_t, y| {
                Vector::from_array([
                    -0.04 * y[0] + 1e4 * y[1] * y[2],
                    0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1],
                    3e7 * y[1] * y[1],
                ])
            },
            &settings,
        )
        .unwrap();

        let mass: f64 = sol.y[0] + sol.y[1] + sol.y[2];
        assert!(
            (mass - 1.0).abs() < 1e-4,
            "auto-jac Robertson mass conservation: sum = {}",
            mass
        );
    }

    #[test]
    fn too_many_rejections_explicit() {
        // Absurdly tight tolerance on a simple problem should trigger
        // consecutive rejection limit before max_steps is reached.
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let settings = AdaptiveSettings {
            abs_tol: 1e-100,
            rel_tol: 1e-100,
            min_step: 0.0, // disable min_step acceptance
            max_steps: 1_000_000,
            ..AdaptiveSettings::default()
        };
        let result = RKTS54::integrate(0.0, TAU, &y0, ydot, &settings);
        assert!(matches!(result, Err(OdeError::TooManyRejections)));
    }

    #[test]
    fn too_many_rejections_rosenbrock() {
        // Same test for Rosenbrock solver.
        let y0 = Vector::from_array([1.0_f64]);
        let settings = AdaptiveSettings {
            abs_tol: 1e-100,
            rel_tol: 1e-100,
            min_step: 0.0,
            max_steps: 1_000_000,
            ..AdaptiveSettings::default()
        };
        let result = RODAS4::integrate(
            0.0,
            0.01,
            &y0,
            |_t, y| Vector::from_array([-1000.0 * y[0]]),
            |_t, _y| crate::Matrix::new([[-1000.0]]),
            &settings,
        );
        assert!(matches!(result, Err(OdeError::TooManyRejections)));
    }

    #[test]
    fn h_min_prevents_rejection_loop() {
        // Verify that h_min prevents TooManyRejections by forcing acceptance
        // at the minimum step size. Use tight tolerance that would normally
        // cause rejections, but h_min should force acceptance once h shrinks.
        let y0 = Vector::from_array([1.0_f64, 0.0]);

        // Without h_min: tight tolerance + min_step=0 triggers TooManyRejections
        let settings_no_hmin = AdaptiveSettings {
            abs_tol: 1e-100,
            rel_tol: 1e-100,
            min_step: 0.0,
            max_steps: 1_000_000,
            ..AdaptiveSettings::default()
        };
        let result_no_hmin = RKTS54::integrate(0.0, TAU, &y0, ydot, &settings_no_hmin);
        assert!(matches!(result_no_hmin, Err(OdeError::TooManyRejections)));

        // With h_min: same tight tolerance, but h_min forces acceptance
        let settings_hmin = AdaptiveSettings {
            abs_tol: 1e-100,
            rel_tol: 1e-100,
            min_step: 0.0,
            max_steps: 1_000_000,
            h_min: Some(0.5),
            ..AdaptiveSettings::default()
        };
        let result_hmin = RKTS54::integrate(0.0, TAU, &y0, ydot, &settings_hmin);
        // Should NOT be TooManyRejections — h_min forces acceptance
        assert!(!matches!(result_hmin, Err(OdeError::TooManyRejections)));
    }

    // ── Kinks / discontinuities in the right-hand side ─────────────────
    //
    // A step straddling a jump in `f` has local error O(h) instead of
    // O(h^(p+1)), so the order-based rejection shrink is far too gentle and
    // the old fixed limit of 10 consecutive rejections aborted the integration
    // with `TooManyRejections`. Every test below failed before the forced
    // 2× shrink on repeated rejections (see `adaptive::rejected_step_factor`).

    /// `y' = sign(0.5 − t)`, `y(0) = 0` ⇒ `y(1) = 0` exactly.
    fn kink_rhs(t: f64, _y: &Vector<f64, 1>) -> Vector<f64, 1> {
        Vector::from_array([if t < 0.5 { 1.0 } else { -1.0 }])
    }

    /// Tight tolerance with `min_step = 0`, so a straddling step can only be
    /// accepted once its O(h) error is genuinely under tolerance — no forced
    /// acceptance path hides a failure to converge.
    fn kink_settings() -> AdaptiveSettings<f64> {
        AdaptiveSettings {
            abs_tol: 1e-10,
            rel_tol: 1e-10,
            min_step: 0.0,
            max_steps: 100_000,
            ..AdaptiveSettings::default()
        }
    }

    fn check_kink<S: RKAdaptive<ST, NI>, const ST: usize, const NI: usize>(name: &str) {
        let y0 = Vector::from_array([0.0_f64]);
        let sol = S::integrate(0.0, 1.0, &y0, kink_rhs, &kink_settings())
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert!(
            sol.y[0].abs() < 1e-8,
            "{name}: y(1) = {} (expected 0)",
            sol.y[0]
        );
        // Stepping past the kink costs a few dozen rejections total; a
        // regression back to the gentle shrink would need hundreds.
        assert!(
            sol.rejected < 100,
            "{name}: {} rejections stepping past the kink",
            sol.rejected
        );
    }

    #[test]
    fn kink_in_rhs_explicit_solvers() {
        check_kink::<RKF45, 6, 1>("RKF45");
        check_kink::<RKTS54, 7, 4>("RKTS54");
        check_kink::<RKV65, 10, 6>("RKV65");
        check_kink::<RKV87, 17, 7>("RKV87");
        check_kink::<RKV98, 21, 8>("RKV98");
        check_kink::<RKV98NoInterp, 16, 1>("RKV98NoInterp");
        check_kink::<RKV98Efficient, 26, 9>("RKV98Efficient");
    }

    #[test]
    fn kink_in_rhs_rodas4() {
        // `f` does not depend on `y`, so the Jacobian is identically zero.
        let y0 = Vector::from_array([0.0_f64]);
        let sol = RODAS4::integrate(
            0.0,
            1.0,
            &y0,
            kink_rhs,
            |_t, _y| Matrix::new([[0.0]]),
            &kink_settings(),
        )
        .unwrap_or_else(|e| panic!("RODAS4: {e:?}"));
        assert!(sol.y[0].abs() < 1e-8, "RODAS4: y(1) = {}", sol.y[0]);
        assert!(sol.rejected < 100, "RODAS4: {} rejections", sol.rejected);
    }

    #[test]
    fn small_step_forcing_rkv98_tight_tol() {
        // Harmonic oscillator with a 1e-7 forcing switched on at t = π — the
        // shape of Earth-shadow switching solar radiation pressure on in an
        // orbit propagator (the kink is tiny relative to the state, so
        // `enorm` at the kink is modest and the order-based shrink is at its
        // gentlest). RKV98 at tol = 1e-10 aborted here before the fix.
        // Exact solution for t ≥ π: y = ε + (1 + ε) cos t ⇒ y(2π) = 1 + 2ε.
        let eps = 1e-7;
        let f = |t: f64, y: &Vector<f64, 2>| {
            let forcing = if t > PI { eps } else { 0.0 };
            Vector::from_array([y[1], -y[0] + forcing])
        };
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let sol = RKV98::integrate(0.0, TAU, &y0, f, &kink_settings()).unwrap();
        assert!(
            (sol.y[0] - (1.0 + 2.0 * eps)).abs() < 1e-9,
            "y(2π) = {}",
            sol.y[0]
        );
        assert!(sol.rejected < 50, "{} rejections", sol.rejected);
    }

    #[test]
    fn smooth_problem_unaffected_by_rejection_rule() {
        // The forced 2× shrink only engages from the second *consecutive*
        // rejection, which a smooth problem essentially never produces —
        // the step count must be identical to the pre-fix controller.
        let y0 = Vector::from_array([1.0_f64, 0.0]);
        let settings = AdaptiveSettings {
            abs_tol: 1e-10,
            rel_tol: 1e-10,
            ..AdaptiveSettings::default()
        };
        let sol = RKV98::integrate(0.0, TAU, &y0, ydot, &settings).unwrap();
        assert_eq!(sol.rejected, 0);
        assert!((sol.y[0] - 1.0).abs() < 1e-9);
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
            0.0,
            1.0,
            &y0,
            |_t, y| *y,
            |_t, _y| crate::Matrix::new([[1.0]]),
            &settings,
        );
        assert!(matches!(result, Err(OdeError::StepNotFinite)));
    }

    // ── Matrix state ODE tests ────────────────────────────────────

    #[test]
    fn rk4_matrix_exponential() {
        // dX/dt = A*X where A = [[0, 1], [-1, 0]] (rotation)
        // X(0) = I, exact: X(t) = [[cos(t), sin(t)], [-sin(t), cos(t)]]
        let a = Matrix::new([[0.0_f64, 1.0], [-1.0, 0.0]]);
        let x0: Matrix<f64, 2, 2> = Matrix::eye();
        let xf = rk4(0.0, TAU, 0.001, &x0, |_t, x| a * *x);
        // After one full rotation, X(2π) ≈ I
        assert!((xf[(0, 0)] - 1.0).abs() < 1e-7, "x00 = {}", xf[(0, 0)]);
        assert!(xf[(0, 1)].abs() < 1e-7, "x01 = {}", xf[(0, 1)]);
        assert!(xf[(1, 0)].abs() < 1e-7, "x10 = {}", xf[(1, 0)]);
        assert!((xf[(1, 1)] - 1.0).abs() < 1e-7, "x11 = {}", xf[(1, 1)]);
    }

    #[test]
    fn adaptive_matrix_exponential() {
        // Same problem with adaptive solver
        let a = Matrix::new([[0.0_f64, 1.0], [-1.0, 0.0]]);
        let x0: Matrix<f64, 2, 2> = Matrix::eye();
        let settings = AdaptiveSettings {
            abs_tol: 1e-10,
            rel_tol: 1e-10,
            ..AdaptiveSettings::default()
        };
        let sol = RKTS54::integrate(0.0, TAU, &x0, |_t, x| a * *x, &settings).unwrap();
        assert!(
            (sol.y[(0, 0)] - 1.0).abs() < 1e-8,
            "x00 = {}",
            sol.y[(0, 0)]
        );
        assert!(sol.y[(0, 1)].abs() < 1e-8, "x01 = {}", sol.y[(0, 1)]);
        assert!(sol.y[(1, 0)].abs() < 1e-8, "x10 = {}", sol.y[(1, 0)]);
        assert!(
            (sol.y[(1, 1)] - 1.0).abs() < 1e-8,
            "x11 = {}",
            sol.y[(1, 1)]
        );
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
            0.0,
            TAU,
            &y0,
            ydot,
            |_t, _y| crate::Matrix::new([[0.0, 1.0], [-1.0, 0.0]]),
            &settings,
        )
        .unwrap();
        assert!(
            (sol.y[0] - 1.0).abs() < 1e-6,
            "harmonic y[0] = {}, err = {}",
            sol.y[0],
            (sol.y[0] - 1.0).abs()
        );
        assert!(sol.y[1].abs() < 1e-6, "harmonic y[1] = {}", sol.y[1]);
    }

    // ── Initial-step clamp: probe must stay inside [t0, tf] ───────────

    // Nearly-constant RHS: the initial-step heuristic's raw probe step
    // (0.01·d0/d1) is many orders of magnitude larger than the integration
    // interval, so without the |tf − t0| clamp the trial evaluation samples
    // `f` far outside [t0, tf].

    #[test]
    fn initial_step_clamped_to_interval() {
        let y0 = Vector::from_array([1.0_f64]);
        let (t0, tf) = (0.0, 1e-3);
        let sol = RKTS54::integrate(
            t0,
            tf,
            &y0,
            |t, _y| {
                assert!(
                    (t0..=tf).contains(&t),
                    "RHS evaluated outside [t0, tf]: t = {t}"
                );
                Vector::from_array([1e-8])
            },
            &AdaptiveSettings::default(),
        )
        .unwrap();
        assert!((sol.y[0] - (1.0 + 1e-8 * (tf - t0))).abs() < 1e-12);
    }

    #[test]
    fn initial_step_clamped_to_interval_backward() {
        let y0 = Vector::from_array([1.0_f64]);
        let (t0, tf) = (0.0, -1e-3);
        let sol = RKTS54::integrate(
            t0,
            tf,
            &y0,
            |t, _y| {
                assert!(
                    (tf..=t0).contains(&t),
                    "RHS evaluated outside [tf, t0]: t = {t}"
                );
                Vector::from_array([1e-8])
            },
            &AdaptiveSettings::default(),
        )
        .unwrap();
        assert!((sol.y[0] - (1.0 + 1e-8 * (tf - t0))).abs() < 1e-12);
    }

    #[test]
    fn initial_step_clamped_to_interval_rosenbrock() {
        let y0 = Vector::from_array([1.0_f64]);
        let (t0, tf) = (0.0, 1e-3);
        let sol = RODAS4::integrate(
            t0,
            tf,
            &y0,
            |t, _y| {
                assert!(
                    (t0..=tf).contains(&t),
                    "RHS evaluated outside [t0, tf]: t = {t}"
                );
                Vector::from_array([1e-8])
            },
            |_t, _y| crate::Matrix::new([[0.0]]),
            &AdaptiveSettings::default(),
        )
        .unwrap();
        assert!((sol.y[0] - (1.0 + 1e-8 * (tf - t0))).abs() < 1e-12);
    }
}
