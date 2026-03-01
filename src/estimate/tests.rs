use super::*;
use crate::matrix::vector::ColumnVector;
use crate::Matrix;

fn approx_eq(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() < tol,
        "expected {} ≈ {} (diff = {}, tol = {})",
        a,
        b,
        (a - b).abs(),
        tol
    );
}

// ── EKF tests ───────────────────────────────────────────────────────

#[test]
fn ekf_linear_predict() {
    // Constant velocity: x = [pos, vel], F = [[1, dt], [0, 1]]
    let dt = 0.1;
    let x0 = ColumnVector::from_column([0.0_f64, 1.0]);
    let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
    let q = Matrix::new([[0.001, 0.0], [0.0, 0.001]]);
    let mut ekf = Ekf::<f64, 2, 1>::new(x0, p0);

    ekf.predict(
        |x| ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]]),
        |_x| Matrix::new([[1.0, dt], [0.0, 1.0]]),
        Some(&q),
    );

    approx_eq(ekf.x[(0, 0)], 0.1, 1e-12);
    approx_eq(ekf.x[(1, 0)], 1.0, 1e-12);
}

#[test]
fn ekf_linear_update() {
    let x0 = ColumnVector::from_column([0.0_f64, 1.0]);
    let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
    let r = Matrix::new([[0.1]]);
    let mut ekf = Ekf::<f64, 2, 1>::new(x0, p0);

    // Measure position = 0.5
    ekf.update(
        &ColumnVector::from_column([0.5]),
        |x| ColumnVector::from_column([x[(0, 0)]]),
        |_x| Matrix::new([[1.0, 0.0]]),
        &r,
    )
    .unwrap();

    // State should move toward measurement
    assert!(ekf.x[(0, 0)] > 0.0);
    assert!(ekf.x[(0, 0)] < 0.5);
    // Covariance should decrease
    assert!(ekf.p[(0, 0)] < 1.0);
}

#[test]
fn ekf_converges_to_true_state() {
    // Linear constant-velocity, measure position. True state: pos=5, vel=1
    let dt = 0.1;
    let x0 = ColumnVector::from_column([0.0_f64, 0.0]);
    let p0 = Matrix::new([[10.0, 0.0], [0.0, 10.0]]);
    let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
    let r = Matrix::new([[0.1]]);
    let mut ekf = Ekf::<f64, 2, 1>::new(x0, p0);

    let mut true_pos = 5.0;
    let true_vel = 1.0;

    for _ in 0..100 {
        ekf.predict(
            |x| ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]]),
            |_x| Matrix::new([[1.0, dt], [0.0, 1.0]]),
            Some(&q),
        );

        true_pos += dt * true_vel;
        ekf.update(
            &ColumnVector::from_column([true_pos]),
            |x| ColumnVector::from_column([x[(0, 0)]]),
            |_x| Matrix::new([[1.0, 0.0]]),
            &r,
        )
        .unwrap();
    }

    approx_eq(ekf.x[(0, 0)], true_pos, 0.5);
    approx_eq(ekf.x[(1, 0)], true_vel, 0.5);
}

#[test]
fn ekf_predict_fd_matches_explicit() {
    let dt = 0.1;
    let x0 = ColumnVector::from_column([1.0_f64, 2.0]);
    let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
    let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);

    let f = |x: &ColumnVector<f64, 2>| {
        ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]])
    };
    let fj = |_x: &ColumnVector<f64, 2>| Matrix::new([[1.0, dt], [0.0, 1.0]]);

    let mut ekf_explicit = Ekf::<f64, 2, 1>::new(x0, p0);
    let mut ekf_fd = Ekf::<f64, 2, 1>::new(x0, p0);

    ekf_explicit.predict(f, fj, Some(&q));
    ekf_fd.predict_fd(f, Some(&q));

    for i in 0..2 {
        approx_eq(ekf_explicit.x[(i, 0)], ekf_fd.x[(i, 0)], 1e-10);
        for j in 0..2 {
            approx_eq(ekf_explicit.p[(i, j)], ekf_fd.p[(i, j)], 1e-6);
        }
    }
}

#[test]
fn ekf_update_fd_matches_explicit() {
    let x0 = ColumnVector::from_column([1.0_f64, 2.0]);
    let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
    let r = Matrix::new([[0.1]]);
    let z = ColumnVector::from_column([1.5]);

    let h = |x: &ColumnVector<f64, 2>| ColumnVector::from_column([x[(0, 0)]]);
    let hj = |_x: &ColumnVector<f64, 2>| Matrix::new([[1.0, 0.0]]);

    let mut ekf_explicit = Ekf::<f64, 2, 1>::new(x0, p0);
    let mut ekf_fd = Ekf::<f64, 2, 1>::new(x0, p0);

    ekf_explicit.update(&z, h, hj, &r).unwrap();
    ekf_fd.update_fd(&z, h, &r).unwrap();

    for i in 0..2 {
        approx_eq(ekf_explicit.x[(i, 0)], ekf_fd.x[(i, 0)], 1e-6);
        for j in 0..2 {
            approx_eq(ekf_explicit.p[(i, j)], ekf_fd.p[(i, j)], 1e-4);
        }
    }
}

#[test]
fn ekf_nonlinear_measurement() {
    // State: [x, y], Measurement: range = sqrt(x² + y²)
    let x0 = ColumnVector::from_column([3.0_f64, 4.0]);
    let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
    let r = Matrix::new([[0.01]]);
    let mut ekf = Ekf::<f64, 2, 1>::new(x0, p0);

    let true_range = 5.0; // sqrt(9 + 16)
    ekf.update(
        &ColumnVector::from_column([true_range]),
        |x| ColumnVector::from_column([(x[(0, 0)] * x[(0, 0)] + x[(1, 0)] * x[(1, 0)]).sqrt()]),
        |x| {
            let r = (x[(0, 0)] * x[(0, 0)] + x[(1, 0)] * x[(1, 0)]).sqrt();
            Matrix::new([[x[(0, 0)] / r, x[(1, 0)] / r]])
        },
        &r,
    )
    .unwrap();

    // State should stay near [3, 4] since measurement matches
    approx_eq(ekf.x[(0, 0)], 3.0, 0.1);
    approx_eq(ekf.x[(1, 0)], 4.0, 0.1);
}

#[test]
fn ekf_zero_process_noise() {
    let x0 = ColumnVector::from_column([1.0_f64, 2.0]);
    let p0 = Matrix::new([[0.5, 0.0], [0.0, 0.5]]);
    let mut ekf = Ekf::<f64, 2, 1>::new(x0, p0);

    // Identity dynamics, no process noise
    ekf.predict(|x| *x, |_x| Matrix::eye(), None);

    approx_eq(ekf.x[(0, 0)], 1.0, 1e-14);
    approx_eq(ekf.x[(1, 0)], 2.0, 1e-14);
    approx_eq(ekf.p[(0, 0)], 0.5, 1e-14);
    approx_eq(ekf.p[(1, 1)], 0.5, 1e-14);
}

#[test]
fn ekf_joseph_form_symmetry() {
    // After many updates, P should remain symmetric
    let x0 = ColumnVector::from_column([0.0_f64, 0.0, 0.0]);
    let p0 = Matrix::new([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]);
    let q = Matrix::new([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]);
    let r = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
    let mut ekf = Ekf::<f64, 3, 2>::new(x0, p0);

    for k in 0..50 {
        let t = k as f64 * 0.1;
        ekf.predict(
            |x| {
                ColumnVector::from_column([
                    x[(0, 0)] + 0.1 * x[(1, 0)],
                    x[(1, 0)] + 0.1 * x[(2, 0)],
                    x[(2, 0)],
                ])
            },
            |_x| {
                Matrix::new([
                    [1.0, 0.1, 0.0],
                    [0.0, 1.0, 0.1],
                    [0.0, 0.0, 1.0],
                ])
            },
            Some(&q),
        );

        let z = ColumnVector::from_column([t.sin(), t.cos()]);
        ekf.update(
            &z,
            |x| ColumnVector::from_column([x[(0, 0)], x[(1, 0)]]),
            |_x| {
                Matrix::new([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ])
            },
            &r,
        )
        .unwrap();
    }

    // Check symmetry
    for i in 0..3 {
        for j in 0..3 {
            approx_eq(ekf.p[(i, j)], ekf.p[(j, i)], 1e-10);
        }
    }
}

#[test]
fn ekf_f32() {
    let x0 = ColumnVector::from_column([0.0_f32, 1.0]);
    let p0 = Matrix::new([[1.0_f32, 0.0], [0.0, 1.0]]);
    let q = Matrix::new([[0.01_f32, 0.0], [0.0, 0.01]]);
    let r = Matrix::new([[0.5_f32]]);
    let mut ekf = Ekf::<f32, 2, 1>::new(x0, p0);

    let dt = 0.1_f32;
    ekf.predict(
        |x| ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]]),
        |_x| Matrix::new([[1.0, dt], [0.0, 1.0]]),
        Some(&q),
    );

    ekf.update(
        &ColumnVector::from_column([0.12_f32]),
        |x| ColumnVector::from_column([x[(0, 0)]]),
        |_x| Matrix::new([[1.0_f32, 0.0]]),
        &r,
    )
    .unwrap();

    // Just check it runs and produces reasonable output
    assert!(ekf.x[(0, 0)].is_finite());
    assert!(ekf.p[(0, 0)] > 0.0);
}

// ── UKF tests ───────────────────────────────────────────────────────

#[cfg(feature = "alloc")]
mod ukf_tests {
    use super::*;

    #[test]
    fn ukf_linear_predict() {
        let dt = 0.1;
        let x0 = ColumnVector::from_column([0.0_f64, 1.0]);
        let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        let q = Matrix::new([[0.001, 0.0], [0.0, 0.001]]);
        let mut ukf = Ukf::<f64, 2, 1>::new(x0, p0);

        ukf.predict(
            |x| ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]]),
            Some(&q),
        )
        .unwrap();

        approx_eq(ukf.x[(0, 0)], 0.1, 1e-6);
        approx_eq(ukf.x[(1, 0)], 1.0, 1e-6);
    }

    #[test]
    fn ukf_linear_update() {
        let x0 = ColumnVector::from_column([0.0_f64, 1.0]);
        let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        let r = Matrix::new([[0.1]]);
        let mut ukf = Ukf::<f64, 2, 1>::new(x0, p0);

        ukf.update(
            &ColumnVector::from_column([0.5]),
            |x| ColumnVector::from_column([x[(0, 0)]]),
            &r,
        )
        .unwrap();

        // State should move toward measurement
        assert!(ukf.x[(0, 0)] > 0.0);
        assert!(ukf.x[(0, 0)] < 0.5);
        assert!(ukf.p[(0, 0)] < 1.0);
    }

    #[test]
    fn ukf_converges_to_true_state() {
        let dt = 0.1;
        let x0 = ColumnVector::from_column([0.0_f64, 0.0]);
        let p0 = Matrix::new([[10.0, 0.0], [0.0, 10.0]]);
        let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
        let r = Matrix::new([[0.1]]);
        let mut ukf = Ukf::<f64, 2, 1>::new(x0, p0);

        let mut true_pos = 5.0;
        let true_vel = 1.0;

        for _ in 0..100 {
            ukf.predict(
                |x| ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]]),
                Some(&q),
            )
            .unwrap();

            true_pos += dt * true_vel;
            ukf.update(
                &ColumnVector::from_column([true_pos]),
                |x| ColumnVector::from_column([x[(0, 0)]]),
                &r,
            )
            .unwrap();
        }

        approx_eq(ukf.x[(0, 0)], true_pos, 0.5);
        approx_eq(ukf.x[(1, 0)], true_vel, 0.5);
    }

    #[test]
    fn ukf_nonlinear_measurement() {
        // State: [x, y], Measurement: range = sqrt(x² + y²)
        let x0 = ColumnVector::from_column([3.0_f64, 4.0]);
        let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        let r = Matrix::new([[0.01]]);
        let mut ukf = Ukf::<f64, 2, 1>::new(x0, p0);

        let true_range = 5.0;
        ukf.update(
            &ColumnVector::from_column([true_range]),
            |x| {
                ColumnVector::from_column([
                    (x[(0, 0)] * x[(0, 0)] + x[(1, 0)] * x[(1, 0)]).sqrt(),
                ])
            },
            &r,
        )
        .unwrap();

        approx_eq(ukf.x[(0, 0)], 3.0, 0.2);
        approx_eq(ukf.x[(1, 0)], 4.0, 0.2);
    }

    #[test]
    fn ukf_custom_params() {
        let x0 = ColumnVector::from_column([0.0_f64, 1.0]);
        let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);

        // alpha=0.5 (wider sigma spread)
        let mut ukf = Ukf::<f64, 2, 1>::with_params(x0, p0, 0.5, 2.0, 0.0);

        ukf.predict(
            |x| ColumnVector::from_column([x[(0, 0)] + 0.1 * x[(1, 0)], x[(1, 0)]]),
            Some(&q),
        )
        .unwrap();

        // Should still produce reasonable results
        approx_eq(ukf.x[(0, 0)], 0.1, 1e-6);
        approx_eq(ukf.x[(1, 0)], 1.0, 1e-6);
    }

    #[test]
    fn ukf_covariance_stays_spd() {
        let x0 = ColumnVector::from_column([0.0_f64, 0.0, 0.0]);
        let p0 = Matrix::new([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]);
        let q = Matrix::new([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]);
        let r = Matrix::new([[1.0]]);
        let mut ukf = Ukf::<f64, 3, 1>::new(x0, p0);

        for k in 0..30 {
            ukf.predict(
                |x| {
                    ColumnVector::from_column([
                        x[(0, 0)] + 0.1 * x[(1, 0)],
                        x[(1, 0)] + 0.1 * x[(2, 0)],
                        x[(2, 0)],
                    ])
                },
                Some(&q),
            )
            .unwrap();

            let z_val = (k as f64 * 0.1).sin();
            ukf.update(
                &ColumnVector::from_column([z_val]),
                |x| ColumnVector::from_column([x[(0, 0)]]),
                &r,
            )
            .unwrap();
        }

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(ukf.p[(i, j)], ukf.p[(j, i)], 1e-10);
            }
        }

        // Check positive diagonal (necessary for PD)
        for i in 0..3 {
            assert!(ukf.p[(i, i)] > 0.0, "P[{},{}] = {} not positive", i, i, ukf.p[(i, i)]);
        }

        // Verify Cholesky succeeds (sufficient for PD)
        assert!(ukf.p.cholesky().is_ok());
    }

    #[test]
    fn ukf_f32() {
        let x0 = ColumnVector::from_column([0.0_f32, 1.0]);
        let p0 = Matrix::new([[1.0_f32, 0.0], [0.0, 1.0]]);
        let q = Matrix::new([[0.01_f32, 0.0], [0.0, 0.01]]);
        let r = Matrix::new([[0.5_f32]]);
        let mut ukf = Ukf::<f32, 2, 1>::new(x0, p0);

        ukf.predict(
            |x| ColumnVector::from_column([x[(0, 0)] + 0.1 * x[(1, 0)], x[(1, 0)]]),
            Some(&q),
        )
        .unwrap();

        ukf.update(
            &ColumnVector::from_column([0.12_f32]),
            |x| ColumnVector::from_column([x[(0, 0)]]),
            &r,
        )
        .unwrap();

        assert!(ukf.x[(0, 0)].is_finite());
        assert!(ukf.p[(0, 0)] > 0.0);
    }

    #[test]
    fn ukf_covariance_not_pd_error() {
        let x0 = ColumnVector::from_column([0.0_f64, 0.0]);
        // Not positive definite — negative diagonal
        let p0 = Matrix::new([[-1.0, 0.0], [0.0, -1.0]]);
        let mut ukf = Ukf::<f64, 2, 1>::new(x0, p0);

        let result = ukf.predict(|x| *x, None);
        assert_eq!(result, Err(EstimateError::CovarianceNotPD));
    }

    #[test]
    fn ukf_predict_no_process_noise() {
        let x0 = ColumnVector::from_column([1.0_f64, 2.0]);
        let p0 = Matrix::new([[0.5, 0.0], [0.0, 0.5]]);
        let mut ukf = Ukf::<f64, 2, 1>::new(x0, p0);

        // Identity dynamics, no process noise
        ukf.predict(|x| *x, None).unwrap();

        approx_eq(ukf.x[(0, 0)], 1.0, 1e-6);
        approx_eq(ukf.x[(1, 0)], 2.0, 1e-6);
    }

    // ── Cross-filter tests ────────────────────────────────────────

    #[test]
    fn ekf_ukf_agree_linear() {
        // On a purely linear problem, EKF and UKF should give very similar results.
        let dt = 0.1;
        let x0 = ColumnVector::from_column([0.0_f64, 1.0]);
        let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
        let r = Matrix::new([[0.5]]);

        let mut ekf = Ekf::<f64, 2, 1>::new(x0, p0);
        let mut ukf = Ukf::<f64, 2, 1>::with_params(x0, p0, 1.0, 2.0, 0.0);

        let f = |x: &ColumnVector<f64, 2>| {
            ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]])
        };
        let fj = |_x: &ColumnVector<f64, 2>| Matrix::new([[1.0, dt], [0.0, 1.0]]);
        let h = |x: &ColumnVector<f64, 2>| ColumnVector::from_column([x[(0, 0)]]);
        let hj = |_x: &ColumnVector<f64, 2>| Matrix::new([[1.0, 0.0]]);

        let measurements = [0.1, 0.22, 0.31, 0.45, 0.53, 0.68, 0.79, 0.9, 1.02, 1.11];

        for &z_val in &measurements {
            ekf.predict(f, fj, Some(&q));
            ukf.predict(f, Some(&q)).unwrap();

            let z = ColumnVector::from_column([z_val]);
            ekf.update(&z, h, hj, &r).unwrap();
            ukf.update(&z, h, &r).unwrap();
        }

        // UKF with alpha=1.0 should give nearly identical results to EKF on linear problem
        for i in 0..2 {
            approx_eq(ekf.x[(i, 0)], ukf.x[(i, 0)], 0.05);
        }
    }

    #[test]
    fn ekf_ukf_both_converge_nonlinear() {
        // Nonlinear dynamics: polar kinematics
        // State: [r, theta], Measurement: x-position = r * cos(theta)
        let x0 = ColumnVector::from_column([5.0_f64, 0.3]);
        let p0 = Matrix::new([[1.0, 0.0], [0.0, 0.1]]);
        let q = Matrix::new([[0.01, 0.0], [0.0, 0.001]]);
        let r = Matrix::new([[0.1]]);

        let mut ekf = Ekf::<f64, 2, 1>::new(x0, p0);
        let mut ukf = Ukf::<f64, 2, 1>::new(x0, p0);

        // True: r=5, theta=0.3 → x = 5*cos(0.3) ≈ 4.7766
        let true_x = 5.0 * 0.3_f64.cos();

        let h = |x: &ColumnVector<f64, 2>| ColumnVector::from_column([x[(0, 0)] * x[(1, 0)].cos()]);
        let hj = |x: &ColumnVector<f64, 2>| {
            Matrix::new([[x[(1, 0)].cos(), -x[(0, 0)] * x[(1, 0)].sin()]])
        };

        for _ in 0..20 {
            // Identity dynamics
            ekf.predict(|x| *x, |_x| Matrix::eye(), Some(&q));
            ukf.predict(|x| *x, Some(&q)).unwrap();

            let z = ColumnVector::from_column([true_x]);
            ekf.update(&z, h, hj, &r).unwrap();
            ukf.update(&z, h, &r).unwrap();
        }

        // Both should have reasonable x-position predictions
        let ekf_x_pred = ekf.x[(0, 0)] * ekf.x[(1, 0)].cos();
        let ukf_x_pred = ukf.x[(0, 0)] * ukf.x[(1, 0)].cos();

        approx_eq(ekf_x_pred, true_x, 0.5);
        approx_eq(ukf_x_pred, true_x, 0.5);
    }
}

// ── cholupdate tests ────────────────────────────────────────────────

#[cfg(feature = "alloc")]
mod cholupdate_tests {
    use super::*;
    use super::super::cholupdate::cholupdate;
    use crate::linalg::CholeskyDecomposition;

    #[test]
    fn cholupdate_rank1_update_roundtrip() {
        // P = LLᵀ, add v·vᵀ, verify L'·L'ᵀ = P + v·vᵀ
        let p = Matrix::new([[4.0, 2.0], [2.0, 3.0_f64]]);
        let chol = CholeskyDecomposition::new(&p).unwrap();
        let mut l = chol.l_full();
        let v_orig = ColumnVector::from_column([1.0, 0.5_f64]);
        let mut v = v_orig;

        cholupdate(&mut l, &mut v, 1.0).unwrap();

        let p_new = l * l.transpose();
        let p_expected = p + v_orig * v_orig.transpose();
        for i in 0..2 {
            for j in 0..2 {
                approx_eq(p_new[(i, j)], p_expected[(i, j)], 1e-10);
            }
        }
    }

    #[test]
    fn cholupdate_rank1_downdate_roundtrip() {
        // Start with P + v·vᵀ, downdate by v·vᵀ, recover P
        let p_base = Matrix::new([[4.0, 2.0], [2.0, 3.0_f64]]);
        let v = ColumnVector::from_column([0.5, 0.3_f64]);
        let p_aug = p_base + v * v.transpose();

        let chol = CholeskyDecomposition::new(&p_aug).unwrap();
        let mut l = chol.l_full();
        let mut v_work = v;

        cholupdate(&mut l, &mut v_work, -1.0).unwrap();

        let p_recovered = l * l.transpose();
        for i in 0..2 {
            for j in 0..2 {
                approx_eq(p_recovered[(i, j)], p_base[(i, j)], 1e-10);
            }
        }
    }

    #[test]
    fn cholupdate_downdate_fails_non_pd() {
        // Try to downdate identity by a vector that would make result non-PD
        let mut l = Matrix::<f64, 2, 2>::eye();
        let mut v = ColumnVector::from_column([1.5, 0.0]);

        let result = cholupdate(&mut l, &mut v, -1.0);
        assert_eq!(result, Err(EstimateError::CholdowndateFailed));
    }

    #[test]
    fn cholupdate_large_matrix() {
        // 4×4 update
        let p = Matrix::new([
            [5.0, 1.0, 0.5, 0.1],
            [1.0, 4.0, 0.3, 0.2],
            [0.5, 0.3, 3.0, 0.1],
            [0.1, 0.2, 0.1, 2.0_f64],
        ]);
        let chol = CholeskyDecomposition::new(&p).unwrap();
        let mut l = chol.l_full();
        let v_orig = ColumnVector::from_column([0.3, 0.7, 0.1, 0.5]);
        let mut v = v_orig;

        cholupdate(&mut l, &mut v, 1.0).unwrap();

        let p_new = l * l.transpose();
        let p_expected = p + v_orig * v_orig.transpose();
        for i in 0..4 {
            for j in 0..4 {
                approx_eq(p_new[(i, j)], p_expected[(i, j)], 1e-10);
            }
        }
    }
}

// ── SR-UKF tests ────────────────────────────────────────────────────

#[cfg(feature = "alloc")]
mod srukf_tests {
    use super::*;

    #[test]
    fn srukf_linear_cv_converges() {
        let dt = 0.1;
        let x0 = ColumnVector::from_column([0.0_f64, 0.0]);
        let p0 = Matrix::new([[10.0, 0.0], [0.0, 10.0]]);
        let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
        let r = Matrix::new([[0.1]]);
        let mut srukf = SrUkf::<f64, 2, 1>::from_covariance(x0, p0).unwrap();

        let mut true_pos = 5.0;
        let true_vel = 1.0;

        for _ in 0..100 {
            srukf
                .predict(
                    |x| ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]]),
                    Some(&q),
                )
                .unwrap();

            true_pos += dt * true_vel;
            srukf
                .update(
                    &ColumnVector::from_column([true_pos]),
                    |x| ColumnVector::from_column([x[(0, 0)]]),
                    &r,
                )
                .unwrap();
        }

        approx_eq(srukf.x[(0, 0)], true_pos, 0.5);
        approx_eq(srukf.x[(1, 0)], true_vel, 0.5);
    }

    #[test]
    fn srukf_agrees_with_ukf_linear() {
        let dt = 0.1;
        let x0 = ColumnVector::from_column([0.0_f64, 1.0]);
        let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
        let r = Matrix::new([[0.5]]);

        let mut ukf = Ukf::<f64, 2, 1>::with_params(x0, p0, 0.5, 2.0, 0.0);
        let mut srukf =
            SrUkf::<f64, 2, 1>::from_covariance_with_params(x0, p0, 0.5, 2.0, 0.0).unwrap();

        let f = |x: &ColumnVector<f64, 2>| {
            ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]])
        };
        let h = |x: &ColumnVector<f64, 2>| ColumnVector::from_column([x[(0, 0)]]);

        let measurements = [0.1, 0.22, 0.31, 0.45, 0.53];
        for &z_val in &measurements {
            ukf.predict(f, Some(&q)).unwrap();
            srukf.predict(f, Some(&q)).unwrap();

            let z = ColumnVector::from_column([z_val]);
            ukf.update(&z, h, &r).unwrap();
            srukf.update(&z, h, &r).unwrap();
        }

        // States should be very close
        for i in 0..2 {
            approx_eq(ukf.x[(i, 0)], srukf.x[(i, 0)], 0.1);
        }

        // Covariances should be close
        let srukf_p = srukf.covariance();
        for i in 0..2 {
            for j in 0..2 {
                approx_eq(ukf.p[(i, j)], srukf_p[(i, j)], 0.1);
            }
        }
    }

    #[test]
    fn srukf_covariance_stays_pd() {
        let x0 = ColumnVector::from_column([0.0_f64, 0.0, 0.0]);
        let p0 = Matrix::new([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]);
        let q = Matrix::new([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]);
        let r = Matrix::new([[1.0]]);
        let mut srukf = SrUkf::<f64, 3, 1>::from_covariance(x0, p0).unwrap();

        for k in 0..30 {
            srukf
                .predict(
                    |x| {
                        ColumnVector::from_column([
                            x[(0, 0)] + 0.1 * x[(1, 0)],
                            x[(1, 0)] + 0.1 * x[(2, 0)],
                            x[(2, 0)],
                        ])
                    },
                    Some(&q),
                )
                .unwrap();

            let z_val = (k as f64 * 0.1).sin();
            srukf
                .update(
                    &ColumnVector::from_column([z_val]),
                    |x| ColumnVector::from_column([x[(0, 0)]]),
                    &r,
                )
                .unwrap();
        }

        // S should still be valid → P = SSᵀ should be PD
        let p = srukf.covariance();
        assert!(p.cholesky().is_ok());
    }

    #[test]
    fn srukf_predict_only() {
        let x0 = ColumnVector::from_column([1.0_f64, 2.0]);
        let p0 = Matrix::new([[0.5, 0.0], [0.0, 0.5]]);
        let mut srukf = SrUkf::<f64, 2, 1>::from_covariance(x0, p0).unwrap();

        srukf.predict(|x| *x, None).unwrap();

        approx_eq(srukf.x[(0, 0)], 1.0, 1e-6);
        approx_eq(srukf.x[(1, 0)], 2.0, 1e-6);
    }

    #[test]
    fn srukf_custom_params() {
        let x0 = ColumnVector::from_column([0.0_f64, 1.0]);
        let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);

        let mut srukf =
            SrUkf::<f64, 2, 1>::from_covariance_with_params(x0, p0, 0.5, 2.0, 0.0).unwrap();
        srukf
            .predict(
                |x| ColumnVector::from_column([x[(0, 0)] + 0.1 * x[(1, 0)], x[(1, 0)]]),
                Some(&q),
            )
            .unwrap();

        approx_eq(srukf.x[(0, 0)], 0.1, 1e-6);
        approx_eq(srukf.x[(1, 0)], 1.0, 1e-6);
    }

    #[test]
    fn srukf_nonlinear_measurement() {
        let x0 = ColumnVector::from_column([3.0_f64, 4.0]);
        let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        let r = Matrix::new([[0.01]]);
        let mut srukf = SrUkf::<f64, 2, 1>::from_covariance(x0, p0).unwrap();

        let true_range = 5.0;
        srukf
            .update(
                &ColumnVector::from_column([true_range]),
                |x| {
                    ColumnVector::from_column([
                        (x[(0, 0)] * x[(0, 0)] + x[(1, 0)] * x[(1, 0)]).sqrt(),
                    ])
                },
                &r,
            )
            .unwrap();

        approx_eq(srukf.x[(0, 0)], 3.0, 0.3);
        approx_eq(srukf.x[(1, 0)], 4.0, 0.3);
    }

    #[test]
    fn srukf_f32() {
        let x0 = ColumnVector::from_column([0.0_f32, 1.0]);
        let p0 = Matrix::new([[1.0_f32, 0.0], [0.0, 1.0]]);
        let q = Matrix::new([[0.01_f32, 0.0], [0.0, 0.01]]);
        let r = Matrix::new([[0.5_f32]]);
        let mut srukf = SrUkf::<f32, 2, 1>::from_covariance(x0, p0).unwrap();

        srukf
            .predict(
                |x| ColumnVector::from_column([x[(0, 0)] + 0.1 * x[(1, 0)], x[(1, 0)]]),
                Some(&q),
            )
            .unwrap();

        srukf
            .update(
                &ColumnVector::from_column([0.12_f32]),
                |x| ColumnVector::from_column([x[(0, 0)]]),
                &r,
            )
            .unwrap();

        assert!(srukf.x[(0, 0)].is_finite());
        let p = srukf.covariance();
        assert!(p[(0, 0)] > 0.0);
    }

    #[test]
    fn srukf_covariance_not_pd_error() {
        let x0 = ColumnVector::from_column([0.0_f64, 0.0]);
        let p0 = Matrix::new([[-1.0, 0.0], [0.0, -1.0]]);
        let result = SrUkf::<f64, 2, 1>::from_covariance(x0, p0);
        match result {
            Err(EstimateError::CovarianceNotPD) => {}
            _ => panic!("expected CovarianceNotPD error"),
        }
    }
}

// ── CKF tests ───────────────────────────────────────────────────────

#[cfg(feature = "alloc")]
mod ckf_tests {
    use super::*;

    #[test]
    fn ckf_linear_cv_converges() {
        let dt = 0.1;
        let x0 = ColumnVector::from_column([0.0_f64, 0.0]);
        let p0 = Matrix::new([[10.0, 0.0], [0.0, 10.0]]);
        let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
        let r = Matrix::new([[0.1]]);
        let mut ckf = Ckf::<f64, 2, 1>::new(x0, p0);

        let mut true_pos = 5.0;
        let true_vel = 1.0;

        for _ in 0..100 {
            ckf.predict(
                |x| ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]]),
                Some(&q),
            )
            .unwrap();

            true_pos += dt * true_vel;
            ckf.update(
                &ColumnVector::from_column([true_pos]),
                |x| ColumnVector::from_column([x[(0, 0)]]),
                &r,
            )
            .unwrap();
        }

        approx_eq(ckf.x[(0, 0)], true_pos, 0.5);
        approx_eq(ckf.x[(1, 0)], true_vel, 0.5);
    }

    #[test]
    fn ckf_agrees_with_ekf_linear() {
        let dt = 0.1;
        let x0 = ColumnVector::from_column([0.0_f64, 1.0]);
        let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
        let r = Matrix::new([[0.5]]);

        let mut ekf = Ekf::<f64, 2, 1>::new(x0, p0);
        let mut ckf = Ckf::<f64, 2, 1>::new(x0, p0);

        let f = |x: &ColumnVector<f64, 2>| {
            ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]])
        };
        let fj = |_x: &ColumnVector<f64, 2>| Matrix::new([[1.0, dt], [0.0, 1.0]]);
        let h = |x: &ColumnVector<f64, 2>| ColumnVector::from_column([x[(0, 0)]]);
        let hj = |_x: &ColumnVector<f64, 2>| Matrix::new([[1.0, 0.0]]);

        let measurements = [0.1, 0.22, 0.31, 0.45, 0.53, 0.68, 0.79, 0.9, 1.02, 1.11];
        for &z_val in &measurements {
            ekf.predict(f, fj, Some(&q));
            ckf.predict(f, Some(&q)).unwrap();

            let z = ColumnVector::from_column([z_val]);
            ekf.update(&z, h, hj, &r).unwrap();
            ckf.update(&z, h, &r).unwrap();
        }

        // CKF should be close to EKF on linear problem
        for i in 0..2 {
            approx_eq(ekf.x[(i, 0)], ckf.x[(i, 0)], 0.1);
        }
    }

    #[test]
    fn ckf_nonlinear_dynamics() {
        // Nonlinear range measurement
        let x0 = ColumnVector::from_column([3.0_f64, 4.0]);
        let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        let r = Matrix::new([[0.01]]);
        let mut ckf = Ckf::<f64, 2, 1>::new(x0, p0);

        let true_range = 5.0;
        ckf.update(
            &ColumnVector::from_column([true_range]),
            |x| {
                ColumnVector::from_column([
                    (x[(0, 0)] * x[(0, 0)] + x[(1, 0)] * x[(1, 0)]).sqrt(),
                ])
            },
            &r,
        )
        .unwrap();

        approx_eq(ckf.x[(0, 0)], 3.0, 0.3);
        approx_eq(ckf.x[(1, 0)], 4.0, 0.3);
    }

    #[test]
    fn ckf_predict_only() {
        let x0 = ColumnVector::from_column([1.0_f64, 2.0]);
        let p0 = Matrix::new([[0.5, 0.0], [0.0, 0.5]]);
        let mut ckf = Ckf::<f64, 2, 1>::new(x0, p0);

        ckf.predict(|x| *x, None).unwrap();

        approx_eq(ckf.x[(0, 0)], 1.0, 1e-6);
        approx_eq(ckf.x[(1, 0)], 2.0, 1e-6);
    }

    #[test]
    fn ckf_predict_no_process_noise() {
        let x0 = ColumnVector::from_column([1.0_f64, 2.0]);
        let p0 = Matrix::new([[0.5, 0.0], [0.0, 0.5]]);
        let mut ckf = Ckf::<f64, 2, 1>::new(x0, p0);

        ckf.predict(|x| *x, None).unwrap();

        // Covariance should be unchanged for identity dynamics
        approx_eq(ckf.p[(0, 0)], 0.5, 1e-6);
        approx_eq(ckf.p[(1, 1)], 0.5, 1e-6);
    }

    #[test]
    fn ckf_covariance_symmetry_and_pd() {
        let x0 = ColumnVector::from_column([0.0_f64, 0.0, 0.0]);
        let p0 = Matrix::new([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]);
        let q = Matrix::new([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]);
        let r = Matrix::new([[1.0]]);
        let mut ckf = Ckf::<f64, 3, 1>::new(x0, p0);

        for k in 0..30 {
            ckf.predict(
                |x| {
                    ColumnVector::from_column([
                        x[(0, 0)] + 0.1 * x[(1, 0)],
                        x[(1, 0)] + 0.1 * x[(2, 0)],
                        x[(2, 0)],
                    ])
                },
                Some(&q),
            )
            .unwrap();

            let z_val = (k as f64 * 0.1).sin();
            ckf.update(
                &ColumnVector::from_column([z_val]),
                |x| ColumnVector::from_column([x[(0, 0)]]),
                &r,
            )
            .unwrap();
        }

        // Symmetry
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(ckf.p[(i, j)], ckf.p[(j, i)], 1e-10);
            }
        }
        // PD
        assert!(ckf.p.cholesky().is_ok());
    }

    #[test]
    fn ckf_multiple_update_cycles() {
        let dt = 0.1;
        let x0 = ColumnVector::from_column([0.0_f64, 1.0]);
        let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
        let r = Matrix::new([[0.1]]);
        let mut ckf = Ckf::<f64, 2, 1>::new(x0, p0);

        // Multiple predict-update cycles
        for k in 0..20 {
            ckf.predict(
                |x| ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]]),
                Some(&q),
            )
            .unwrap();

            let z_val = (k + 1) as f64 * dt;
            ckf.update(
                &ColumnVector::from_column([z_val]),
                |x| ColumnVector::from_column([x[(0, 0)]]),
                &r,
            )
            .unwrap();
        }

        assert!(ckf.x[(0, 0)].is_finite());
        assert!(ckf.p[(0, 0)] > 0.0);
    }

    #[test]
    fn ckf_f32() {
        let x0 = ColumnVector::from_column([0.0_f32, 1.0]);
        let p0 = Matrix::new([[1.0_f32, 0.0], [0.0, 1.0]]);
        let q = Matrix::new([[0.01_f32, 0.0], [0.0, 0.01]]);
        let r = Matrix::new([[0.5_f32]]);
        let mut ckf = Ckf::<f32, 2, 1>::new(x0, p0);

        ckf.predict(
            |x| ColumnVector::from_column([x[(0, 0)] + 0.1 * x[(1, 0)], x[(1, 0)]]),
            Some(&q),
        )
        .unwrap();

        ckf.update(
            &ColumnVector::from_column([0.12_f32]),
            |x| ColumnVector::from_column([x[(0, 0)]]),
            &r,
        )
        .unwrap();

        assert!(ckf.x[(0, 0)].is_finite());
        assert!(ckf.p[(0, 0)] > 0.0);
    }
}

// ── RTS Smoother tests ──────────────────────────────────────────────

#[cfg(feature = "alloc")]
mod rts_tests {
    use super::*;
    extern crate alloc;
    use alloc::vec::Vec;

    fn run_forward_ekf(measurements: &[f64]) -> (Ekf<f64, 2, 1>, Vec<EkfStep<f64, 2>>) {
        let dt = 0.1;
        let x0 = ColumnVector::from_column([0.0_f64, 0.0]);
        let p0 = Matrix::new([[10.0, 0.0], [0.0, 10.0]]);
        let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
        let r = Matrix::new([[0.5]]);
        let f_jac = Matrix::new([[1.0, dt], [0.0, 1.0]]);

        let mut ekf = Ekf::<f64, 2, 1>::new(x0, p0);
        let mut steps = Vec::new();

        for &z_val in measurements {
            ekf.predict(
                |x| ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]]),
                |_| f_jac,
                Some(&q),
            );
            let x_predicted = ekf.x;
            let p_predicted = ekf.p;

            ekf.update(
                &ColumnVector::from_column([z_val]),
                |x| ColumnVector::from_column([x[(0, 0)]]),
                |_| Matrix::new([[1.0, 0.0]]),
                &r,
            )
            .unwrap();

            steps.push(EkfStep {
                x_predicted,
                p_predicted,
                x_updated: ekf.x,
                p_updated: ekf.p,
                f_jacobian: f_jac,
            });
        }

        (ekf, steps)
    }

    #[test]
    fn rts_improves_estimates() {
        // Forward EKF on linear CV, then smooth. Smoothed should have smaller errors.
        let measurements: Vec<f64> = (1..=20).map(|i| i as f64 * 0.1).collect();
        let (_, steps) = run_forward_ekf(&measurements);

        let smoothed = rts_smooth(&steps).unwrap();
        assert_eq!(smoothed.len(), steps.len());

        // Smoothed covariances should be ≤ filtered covariances (in trace)
        for (k, (_, p_smooth)) in smoothed.iter().enumerate() {
            let trace_smooth = p_smooth[(0, 0)] + p_smooth[(1, 1)];
            let trace_filter = steps[k].p_updated[(0, 0)] + steps[k].p_updated[(1, 1)];
            assert!(
                trace_smooth <= trace_filter + 1e-10,
                "step {}: smooth trace {} > filter trace {}",
                k,
                trace_smooth,
                trace_filter
            );
        }
    }

    #[test]
    fn rts_smoothed_covariance_leq_filtered() {
        let measurements: Vec<f64> = (1..=10).map(|i| i as f64 * 0.1).collect();
        let (_, steps) = run_forward_ekf(&measurements);
        let smoothed = rts_smooth(&steps).unwrap();

        // Each diagonal element of smoothed P should be ≤ filtered P
        for (k, (_, p_smooth)) in smoothed.iter().enumerate() {
            for d in 0..2 {
                assert!(
                    p_smooth[(d, d)] <= steps[k].p_updated[(d, d)] + 1e-10,
                    "step {} dim {}: smooth P {} > filter P {}",
                    k,
                    d,
                    p_smooth[(d, d)],
                    steps[k].p_updated[(d, d)]
                );
            }
        }
    }

    #[test]
    fn rts_empty_input() {
        let result = rts_smooth::<f64, 2>(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn rts_single_step() {
        let measurements = [1.0];
        let (_, steps) = run_forward_ekf(&measurements);
        let smoothed = rts_smooth(&steps).unwrap();

        assert_eq!(smoothed.len(), 1);
        // Single step: smoothed == filtered
        for i in 0..2 {
            approx_eq(smoothed[0].0[(i, 0)], steps[0].x_updated[(i, 0)], 1e-14);
            for j in 0..2 {
                approx_eq(smoothed[0].1[(i, j)], steps[0].p_updated[(i, j)], 1e-14);
            }
        }
    }

    #[test]
    fn rts_last_step_equals_filtered() {
        let measurements: Vec<f64> = (1..=5).map(|i| i as f64 * 0.1).collect();
        let (_, steps) = run_forward_ekf(&measurements);
        let smoothed = rts_smooth(&steps).unwrap();

        // Last smoothed step should exactly equal the last filtered step
        let last = smoothed.len() - 1;
        for i in 0..2 {
            approx_eq(
                smoothed[last].0[(i, 0)],
                steps[last].x_updated[(i, 0)],
                1e-14,
            );
        }
    }

    #[test]
    fn rts_backward_consistency() {
        // More measurements → earlier estimates should improve more
        let measurements: Vec<f64> = (1..=20).map(|i| i as f64 * 0.1).collect();
        let (_, steps) = run_forward_ekf(&measurements);
        let smoothed = rts_smooth(&steps).unwrap();

        // The improvement (filter_trace - smooth_trace) should generally be larger
        // for earlier steps
        let improvement_first = {
            let tf = steps[0].p_updated[(0, 0)] + steps[0].p_updated[(1, 1)];
            let ts = smoothed[0].1[(0, 0)] + smoothed[0].1[(1, 1)];
            tf - ts
        };
        let improvement_last = {
            let n = steps.len() - 1;
            let tf = steps[n].p_updated[(0, 0)] + steps[n].p_updated[(1, 1)];
            let ts = smoothed[n].1[(0, 0)] + smoothed[n].1[(1, 1)];
            tf - ts
        };

        // First step should see more improvement than last (which sees none)
        assert!(
            improvement_first > improvement_last - 1e-10,
            "first improvement {} not > last improvement {}",
            improvement_first,
            improvement_last
        );
    }

    #[test]
    fn rts_symmetry_preserved() {
        let measurements: Vec<f64> = (1..=10).map(|i| i as f64 * 0.1).collect();
        let (_, steps) = run_forward_ekf(&measurements);
        let smoothed = rts_smooth(&steps).unwrap();

        for (_, (_, p)) in smoothed.iter().enumerate() {
            for i in 0..2 {
                for j in 0..2 {
                    approx_eq(
                        p[(i, j)],
                        p[(j, i)],
                        1e-10,
                    );
                }
            }
        }
    }

    #[test]
    fn rts_f32() {
        let dt = 0.1_f32;
        let x0 = ColumnVector::from_column([0.0_f32, 0.0]);
        let p0 = Matrix::new([[10.0_f32, 0.0], [0.0, 10.0]]);
        let q = Matrix::new([[0.01_f32, 0.0], [0.0, 0.01]]);
        let r = Matrix::new([[0.5_f32]]);
        let f_jac = Matrix::new([[1.0_f32, dt], [0.0, 1.0]]);

        let mut ekf = Ekf::<f32, 2, 1>::new(x0, p0);
        let mut steps = Vec::new();

        for k in 1..=5 {
            ekf.predict(
                |x| ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]]),
                |_| f_jac,
                Some(&q),
            );
            let x_predicted = ekf.x;
            let p_predicted = ekf.p;

            ekf.update(
                &ColumnVector::from_column([k as f32 * dt]),
                |x| ColumnVector::from_column([x[(0, 0)]]),
                |_| Matrix::new([[1.0_f32, 0.0]]),
                &r,
            )
            .unwrap();

            steps.push(EkfStep {
                x_predicted,
                p_predicted,
                x_updated: ekf.x,
                p_updated: ekf.p,
                f_jacobian: f_jac,
            });
        }

        let smoothed = rts_smooth(&steps).unwrap();
        assert_eq!(smoothed.len(), 5);
        assert!(smoothed[0].0[(0, 0)].is_finite());
    }
}

// ── BatchLSQ tests ──────────────────────────────────────────────────

mod batch_tests {
    use super::*;

    #[test]
    fn batch_single_observation() {
        let mut lsq = BatchLsq::<f64, 1>::new();
        let h = Matrix::new([[1.0_f64]]);
        let r = Matrix::new([[0.1]]);
        let z = ColumnVector::from_column([3.0]);

        lsq.add_observation(&z, &h, &r).unwrap();
        let (x, p) = lsq.solve().unwrap();

        approx_eq(x[(0, 0)], 3.0, 1e-12);
        approx_eq(p[(0, 0)], 0.1, 1e-12);
    }

    #[test]
    fn batch_multiple_observations_converge() {
        let mut lsq = BatchLsq::<f64, 2>::new();
        let r = Matrix::new([[0.1]]);

        // Observe x0 = 1.0
        let h1 = Matrix::new([[1.0, 0.0_f64]]);
        lsq.add_observation(&ColumnVector::from_column([1.05]), &h1, &r)
            .unwrap();
        lsq.add_observation(&ColumnVector::from_column([0.95]), &h1, &r)
            .unwrap();
        lsq.add_observation(&ColumnVector::from_column([1.02]), &h1, &r)
            .unwrap();

        // Observe x1 = 2.0
        let h2 = Matrix::new([[0.0, 1.0_f64]]);
        lsq.add_observation(&ColumnVector::from_column([2.1]), &h2, &r)
            .unwrap();
        lsq.add_observation(&ColumnVector::from_column([1.9]), &h2, &r)
            .unwrap();
        lsq.add_observation(&ColumnVector::from_column([2.03]), &h2, &r)
            .unwrap();

        let (x, _p) = lsq.solve().unwrap();
        approx_eq(x[(0, 0)], 1.0067, 0.01);
        approx_eq(x[(1, 0)], 2.01, 0.01);
    }

    #[test]
    fn batch_with_prior() {
        let x0 = ColumnVector::from_column([0.0_f64, 0.0]);
        let p0 = Matrix::new([[100.0, 0.0], [0.0, 100.0]]); // weak prior
        let mut lsq = BatchLsq::<f64, 2>::with_prior(&x0, &p0).unwrap();

        let h = Matrix::new([[1.0, 0.0_f64]]);
        let r = Matrix::new([[0.1]]);
        lsq.add_observation(&ColumnVector::from_column([5.0]), &h, &r)
            .unwrap();

        let (x, _p) = lsq.solve().unwrap();
        // With weak prior, should be close to measurement
        approx_eq(x[(0, 0)], 5.0, 0.1);
    }

    #[test]
    fn batch_overdetermined_system() {
        // Overdetermined: 4 observations of 1 state
        let mut lsq = BatchLsq::<f64, 1>::new();
        let h = Matrix::new([[1.0_f64]]);
        let r = Matrix::new([[1.0]]);

        let obs = [10.1, 9.8, 10.3, 9.9];
        for &z in &obs {
            lsq.add_observation(&ColumnVector::from_column([z]), &h, &r)
                .unwrap();
        }

        let (x, p) = lsq.solve().unwrap();
        // Mean of observations
        let mean = obs.iter().sum::<f64>() / obs.len() as f64;
        approx_eq(x[(0, 0)], mean, 1e-10);
        // P = R/n = 1/4 = 0.25
        approx_eq(p[(0, 0)], 0.25, 1e-10);
    }

    #[test]
    fn batch_reset_works() {
        let mut lsq = BatchLsq::<f64, 2>::new();
        let h = Matrix::new([[1.0, 0.0_f64]]);
        let r = Matrix::new([[0.1]]);
        lsq.add_observation(&ColumnVector::from_column([1.0]), &h, &r)
            .unwrap();

        lsq.reset();

        // After reset, info should be zero → solve fails
        assert_eq!(lsq.solve(), Err(EstimateError::SingularInnovation));
    }

    #[test]
    fn batch_singular_r_error() {
        let mut lsq = BatchLsq::<f64, 1>::new();
        let h = Matrix::new([[1.0_f64]]);
        let r = Matrix::new([[0.0]]); // singular

        let result =
            lsq.add_observation(&ColumnVector::from_column([1.0]), &h, &r);
        assert_eq!(result, Err(EstimateError::SingularInnovation));
    }

    #[test]
    fn batch_singular_info_error() {
        // No observations → info matrix is zero → solve fails
        let lsq = BatchLsq::<f64, 2>::new();
        assert_eq!(lsq.solve(), Err(EstimateError::SingularInnovation));
    }

    #[test]
    fn batch_mixed_measurement_dimensions() {
        // Add 1D and 2D observations to a 2-state system
        let mut lsq = BatchLsq::<f64, 2>::new();

        // 1D observation: measure x0
        let h1 = Matrix::new([[1.0, 0.0_f64]]);
        let r1 = Matrix::new([[0.1]]);
        lsq.add_observation(&ColumnVector::from_column([3.0]), &h1, &r1)
            .unwrap();

        // 2D observation: measure both states
        let h2 = Matrix::new([[1.0, 0.0], [0.0, 1.0_f64]]);
        let r2 = Matrix::new([[0.1, 0.0], [0.0, 0.1]]);
        lsq.add_observation(
            &ColumnVector::from_column([3.1, 7.0]),
            &h2,
            &r2,
        )
        .unwrap();

        let (x, _p) = lsq.solve().unwrap();
        approx_eq(x[(0, 0)], 3.05, 0.05);
        approx_eq(x[(1, 0)], 7.0, 1e-10);
    }

    #[test]
    fn batch_f32() {
        let mut lsq = BatchLsq::<f32, 1>::new();
        let h = Matrix::new([[1.0_f32]]);
        let r = Matrix::new([[0.1_f32]]);
        lsq.add_observation(&ColumnVector::from_column([2.0_f32]), &h, &r)
            .unwrap();

        let (x, p) = lsq.solve().unwrap();
        assert!((x[(0, 0)] - 2.0).abs() < 1e-5);
        assert!(p[(0, 0)] > 0.0);
    }
}

// ── Cross-filter tests ──────────────────────────────────────────────

#[cfg(feature = "alloc")]
mod cross_filter_tests {
    use super::*;

    #[test]
    fn all_filters_agree_linear() {
        let dt = 0.1;
        let x0 = ColumnVector::from_column([0.0_f64, 1.0]);
        let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
        let r = Matrix::new([[0.5]]);

        let mut ekf = Ekf::<f64, 2, 1>::new(x0, p0);
        let mut ukf = Ukf::<f64, 2, 1>::with_params(x0, p0, 1.0, 2.0, 0.0);
        let mut ckf = Ckf::<f64, 2, 1>::new(x0, p0);
        let mut srukf =
            SrUkf::<f64, 2, 1>::from_covariance_with_params(x0, p0, 1.0, 2.0, 0.0).unwrap();

        let f = |x: &ColumnVector<f64, 2>| {
            ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]])
        };
        let fj = |_x: &ColumnVector<f64, 2>| Matrix::new([[1.0, dt], [0.0, 1.0]]);
        let h = |x: &ColumnVector<f64, 2>| ColumnVector::from_column([x[(0, 0)]]);
        let hj = |_x: &ColumnVector<f64, 2>| Matrix::new([[1.0, 0.0]]);

        let measurements = [0.1, 0.22, 0.31, 0.45, 0.53, 0.68, 0.79, 0.9, 1.02, 1.11];

        for &z_val in &measurements {
            ekf.predict(f, fj, Some(&q));
            ukf.predict(f, Some(&q)).unwrap();
            ckf.predict(f, Some(&q)).unwrap();
            srukf.predict(f, Some(&q)).unwrap();

            let z = ColumnVector::from_column([z_val]);
            ekf.update(&z, h, hj, &r).unwrap();
            ukf.update(&z, h, &r).unwrap();
            ckf.update(&z, h, &r).unwrap();
            srukf.update(&z, h, &r).unwrap();
        }

        // All should be close on a linear problem
        for i in 0..2 {
            let ekf_val = ekf.x[(i, 0)];
            approx_eq(ukf.x[(i, 0)], ekf_val, 0.1);
            approx_eq(ckf.x[(i, 0)], ekf_val, 0.1);
            approx_eq(srukf.x[(i, 0)], ekf_val, 0.1);
        }
    }

    #[test]
    fn ckf_vs_ukf_nonlinear() {
        // Both should handle nonlinear range measurement
        let x0 = ColumnVector::from_column([3.0_f64, 4.0]);
        let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
        let r = Matrix::new([[0.1]]);

        let mut ukf = Ukf::<f64, 2, 1>::new(x0, p0);
        let mut ckf = Ckf::<f64, 2, 1>::new(x0, p0);

        let h = |x: &ColumnVector<f64, 2>| {
            ColumnVector::from_column([
                (x[(0, 0)] * x[(0, 0)] + x[(1, 0)] * x[(1, 0)]).sqrt(),
            ])
        };

        let true_range = 5.0;
        for _ in 0..10 {
            ukf.predict(|x| *x, Some(&q)).unwrap();
            ckf.predict(|x| *x, Some(&q)).unwrap();

            let z = ColumnVector::from_column([true_range]);
            ukf.update(&z, h, &r).unwrap();
            ckf.update(&z, h, &r).unwrap();
        }

        // Both should track the range
        let ukf_range =
            (ukf.x[(0, 0)] * ukf.x[(0, 0)] + ukf.x[(1, 0)] * ukf.x[(1, 0)]).sqrt();
        let ckf_range =
            (ckf.x[(0, 0)] * ckf.x[(0, 0)] + ckf.x[(1, 0)] * ckf.x[(1, 0)]).sqrt();

        approx_eq(ukf_range, 5.0, 0.5);
        approx_eq(ckf_range, 5.0, 0.5);
    }

    #[test]
    fn srukf_vs_ukf_covariance_equivalence() {
        let dt = 0.1;
        let x0 = ColumnVector::from_column([0.0_f64, 1.0]);
        let p0 = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        let q = Matrix::new([[0.01, 0.0], [0.0, 0.01]]);
        let r = Matrix::new([[0.5]]);

        // Same params for both
        let mut ukf = Ukf::<f64, 2, 1>::with_params(x0, p0, 0.5, 2.0, 0.0);
        let mut srukf =
            SrUkf::<f64, 2, 1>::from_covariance_with_params(x0, p0, 0.5, 2.0, 0.0).unwrap();

        let f = |x: &ColumnVector<f64, 2>| {
            ColumnVector::from_column([x[(0, 0)] + dt * x[(1, 0)], x[(1, 0)]])
        };
        let h = |x: &ColumnVector<f64, 2>| ColumnVector::from_column([x[(0, 0)]]);

        for k in 0..10 {
            ukf.predict(f, Some(&q)).unwrap();
            srukf.predict(f, Some(&q)).unwrap();

            let z = ColumnVector::from_column([(k + 1) as f64 * dt]);
            ukf.update(&z, h, &r).unwrap();
            srukf.update(&z, h, &r).unwrap();
        }

        // Covariances should be close
        let srukf_p = srukf.covariance();
        for i in 0..2 {
            for j in 0..2 {
                approx_eq(ukf.p[(i, j)], srukf_p[(i, j)], 0.1);
            }
        }
    }

    #[test]
    fn batch_vs_ekf_linear() {
        // 1-state batch vs EKF. Batch sees all data at once.
        let r_val = 0.5;
        let r = Matrix::new([[r_val]]);

        let measurements = [0.1, 0.22, 0.31, 0.45, 0.53];

        // Batch: accumulate all position observations for a scalar state
        let mut lsq = BatchLsq::<f64, 1>::new();
        let h = Matrix::new([[1.0_f64]]);
        for &z_val in &measurements {
            lsq.add_observation(&ColumnVector::from_column([z_val]), &h, &r)
                .unwrap();
        }

        let (x_batch, _) = lsq.solve().unwrap();

        // Batch estimate should be the mean of measurements
        let mean = measurements.iter().sum::<f64>() / measurements.len() as f64;
        approx_eq(x_batch[(0, 0)], mean, 1e-10);
    }
}

// ── fd_jacobian tests ───────────────────────────────────────────────

#[test]
fn fd_jacobian_linear() {
    // f(x) = Ax where A = [[1, 2], [3, 4]]
    let x = ColumnVector::from_column([1.0_f64, 1.0]);
    let jac = fd_jacobian(
        &|x: &ColumnVector<f64, 2>| {
            ColumnVector::from_column([x[(0, 0)] + 2.0 * x[(1, 0)], 3.0 * x[(0, 0)] + 4.0 * x[(1, 0)]])
        },
        &x,
    );

    approx_eq(jac[(0, 0)], 1.0, 1e-6);
    approx_eq(jac[(0, 1)], 2.0, 1e-6);
    approx_eq(jac[(1, 0)], 3.0, 1e-6);
    approx_eq(jac[(1, 1)], 4.0, 1e-6);
}

#[test]
fn fd_jacobian_nonlinear() {
    // f(x) = [x0^2, x0*x1] at x = [3, 4]
    // J = [[2*x0, 0], [x1, x0]] = [[6, 0], [4, 3]]
    let x = ColumnVector::from_column([3.0_f64, 4.0]);
    let jac = fd_jacobian(
        &|x: &ColumnVector<f64, 2>| {
            ColumnVector::from_column([x[(0, 0)] * x[(0, 0)], x[(0, 0)] * x[(1, 0)]])
        },
        &x,
    );

    approx_eq(jac[(0, 0)], 6.0, 1e-5);
    approx_eq(jac[(0, 1)], 0.0, 1e-5);
    approx_eq(jac[(1, 0)], 4.0, 1e-5);
    approx_eq(jac[(1, 1)], 3.0, 1e-5);
}

#[test]
fn fd_jacobian_rectangular() {
    // f: R^2 → R^3
    let x = ColumnVector::from_column([1.0_f64, 2.0]);
    let jac = fd_jacobian(
        &|x: &ColumnVector<f64, 2>| {
            ColumnVector::from_column([x[(0, 0)], x[(1, 0)], x[(0, 0)] + x[(1, 0)]])
        },
        &x,
    );

    assert_eq!(jac.nrows(), 3);
    assert_eq!(jac.ncols(), 2);
    approx_eq(jac[(0, 0)], 1.0, 1e-6);
    approx_eq(jac[(0, 1)], 0.0, 1e-6);
    approx_eq(jac[(1, 0)], 0.0, 1e-6);
    approx_eq(jac[(1, 1)], 1.0, 1e-6);
    approx_eq(jac[(2, 0)], 1.0, 1e-6);
    approx_eq(jac[(2, 1)], 1.0, 1e-6);
}

// ── Error case tests ────────────────────────────────────────────────

#[test]
fn ekf_singular_innovation() {
    // Zero R with a zero-variance prior → S should be near-singular
    let x0 = ColumnVector::from_column([0.0_f64]);
    let p0 = Matrix::new([[0.0]]); // zero covariance
    let r = Matrix::new([[0.0]]); // zero measurement noise
    let mut ekf = Ekf::<f64, 1, 1>::new(x0, p0);

    let result = ekf.update(
        &ColumnVector::from_column([1.0]),
        |x| *x,
        |_x| Matrix::new([[1.0]]),
        &r,
    );

    assert_eq!(result, Err(EstimateError::SingularInnovation));
}

#[test]
fn estimate_error_display() {
    let e = EstimateError::CovarianceNotPD;
    assert_eq!(
        alloc::format!("{}", e),
        "covariance matrix is not positive definite"
    );

    let e = EstimateError::SingularInnovation;
    assert_eq!(
        alloc::format!("{}", e),
        "innovation covariance is singular"
    );

    let e = EstimateError::CholdowndateFailed;
    assert_eq!(
        alloc::format!("{}", e),
        "Cholesky downdate failed: result not positive definite"
    );
}
