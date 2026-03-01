use super::*;

// ======================== Linear ========================

#[test]
fn linear_basic() {
    let interp =
        LinearInterp::new([0.0_f64, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, 1.0]).unwrap();
    assert!((interp.eval(0.0) - 0.0).abs() < 1e-14);
    assert!((interp.eval(1.0) - 1.0).abs() < 1e-14);
    assert!((interp.eval(2.0) - 0.0).abs() < 1e-14);
    assert!((interp.eval(3.0) - 1.0).abs() < 1e-14);
    assert!((interp.eval(0.5) - 0.5).abs() < 1e-14);
    assert!((interp.eval(1.5) - 0.5).abs() < 1e-14);
    assert!((interp.eval(2.5) - 0.5).abs() < 1e-14);
}

#[test]
fn linear_derivative() {
    let interp = LinearInterp::new([0.0_f64, 1.0, 3.0], [0.0, 2.0, 2.0]).unwrap();
    let (v, d) = interp.eval_derivative(0.5);
    assert!((v - 1.0).abs() < 1e-14);
    assert!((d - 2.0).abs() < 1e-14);
    let (v2, d2) = interp.eval_derivative(2.0);
    assert!((v2 - 2.0).abs() < 1e-14);
    assert!(d2.abs() < 1e-14); // flat segment
}

#[test]
fn linear_extrapolation() {
    let interp = LinearInterp::new([1.0_f64, 2.0, 3.0], [1.0, 3.0, 2.0]).unwrap();
    // Below left boundary: extrapolates segment 0 (slope = 2)
    let v = interp.eval(0.0);
    assert!((v - (-1.0)).abs() < 1e-14);
    // Above right boundary: extrapolates last segment (slope = -1)
    let v2 = interp.eval(4.0);
    assert!((v2 - 1.0).abs() < 1e-14);
}

#[test]
fn linear_too_few_points() {
    let r = LinearInterp::new([1.0_f64], [2.0]);
    assert_eq!(r.unwrap_err(), InterpError::TooFewPoints);
}

#[test]
fn linear_not_sorted() {
    let r = LinearInterp::new([1.0_f64, 0.0, 2.0], [0.0, 0.0, 0.0]);
    assert_eq!(r.unwrap_err(), InterpError::NotSorted);
}

#[test]
fn linear_duplicate_xs() {
    let r = LinearInterp::new([0.0_f64, 1.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0]);
    assert_eq!(r.unwrap_err(), InterpError::NotSorted);
}

#[test]
fn linear_f32() {
    let interp = LinearInterp::new([0.0_f32, 1.0, 2.0], [0.0, 1.0, 4.0]).unwrap();
    assert!((interp.eval(0.5) - 0.5).abs() < 1e-6);
    assert!((interp.eval(1.5) - 2.5).abs() < 1e-6);
}

#[test]
fn linear_two_points() {
    let interp = LinearInterp::new([0.0_f64, 1.0], [0.0, 1.0]).unwrap();
    assert!((interp.eval(0.5) - 0.5).abs() < 1e-14);
}

// ======================== Hermite ========================

#[test]
fn hermite_basic() {
    // y = x, dy = 1 everywhere
    let interp = HermiteInterp::new(
        [0.0_f64, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [1.0, 1.0, 1.0],
    )
    .unwrap();
    assert!((interp.eval(0.5) - 0.5).abs() < 1e-14);
    assert!((interp.eval(1.5) - 1.5).abs() < 1e-14);
}

#[test]
fn hermite_sin() {
    // Interpolate sin(x) with exact derivatives cos(x)
    let xs = [0.0_f64, 1.0, 2.0, 3.0];
    let ys = xs.map(|x| x.sin());
    let dys = xs.map(|x| x.cos());
    let interp = HermiteInterp::new(xs, ys, dys).unwrap();

    // Check at midpoints — should be very accurate with exact derivatives
    for &x in &[0.5, 1.5, 2.5] {
        let err = (interp.eval(x) - x.sin()).abs();
        assert!(err < 5e-3, "hermite sin error at {x}: {err}");
    }
}

#[test]
fn hermite_derivative() {
    // y = x³, dy = 3x²
    let xs = [0.0_f64, 1.0, 2.0];
    let ys = [0.0, 1.0, 8.0];
    let dys = [0.0, 3.0, 12.0];
    let interp = HermiteInterp::new(xs, ys, dys).unwrap();

    // Hermite with cubic data should be exact
    let (v, d) = interp.eval_derivative(0.5);
    assert!((v - 0.125).abs() < 1e-12, "value: {v}");
    assert!((d - 0.75).abs() < 1e-12, "deriv: {d}");
}

#[test]
fn hermite_endpoints() {
    let interp = HermiteInterp::new(
        [0.0_f64, 1.0, 2.0],
        [1.0, 2.0, 0.0],
        [0.0, 0.0, 0.0],
    )
    .unwrap();
    assert!((interp.eval(0.0) - 1.0).abs() < 1e-14);
    assert!((interp.eval(2.0) - 0.0).abs() < 1e-14);
}

#[test]
fn hermite_too_few() {
    let r = HermiteInterp::new([1.0_f64], [2.0], [0.0]);
    assert_eq!(r.unwrap_err(), InterpError::TooFewPoints);
}

#[test]
fn hermite_f32() {
    let interp = HermiteInterp::new(
        [0.0_f32, 1.0, 2.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, -1.0],
    )
    .unwrap();
    assert!((interp.eval(0.5) - 0.625).abs() < 1e-6);
}

// ======================== Lagrange ========================

#[test]
fn lagrange_linear_data() {
    // Should reproduce linear function exactly
    let interp =
        LagrangeInterp::new([0.0_f64, 1.0, 2.0], [0.0, 1.0, 2.0]).unwrap();
    assert!((interp.eval(0.5) - 0.5).abs() < 1e-13);
    assert!((interp.eval(1.5) - 1.5).abs() < 1e-13);
}

#[test]
fn lagrange_quadratic() {
    // y = x² — exact through 3 points
    let interp =
        LagrangeInterp::new([0.0_f64, 1.0, 2.0], [0.0, 1.0, 4.0]).unwrap();
    assert!((interp.eval(0.5) - 0.25).abs() < 1e-12);
    assert!((interp.eval(1.5) - 2.25).abs() < 1e-12);
    assert!((interp.eval(0.75) - 0.5625).abs() < 1e-12);
}

#[test]
fn lagrange_cubic() {
    // y = x³ — exact through 4 points
    let interp = LagrangeInterp::new(
        [0.0_f64, 1.0, 2.0, 3.0],
        [0.0, 1.0, 8.0, 27.0],
    )
    .unwrap();
    assert!((interp.eval(0.5) - 0.125).abs() < 1e-11);
    assert!((interp.eval(2.5) - 15.625).abs() < 1e-10);
}

#[test]
fn lagrange_at_knots() {
    let interp = LagrangeInterp::new(
        [0.0_f64, 1.0, 2.0, 3.0],
        [5.0, 3.0, 7.0, 1.0],
    )
    .unwrap();
    for i in 0..4 {
        let x = i as f64;
        let expected = [5.0, 3.0, 7.0, 1.0][i];
        assert!(
            (interp.eval(x) - expected).abs() < 1e-12,
            "knot {i}: {} vs {expected}",
            interp.eval(x)
        );
    }
}

#[test]
fn lagrange_derivative() {
    // y = x² → dy/dx = 2x
    let interp =
        LagrangeInterp::new([0.0_f64, 1.0, 2.0], [0.0, 1.0, 4.0]).unwrap();
    let (v, d) = interp.eval_derivative(1.5);
    assert!((v - 2.25).abs() < 1e-12);
    assert!((d - 3.0).abs() < 1e-11);
}

#[test]
fn lagrange_derivative_at_knot() {
    // y = x² → dy/dx = 2x, at x=1 should be 2
    let interp =
        LagrangeInterp::new([0.0_f64, 1.0, 2.0], [0.0, 1.0, 4.0]).unwrap();
    let (v, d) = interp.eval_derivative(1.0);
    assert!((v - 1.0).abs() < 1e-12);
    assert!((d - 2.0).abs() < 1e-10);
}

#[test]
fn lagrange_too_few() {
    let r = LagrangeInterp::new([1.0_f64], [2.0]);
    assert_eq!(r.unwrap_err(), InterpError::TooFewPoints);
}

#[test]
fn lagrange_f32() {
    let interp =
        LagrangeInterp::new([0.0_f32, 1.0, 2.0], [0.0, 1.0, 4.0]).unwrap();
    assert!((interp.eval(1.5) - 2.25).abs() < 1e-5);
}

// ======================== Cubic Spline ========================

#[test]
fn spline_knot_values() {
    let xs = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
    let ys = [0.0, 1.0, 0.0, 1.0, 0.0];
    let spline = CubicSpline::new(xs, ys).unwrap();
    for i in 0..5 {
        assert!(
            (spline.eval(xs[i]) - ys[i]).abs() < 1e-13,
            "knot {i}: {} vs {}",
            spline.eval(xs[i]),
            ys[i]
        );
    }
}

#[test]
fn spline_natural_bc() {
    let xs = [0.0_f64, 1.0, 2.0, 3.0];
    let ys = [1.0, 2.0, 0.0, 3.0];
    let spline = CubicSpline::new(xs, ys).unwrap();

    // Second derivative at endpoints should be ~0 (natural BCs)
    let eps = 1e-6;
    let s0 = spline.eval(xs[0]);
    let s1 = spline.eval(xs[0] + eps);
    let s2 = spline.eval(xs[0] + 2.0 * eps);
    let second_deriv_left = (s2 - 2.0 * s1 + s0) / (eps * eps);
    assert!(
        second_deriv_left.abs() < 1e-4,
        "left BC: {second_deriv_left}"
    );

    let xn = xs[3];
    let sn0 = spline.eval(xn - 2.0 * eps);
    let sn1 = spline.eval(xn - eps);
    let sn2 = spline.eval(xn);
    let second_deriv_right = (sn2 - 2.0 * sn1 + sn0) / (eps * eps);
    assert!(
        second_deriv_right.abs() < 5e-3,
        "right BC: {second_deriv_right}"
    );
}

#[test]
fn spline_linear_reproduction() {
    // Natural cubic spline reproduces linear functions exactly
    let xs = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
    let ys = [0.0, 1.0, 2.0, 3.0, 4.0]; // y = x
    let spline = CubicSpline::new(xs, ys).unwrap();
    for &x in &[0.5, 1.5, 2.5, 3.5] {
        assert!(
            (spline.eval(x) - x).abs() < 1e-12,
            "linear repro at {x}: {}",
            spline.eval(x) - x
        );
    }
}

#[test]
fn spline_derivative() {
    let xs = [0.0_f64, 1.0, 2.0, 3.0];
    let ys = [0.0, 1.0, 0.0, 1.0];
    let spline = CubicSpline::new(xs, ys).unwrap();

    let eps = 1e-7;
    for &x in &[0.5, 1.0, 1.5, 2.0, 2.5] {
        let (_, d_analytic) = spline.eval_derivative(x);
        let fd = (spline.eval(x + eps) - spline.eval(x - eps)) / (2.0 * eps);
        assert!(
            (d_analytic - fd).abs() < 1e-5,
            "deriv at {x}: analytic={d_analytic}, fd={fd}"
        );
    }
}

#[test]
fn spline_derivative_continuity() {
    let xs = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
    let ys = [0.0, 1.0, 0.0, 2.0, 1.0];
    let spline = CubicSpline::new(xs, ys).unwrap();

    let eps = 1e-10;
    for i in 1..4 {
        let (_, d_left) = spline.eval_derivative(xs[i] - eps);
        let (_, d_right) = spline.eval_derivative(xs[i] + eps);
        assert!(
            (d_left - d_right).abs() < 1e-4,
            "deriv discontinuity at knot {i}: left={d_left}, right={d_right}"
        );
    }
}

#[test]
fn spline_extrapolation() {
    let spline =
        CubicSpline::new([0.0_f64, 1.0, 2.0], [0.0, 1.0, 0.0]).unwrap();
    let _ = spline.eval(-1.0);
    let _ = spline.eval(3.0);
}

#[test]
fn spline_too_few() {
    let r = CubicSpline::new([0.0_f64, 1.0], [0.0, 1.0]);
    assert_eq!(r.unwrap_err(), InterpError::TooFewPoints);
}

#[test]
fn spline_three_points() {
    let spline =
        CubicSpline::new([0.0_f64, 1.0, 2.0], [0.0, 1.0, 0.0]).unwrap();
    assert!((spline.eval(0.0) - 0.0).abs() < 1e-14);
    assert!((spline.eval(1.0) - 1.0).abs() < 1e-14);
    assert!((spline.eval(2.0) - 0.0).abs() < 1e-14);
}

#[test]
fn spline_f32() {
    let spline =
        CubicSpline::new([0.0_f32, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, 1.0]).unwrap();
    assert!((spline.eval(0.0) - 0.0).abs() < 1e-6);
    assert!((spline.eval(1.0) - 1.0).abs() < 1e-6);
}

// ======================== Cross-method ========================

#[test]
fn all_methods_agree_on_linear_data() {
    // y = 2x + 1 — all methods should reproduce exactly
    let xs = [0.0_f64, 1.0, 2.0, 3.0];
    let ys = [1.0, 3.0, 5.0, 7.0];

    let lin = LinearInterp::new(xs, ys).unwrap();
    let herm = HermiteInterp::new(xs, ys, [2.0; 4]).unwrap();
    let lag = LagrangeInterp::new(xs, ys).unwrap();
    let spl = CubicSpline::new(xs, ys).unwrap();

    for &x in &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
        let expected = 2.0 * x + 1.0;
        assert!((lin.eval(x) - expected).abs() < 1e-13, "linear at {x}");
        assert!((herm.eval(x) - expected).abs() < 1e-13, "hermite at {x}");
        assert!((lag.eval(x) - expected).abs() < 1e-12, "lagrange at {x}");
        assert!((spl.eval(x) - expected).abs() < 1e-12, "spline at {x}");
    }
}

// ======================== Error display ========================

#[cfg(feature = "alloc")]
#[test]
fn error_display() {
    use core::fmt::Write;
    let mut s = alloc::string::String::new();
    write!(s, "{}", InterpError::TooFewPoints).unwrap();
    assert!(s.contains("not enough"));
    s.clear();
    write!(s, "{}", InterpError::NotSorted).unwrap();
    assert!(s.contains("strictly increasing"));
    s.clear();
    write!(s, "{}", InterpError::LengthMismatch).unwrap();
    assert!(s.contains("same length"));
}

// ======================== find_interval ========================

#[test]
fn find_interval_basic() {
    let xs = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
    assert_eq!(find_interval(&xs, 0.5), 0);
    assert_eq!(find_interval(&xs, 1.5), 1);
    assert_eq!(find_interval(&xs, 2.5), 2);
    assert_eq!(find_interval(&xs, 3.5), 3);
    // At knots
    assert_eq!(find_interval(&xs, 0.0), 0);
    assert_eq!(find_interval(&xs, 1.0), 1);
    assert_eq!(find_interval(&xs, 4.0), 3); // clamped to last segment
    // Out of bounds
    assert_eq!(find_interval(&xs, -1.0), 0);
    assert_eq!(find_interval(&xs, 5.0), 3);
}

// ======================== Dynamic variants ========================

#[cfg(feature = "alloc")]
mod dyn_tests {
    use super::super::*;

    #[test]
    fn dyn_linear_basic() {
        let interp = DynLinearInterp::new(
            alloc::vec![0.0_f64, 1.0, 2.0],
            alloc::vec![0.0, 2.0, 1.0],
        )
        .unwrap();
        assert!((interp.eval(0.5) - 1.0).abs() < 1e-14);
        assert!((interp.eval(1.5) - 1.5).abs() < 1e-14);
    }

    #[test]
    fn dyn_linear_matches_fixed() {
        let xs = [0.0_f64, 1.0, 2.0, 3.0];
        let ys = [1.0, 3.0, 2.0, 5.0];
        let fixed = LinearInterp::new(xs, ys).unwrap();
        let dyn_interp =
            DynLinearInterp::new(xs.to_vec(), ys.to_vec()).unwrap();
        for &x in &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
            assert!(
                (fixed.eval(x) - dyn_interp.eval(x)).abs() < 1e-14,
                "mismatch at {x}"
            );
        }
    }

    #[test]
    fn dyn_linear_length_mismatch() {
        let r = DynLinearInterp::new(
            alloc::vec![0.0_f64, 1.0],
            alloc::vec![0.0],
        );
        assert_eq!(r.unwrap_err(), InterpError::LengthMismatch);
    }

    #[test]
    fn dyn_linear_derivative() {
        let interp = DynLinearInterp::new(
            alloc::vec![0.0_f64, 1.0, 3.0],
            alloc::vec![0.0, 2.0, 2.0],
        )
        .unwrap();
        let (v, d) = interp.eval_derivative(0.5);
        assert!((v - 1.0).abs() < 1e-14);
        assert!((d - 2.0).abs() < 1e-14);
    }

    #[test]
    fn dyn_hermite_basic() {
        let interp = DynHermiteInterp::new(
            alloc::vec![0.0_f64, 1.0, 2.0],
            alloc::vec![0.0, 1.0, 0.0],
            alloc::vec![1.0, 0.0, -1.0],
        )
        .unwrap();
        assert!((interp.eval(0.0) - 0.0).abs() < 1e-14);
        assert!((interp.eval(1.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn dyn_hermite_length_mismatch() {
        let r = DynHermiteInterp::new(
            alloc::vec![0.0_f64, 1.0],
            alloc::vec![0.0, 1.0],
            alloc::vec![0.0],
        );
        assert_eq!(r.unwrap_err(), InterpError::LengthMismatch);
    }

    #[test]
    fn dyn_hermite_derivative() {
        // y = x³, dy = 3x²
        let interp = DynHermiteInterp::new(
            alloc::vec![0.0_f64, 1.0, 2.0],
            alloc::vec![0.0, 1.0, 8.0],
            alloc::vec![0.0, 3.0, 12.0],
        )
        .unwrap();
        let (v, d) = interp.eval_derivative(0.5);
        assert!((v - 0.125).abs() < 1e-12);
        assert!((d - 0.75).abs() < 1e-12);
    }

    #[test]
    fn dyn_lagrange_basic() {
        let interp = DynLagrangeInterp::new(
            alloc::vec![0.0_f64, 1.0, 2.0],
            alloc::vec![0.0, 1.0, 4.0],
        )
        .unwrap();
        assert!((interp.eval(1.5) - 2.25).abs() < 1e-12);
    }

    #[test]
    fn dyn_lagrange_length_mismatch() {
        let r = DynLagrangeInterp::new(
            alloc::vec![0.0_f64, 1.0, 2.0],
            alloc::vec![0.0, 1.0],
        );
        assert_eq!(r.unwrap_err(), InterpError::LengthMismatch);
    }

    #[test]
    fn dyn_lagrange_derivative() {
        let interp = DynLagrangeInterp::new(
            alloc::vec![0.0_f64, 1.0, 2.0],
            alloc::vec![0.0, 1.0, 4.0],
        )
        .unwrap();
        let (v, d) = interp.eval_derivative(1.5);
        assert!((v - 2.25).abs() < 1e-12);
        assert!((d - 3.0).abs() < 1e-11);
    }

    #[test]
    fn dyn_spline_basic() {
        let spline = DynCubicSpline::new(
            alloc::vec![0.0_f64, 1.0, 2.0, 3.0],
            alloc::vec![0.0, 1.0, 0.0, 1.0],
        )
        .unwrap();
        assert!((spline.eval(0.0) - 0.0).abs() < 1e-14);
        assert!((spline.eval(1.0) - 1.0).abs() < 1e-14);
        assert!((spline.eval(2.0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn dyn_spline_matches_fixed() {
        let xs = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let ys = [0.0, 1.0, 0.0, 2.0, 1.0];
        let fixed = CubicSpline::new(xs, ys).unwrap();
        let dyn_spl =
            DynCubicSpline::new(xs.to_vec(), ys.to_vec()).unwrap();
        for &x in &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] {
            assert!(
                (fixed.eval(x) - dyn_spl.eval(x)).abs() < 1e-13,
                "spline mismatch at {x}"
            );
        }
    }

    #[test]
    fn dyn_spline_length_mismatch() {
        let r = DynCubicSpline::new(
            alloc::vec![0.0_f64, 1.0, 2.0],
            alloc::vec![0.0, 1.0],
        );
        assert_eq!(r.unwrap_err(), InterpError::LengthMismatch);
    }

    #[test]
    fn dyn_spline_derivative() {
        let spline = DynCubicSpline::new(
            alloc::vec![0.0_f64, 1.0, 2.0, 3.0],
            alloc::vec![0.0, 1.0, 0.0, 1.0],
        )
        .unwrap();
        let eps = 1e-7;
        for &x in &[0.5, 1.5, 2.5] {
            let (_, d_analytic) = spline.eval_derivative(x);
            let fd = (spline.eval(x + eps) - spline.eval(x - eps)) / (2.0 * eps);
            assert!(
                (d_analytic - fd).abs() < 1e-5,
                "dyn spline deriv at {x}: {d_analytic} vs {fd}"
            );
        }
    }
}
