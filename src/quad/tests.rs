use super::*;

// Helper for approximate equality
fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn approx_eq_f32(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

// =========================================================================
// Gauss-Legendre tests
// =========================================================================

#[test]
fn gl_constant() {
    // Integral of 5.0 from 0 to 3 = 15
    let r = gauss_legendre::<f64, 1>(|_| 5.0, 0.0, 3.0);
    assert!(approx_eq(r, 15.0, 1e-15));
}

#[test]
fn gl_linear_exact() {
    // Integral of 2x + 1 from 0 to 1 = 2.0 (exact for N >= 1)
    let r = gauss_legendre::<f64, 1>(|x| 2.0 * x + 1.0, 0.0, 1.0);
    assert!(approx_eq(r, 2.0, 1e-15));
}

#[test]
fn gl_quadratic_exact() {
    // Integral of x^2 from 0 to 1 = 1/3 (exact for N >= 2)
    let r = gauss_legendre::<f64, 2>(|x| x * x, 0.0, 1.0);
    assert!(approx_eq(r, 1.0 / 3.0, 1e-15));
}

#[test]
fn gl_cubic_exact() {
    // Integral of x^3 from -1 to 1 = 0 (exact for N >= 2)
    let r = gauss_legendre::<f64, 2>(|x| x * x * x, -1.0, 1.0);
    assert!(approx_eq(r, 0.0, 1e-15));
}

#[test]
fn gl_degree_9_exact() {
    // 5-point GL is exact for degree <= 9
    // Integral of x^8 from -1 to 1 = 2/9
    let r = gauss_legendre::<f64, 5>(|x| x.powi(8), -1.0, 1.0);
    assert!(approx_eq(r, 2.0 / 9.0, 1e-14));
}

#[test]
fn gl_5point_odd_function() {
    // Integral of x^9 from -1 to 1 = 0 (odd function)
    let r = gauss_legendre::<f64, 5>(|x| x.powi(9), -1.0, 1.0);
    assert!(approx_eq(r, 0.0, 1e-14));
}

#[test]
fn gl_degree_19_exact() {
    // 10-point GL is exact for degree <= 19
    // Integral of x^18 from -1 to 1 = 2/19
    let r = gauss_legendre::<f64, 10>(|x| x.powi(18), -1.0, 1.0);
    assert!(approx_eq(r, 2.0 / 19.0, 1e-13));
}

#[test]
fn gl_sin() {
    // Integral of sin(x) from 0 to pi = 2
    let r = gauss_legendre::<f64, 10>(|x| x.sin(), 0.0, core::f64::consts::PI);
    assert!(approx_eq(r, 2.0, 1e-14));
}

#[test]
fn gl_exp() {
    // Integral of e^x from 0 to 1 = e - 1
    let exact = 1.0_f64.exp() - 1.0;
    let r = gauss_legendre::<f64, 10>(|x| x.exp(), 0.0, 1.0);
    assert!(approx_eq(r, exact, 1e-14));
}

#[test]
fn gl_15point() {
    // Integral of x^28 from -1 to 1 = 2/29 (degree 28 < 2*15-1 = 29)
    let r = gauss_legendre::<f64, 15>(|x| x.powi(28), -1.0, 1.0);
    assert!(approx_eq(r, 2.0 / 29.0, 1e-12));
}

#[test]
fn gl_20point() {
    // Integral of x^38 from -1 to 1 = 2/39 (degree 38 < 2*20-1 = 39)
    let r = gauss_legendre::<f64, 20>(|x| x.powi(38), -1.0, 1.0);
    assert!(approx_eq(r, 2.0 / 39.0, 1e-10));
}

#[test]
fn gl_all_orders() {
    // Quick check that all supported orders run without panic
    let f = |x: f64| x * x;
    let _ = gauss_legendre::<f64, 1>(f, 0.0, 1.0);
    let _ = gauss_legendre::<f64, 2>(f, 0.0, 1.0);
    let _ = gauss_legendre::<f64, 3>(f, 0.0, 1.0);
    let _ = gauss_legendre::<f64, 4>(f, 0.0, 1.0);
    let _ = gauss_legendre::<f64, 5>(f, 0.0, 1.0);
    let _ = gauss_legendre::<f64, 6>(f, 0.0, 1.0);
    let _ = gauss_legendre::<f64, 7>(f, 0.0, 1.0);
    let _ = gauss_legendre::<f64, 8>(f, 0.0, 1.0);
    let _ = gauss_legendre::<f64, 9>(f, 0.0, 1.0);
    let _ = gauss_legendre::<f64, 10>(f, 0.0, 1.0);
    let _ = gauss_legendre::<f64, 15>(f, 0.0, 1.0);
    let _ = gauss_legendre::<f64, 20>(f, 0.0, 1.0);
}

#[test]
#[should_panic(expected = "unsupported N")]
fn gl_unsupported_order() {
    gauss_legendre::<f64, 11>(|x| x, 0.0, 1.0);
}

#[test]
fn gl_f32() {
    // f32 support
    let r = gauss_legendre::<f32, 5>(|x| x * x, 0.0_f32, 1.0_f32);
    assert!(approx_eq_f32(r, 1.0 / 3.0, 1e-6));
}

#[test]
fn gl_reversed_limits() {
    // a > b: integral should negate
    let r1 = gauss_legendre::<f64, 5>(|x| x * x, 0.0, 1.0);
    let r2 = gauss_legendre::<f64, 5>(|x| x * x, 1.0, 0.0);
    assert!(approx_eq(r1, -r2, 1e-15));
}

// =========================================================================
// Composite trapezoid tests
// =========================================================================

#[test]
fn trapezoid_linear_exact() {
    // Trapezoid is exact for linear functions
    let r = trapezoid(|x: f64| 3.0 * x + 2.0, 0.0, 1.0, 1);
    assert!(approx_eq(r, 3.5, 1e-15));
}

#[test]
fn trapezoid_quadratic() {
    let r = trapezoid(|x: f64| x * x, 0.0, 1.0, 10000);
    assert!(approx_eq(r, 1.0 / 3.0, 1e-7));
}

#[test]
fn trapezoid_sin() {
    let r = trapezoid(|x: f64| x.sin(), 0.0, core::f64::consts::PI, 10000);
    assert!(approx_eq(r, 2.0, 1e-7));
}

#[test]
fn trapezoid_f32() {
    let r = trapezoid(|x: f32| x * x, 0.0_f32, 1.0_f32, 1000);
    assert!(approx_eq_f32(r, 1.0 / 3.0, 1e-5));
}

#[test]
#[should_panic(expected = "n must be > 0")]
fn trapezoid_zero_n() {
    trapezoid(|x: f64| x, 0.0, 1.0, 0);
}

// =========================================================================
// Composite Simpson tests
// =========================================================================

#[test]
fn simpson_quadratic_exact() {
    // Simpson is exact for cubics (and lower)
    let r = simpson(|x: f64| x * x, 0.0, 1.0, 2);
    assert!(approx_eq(r, 1.0 / 3.0, 1e-15));
}

#[test]
fn simpson_cubic_exact() {
    // Simpson is exact for degree <= 3
    let r = simpson(|x: f64| x * x * x, 0.0, 1.0, 2);
    assert!(approx_eq(r, 0.25, 1e-15));
}

#[test]
fn simpson_exp() {
    let exact = 1.0_f64.exp() - 1.0;
    let r = simpson(|x: f64| x.exp(), 0.0, 1.0, 200);
    assert!(approx_eq(r, exact, 1e-10));
}

#[test]
fn simpson_sin() {
    let r = simpson(|x: f64| x.sin(), 0.0, core::f64::consts::PI, 200);
    assert!((r - 2.0).abs() < 1e-9, "got {}, err={}", r, (r - 2.0).abs());
}

#[test]
fn simpson_f32() {
    let r = simpson(|x: f32| x * x, 0.0_f32, 1.0_f32, 100);
    assert!(approx_eq_f32(r, 1.0 / 3.0, 1e-7));
}

#[test]
#[should_panic(expected = "n must be even")]
fn simpson_odd_n() {
    simpson(|x: f64| x, 0.0, 1.0, 3);
}

#[test]
#[should_panic(expected = "n must be even")]
fn simpson_zero_n() {
    simpson(|x: f64| x, 0.0, 1.0, 0);
}

// =========================================================================
// Adaptive Simpson tests
// =========================================================================

#[test]
fn adaptive_sin() {
    let r = adaptive_simpson(|x: f64| x.sin(), 0.0, core::f64::consts::PI, 1e-12).unwrap();
    assert!(approx_eq(r, 2.0, 1e-12));
}

#[test]
fn adaptive_exp() {
    let exact = 1.0_f64.exp() - 1.0;
    let r = adaptive_simpson(|x: f64| x.exp(), 0.0, 1.0, 1e-14).unwrap();
    assert!(approx_eq(r, exact, 1e-14));
}

#[test]
fn adaptive_ln() {
    // Integral of ln(x) from 1 to 2 = 2*ln(2) - 1
    let exact = 2.0 * 2.0_f64.ln() - 1.0;
    let r = adaptive_simpson(|x: f64| x.ln(), 1.0, 2.0, 1e-12).unwrap();
    assert!(approx_eq(r, exact, 1e-12));
}

#[test]
fn adaptive_polynomial() {
    // x^4 from 0 to 1 = 1/5
    let r = adaptive_simpson(|x: f64| x.powi(4), 0.0, 1.0, 1e-14).unwrap();
    assert!(approx_eq(r, 0.2, 1e-14));
}

#[test]
fn adaptive_one_over_x() {
    // Integral of 1/x from 1 to e = 1
    let r = adaptive_simpson(|x: f64| 1.0 / x, 1.0, core::f64::consts::E, 1e-12).unwrap();
    assert!(approx_eq(r, 1.0, 1e-12));
}

#[test]
fn adaptive_gaussian() {
    // Integral of exp(-x^2) from 0 to large = sqrt(pi)/2
    let exact = core::f64::consts::PI.sqrt() / 2.0;
    // Only integrate to 6 (practically the full half-Gaussian)
    let r = adaptive_simpson(|x: f64| (-x * x).exp(), 0.0, 6.0, 1e-10).unwrap();
    assert!(approx_eq(r, exact, 1e-10));
}

#[test]
fn adaptive_invalid_tol() {
    let r = adaptive_simpson(|x: f64| x, 0.0, 1.0, 0.0);
    assert_eq!(r, Err(QuadError::InvalidInput));

    let r = adaptive_simpson(|x: f64| x, 0.0, 1.0, -1.0);
    assert_eq!(r, Err(QuadError::InvalidInput));
}

#[test]
fn adaptive_zero_width() {
    // Integral over zero-width interval should be 0
    let r = adaptive_simpson(|x: f64| x.sin(), 1.0, 1.0, 1e-12).unwrap();
    assert!(approx_eq(r, 0.0, 1e-15));
}

#[test]
fn adaptive_f32() {
    let r = adaptive_simpson(|x: f32| x.sin(), 0.0_f32, core::f32::consts::PI, 1e-6_f32).unwrap();
    assert!(approx_eq_f32(r, 2.0, 1e-6));
}

#[test]
fn adaptive_reversed_limits() {
    // a > b should give negative result
    let r1 = adaptive_simpson(|x: f64| x * x, 0.0, 1.0, 1e-12).unwrap();
    let r2 = adaptive_simpson(|x: f64| x * x, 1.0, 0.0, 1e-12).unwrap();
    assert!(approx_eq(r1, -r2, 1e-12));
}

#[test]
fn adaptive_cos() {
    // Integral of cos(x) from 0 to pi/2 = 1
    let r = adaptive_simpson(
        |x: f64| x.cos(),
        0.0,
        core::f64::consts::FRAC_PI_2,
        1e-14,
    )
    .unwrap();
    assert!(approx_eq(r, 1.0, 1e-14));
}

// =========================================================================
// Cross-method agreement tests
// =========================================================================

#[test]
fn methods_agree_on_sin() {
    let a = 0.0_f64;
    let b = core::f64::consts::PI;
    let exact = 2.0;

    let gl = gauss_legendre::<f64, 10>(|x| x.sin(), a, b);
    let trap = trapezoid(|x: f64| x.sin(), a, b, 100000);
    let simp = simpson(|x: f64| x.sin(), a, b, 10000);
    let adapt = adaptive_simpson(|x: f64| x.sin(), a, b, 1e-12).unwrap();

    assert!(approx_eq(gl, exact, 1e-13));
    assert!(approx_eq(trap, exact, 1e-9));
    assert!(approx_eq(simp, exact, 1e-13));
    assert!(approx_eq(adapt, exact, 1e-12));
}
