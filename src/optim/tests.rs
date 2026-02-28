use super::*;
use crate::{Matrix, Vector};

const TOL: f64 = 1e-8;
const LOOSE_TOL: f64 = 1e-4;

fn assert_near(a: f64, b: f64, tol: f64, msg: &str) {
    assert!(
        (a - b).abs() < tol,
        "{}: {} vs {} (diff {})",
        msg,
        a,
        b,
        (a - b).abs()
    );
}

// ═══════════════════════════════════════════════════════════════════
// Root finding
// ═══════════════════════════════════════════════════════════════════

#[test]
fn brent_sqrt2() {
    let r = brent(|x| x * x - 2.0, 0.0, 2.0, &RootSettings::default()).unwrap();
    assert_near(r.x, core::f64::consts::SQRT_2, 1e-12, "brent √2");
    assert_near(r.fx, 0.0, 1e-11, "brent f(√2)");
}

#[test]
fn brent_negative_bracket() {
    // f(x) = x^3 - x - 2, root near 1.5214
    let r = brent(|x: f64| x * x * x - x - 2.0, 1.0, 2.0, &RootSettings::default()).unwrap();
    assert!(r.fx.abs() < 1e-10, "brent cubic");
}

#[test]
fn brent_invalid_bracket() {
    // Both endpoints positive
    let r = brent(|x| x * x + 1.0, 0.0, 2.0, &RootSettings::default());
    assert_eq!(r.unwrap_err(), OptimError::BracketInvalid);
}

#[test]
fn brent_f32() {
    let settings = RootSettings::<f32>::default();
    let r = brent(|x: f32| x * x - 2.0, 0.0f32, 2.0f32, &settings).unwrap();
    assert!((r.x - core::f32::consts::SQRT_2).abs() < 1e-5, "brent f32");
}

#[test]
fn brent_sin() {
    // Root of sin(x) near π
    let r = brent(
        |x: f64| x.sin(),
        3.0,
        4.0,
        &RootSettings::default(),
    )
    .unwrap();
    assert_near(r.x, core::f64::consts::PI, 1e-12, "brent sin root");
}

#[test]
fn newton_1d_sqrt2() {
    let r = newton_1d(
        |x| x * x - 2.0,
        |x| 2.0 * x,
        1.0,
        &RootSettings::default(),
    )
    .unwrap();
    assert_near(r.x, core::f64::consts::SQRT_2, 1e-12, "newton √2");
}

#[test]
fn newton_1d_cubic() {
    // f(x) = x^3 - x - 2
    let r = newton_1d(
        |x: f64| x * x * x - x - 2.0,
        |x: f64| 3.0 * x * x - 1.0,
        1.5,
        &RootSettings::default(),
    )
    .unwrap();
    assert!(r.fx.abs() < 1e-10, "newton cubic");
}

#[test]
fn newton_1d_singular_derivative() {
    // f(x) = x^2 at x = 0, f'(0) = 0
    let r = newton_1d(|x| x * x, |x| 2.0 * x, 0.0, &RootSettings::default());
    // Should converge to root x=0 since f(0)=0 < f_tol
    assert!(r.is_ok());
}

#[test]
fn newton_1d_exponential() {
    // f(x) = e^x - 3, root = ln(3)
    let r = newton_1d(
        |x: f64| x.exp() - 3.0,
        |x: f64| x.exp(),
        1.0,
        &RootSettings::default(),
    )
    .unwrap();
    assert_near(r.x, 3.0_f64.ln(), 1e-12, "newton exp root");
}

// ═══════════════════════════════════════════════════════════════════
// Finite differences
// ═══════════════════════════════════════════════════════════════════

#[test]
fn finite_diff_gradient_quadratic() {
    // f(x) = x0^2 + 3*x1^2, grad = [2*x0, 6*x1]
    let x = Vector::from_array([2.0_f64, 3.0]);
    let g = finite_difference_gradient(
        |x: &Vector<f64, 2>| x[0] * x[0] + 3.0 * x[1] * x[1],
        &x,
    );
    assert_near(g[0], 4.0, 1e-5, "grad[0]");
    assert_near(g[1], 18.0, 1e-5, "grad[1]");
}

#[test]
fn finite_diff_jacobian_linear() {
    // f(x) = A*x, Jacobian = A
    let a = Matrix::new([[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let x = Vector::from_array([1.0, 1.0]);
    let j = finite_difference_jacobian(|x: &Vector<f64, 2>| a.vecmul(x), &x);
    for i in 0..3 {
        for k in 0..2 {
            assert_near(j[(i, k)], a[(i, k)], 1e-5, "jacobian");
        }
    }
}

#[test]
fn finite_diff_jacobian_nonlinear() {
    // f(x) = [x0*x1, x0^2 + x1], Jacobian = [[x1, x0], [2*x0, 1]]
    let x = Vector::from_array([3.0_f64, 4.0]);
    let j = finite_difference_jacobian(
        |x: &Vector<f64, 2>| Vector::from_array([x[0] * x[1], x[0] * x[0] + x[1]]),
        &x,
    );
    assert_near(j[(0, 0)], 4.0, 1e-5, "J[0,0]=x1");
    assert_near(j[(0, 1)], 3.0, 1e-5, "J[0,1]=x0");
    assert_near(j[(1, 0)], 6.0, 1e-5, "J[1,0]=2x0");
    assert_near(j[(1, 1)], 1.0, 1e-5, "J[1,1]=1");
}

// ═══════════════════════════════════════════════════════════════════
// BFGS
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bfgs_simple_quadratic() {
    // f(x) = (x0-1)^2 + (x1-2)^2, minimum at (1, 2)
    let r = minimize_bfgs(
        |x: &Vector<f64, 2>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2),
        |x: &Vector<f64, 2>| Vector::from_array([2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)]),
        &Vector::from_array([0.0, 0.0]),
        &BfgsSettings::default(),
    )
    .unwrap();

    assert_near(r.x[0], 1.0, TOL, "bfgs quad x0");
    assert_near(r.x[1], 2.0, TOL, "bfgs quad x1");
    assert_near(r.fx, 0.0, TOL, "bfgs quad f");
    // Quadratic should converge in very few iterations
    assert!(r.iterations <= 3, "bfgs quad iters = {}", r.iterations);
}

#[test]
fn bfgs_rosenbrock() {
    // Rosenbrock: f(x) = (1-x0)^2 + 100*(x1 - x0^2)^2
    let r = minimize_bfgs(
        |x: &Vector<f64, 2>| {
            (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
        },
        |x: &Vector<f64, 2>| {
            Vector::from_array([
                -2.0 * (1.0 - x[0]) + 200.0 * (x[1] - x[0] * x[0]) * (-2.0 * x[0]),
                200.0 * (x[1] - x[0] * x[0]),
            ])
        },
        &Vector::from_array([-1.0, 1.0]),
        &BfgsSettings {
            max_iter: 500,
            ..BfgsSettings::default()
        },
    )
    .unwrap();

    assert_near(r.x[0], 1.0, 1e-4, "bfgs rosenbrock x0");
    assert_near(r.x[1], 1.0, 1e-4, "bfgs rosenbrock x1");
}

#[test]
fn bfgs_3d() {
    // f(x) = x0^2 + 2*x1^2 + 3*x2^2 - 2*x0 - 4*x1 - 6*x2
    // minimum at (1, 1, 1), f_min = -6
    let r = minimize_bfgs(
        |x: &Vector<f64, 3>| {
            x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[2] * x[2]
                - 2.0 * x[0] - 4.0 * x[1] - 6.0 * x[2]
        },
        |x: &Vector<f64, 3>| {
            Vector::from_array([
                2.0 * x[0] - 2.0,
                4.0 * x[1] - 4.0,
                6.0 * x[2] - 6.0,
            ])
        },
        &Vector::from_array([0.0, 0.0, 0.0]),
        &BfgsSettings::default(),
    )
    .unwrap();

    assert_near(r.x[0], 1.0, TOL, "bfgs 3d x0");
    assert_near(r.x[1], 1.0, TOL, "bfgs 3d x1");
    assert_near(r.x[2], 1.0, TOL, "bfgs 3d x2");
    assert_near(r.fx, -6.0, TOL, "bfgs 3d f");
}

// ═══════════════════════════════════════════════════════════════════
// Gauss-Newton
// ═══════════════════════════════════════════════════════════════════

#[test]
fn gn_linear_least_squares() {
    // Fit y = c0 + c1*x to (0,1),(1,2),(2,4)
    // A = [[1,0],[1,1],[1,2]], b = [1,2,4]
    let a = Matrix::new([[1.0_f64, 0.0], [1.0, 1.0], [1.0, 2.0]]);
    let b = Vector::from_array([1.0, 2.0, 4.0]);

    let r = least_squares_gn(
        |x: &Vector<f64, 2>| a.vecmul(x) - b,
        |_: &Vector<f64, 2>| a,
        &Vector::from_array([0.0, 0.0]),
        &GaussNewtonSettings::default(),
    )
    .unwrap();

    assert_near(r.x[0], 5.0 / 6.0, TOL, "gn linear c0");
    assert_near(r.x[1], 3.0 / 2.0, TOL, "gn linear c1");
    assert!(r.iterations <= 2, "gn linear iters = {}", r.iterations);
}

#[test]
fn gn_exponential_fit() {
    // Fit y = a * exp(b * x) to data
    let t = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
    let y_data = [2.0, 2.7, 3.65, 4.95, 6.7];

    let r = least_squares_gn(
        |x: &Vector<f64, 2>| {
            let mut r = Vector::<f64, 5>::zeros();
            for i in 0..5 {
                r[i] = x[0] * (x[1] * t[i]).exp() - y_data[i];
            }
            r
        },
        |x: &Vector<f64, 2>| {
            let mut j = Matrix::<f64, 5, 2>::zeros();
            for i in 0..5 {
                let e = (x[1] * t[i]).exp();
                j[(i, 0)] = e;
                j[(i, 1)] = x[0] * t[i] * e;
            }
            j
        },
        &Vector::from_array([1.0, 0.1]),
        &GaussNewtonSettings::default(),
    )
    .unwrap();

    assert!(r.cost < 0.1, "gn exp cost = {}", r.cost);
    assert!(r.x[0] > 1.5 && r.x[0] < 2.5, "gn exp a = {}", r.x[0]);
    assert!(r.x[1] > 0.2 && r.x[1] < 0.4, "gn exp b = {}", r.x[1]);
}

// ═══════════════════════════════════════════════════════════════════
// Levenberg-Marquardt
// ═══════════════════════════════════════════════════════════════════

#[test]
fn lm_exponential_fit() {
    let t = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
    let y_data = [2.0, 2.7, 3.65, 4.95, 6.7];

    let r = least_squares_lm(
        |x: &Vector<f64, 2>| {
            let mut r = Vector::<f64, 5>::zeros();
            for i in 0..5 {
                r[i] = x[0] * (x[1] * t[i]).exp() - y_data[i];
            }
            r
        },
        |x: &Vector<f64, 2>| {
            let mut j = Matrix::<f64, 5, 2>::zeros();
            for i in 0..5 {
                let e = (x[1] * t[i]).exp();
                j[(i, 0)] = e;
                j[(i, 1)] = x[0] * t[i] * e;
            }
            j
        },
        &Vector::from_array([1.0, 0.1]),
        &LmSettings::default(),
    )
    .unwrap();

    assert!(r.cost < 0.1, "lm exp cost = {}", r.cost);
}

#[test]
fn lm_circle_fit() {
    // Fit a circle (cx, cy, r) to noisy points on a unit circle
    // Points on circle of radius 2 centered at (1, 1)
    let angles = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    let mut px = [0.0_f64; 6];
    let mut py = [0.0_f64; 6];
    for i in 0..6 {
        px[i] = 1.0 + 2.0 * angles[i].cos();
        py[i] = 1.0 + 2.0 * angles[i].sin();
    }

    let r = least_squares_lm(
        |x: &Vector<f64, 3>| {
            let mut r = Vector::<f64, 6>::zeros();
            for i in 0..6 {
                let dx = px[i] - x[0];
                let dy = py[i] - x[1];
                r[i] = (dx * dx + dy * dy).sqrt() - x[2];
            }
            r
        },
        |x: &Vector<f64, 3>| {
            let mut j = Matrix::<f64, 6, 3>::zeros();
            for i in 0..6 {
                let dx = px[i] - x[0];
                let dy = py[i] - x[1];
                let d = (dx * dx + dy * dy).sqrt();
                j[(i, 0)] = -dx / d;
                j[(i, 1)] = -dy / d;
                j[(i, 2)] = -1.0;
            }
            j
        },
        &Vector::from_array([0.0, 0.0, 1.0]),
        &LmSettings::default(),
    )
    .unwrap();

    assert_near(r.x[0], 1.0, LOOSE_TOL, "lm circle cx");
    assert_near(r.x[1], 1.0, LOOSE_TOL, "lm circle cy");
    assert_near(r.x[2], 2.0, LOOSE_TOL, "lm circle r");
}

#[test]
fn lm_with_numerical_jacobian() {
    // Same exponential fit but using finite_difference_jacobian
    let t = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
    let y_data = [2.0, 2.7, 3.65, 4.95, 6.7];

    let residual = |x: &Vector<f64, 2>| -> Vector<f64, 5> {
        let mut r = Vector::<f64, 5>::zeros();
        for i in 0..5 {
            r[i] = x[0] * (x[1] * t[i]).exp() - y_data[i];
        }
        r
    };

    let r = least_squares_lm(
        residual,
        |x: &Vector<f64, 2>| finite_difference_jacobian(residual, x),
        &Vector::from_array([1.0, 0.1]),
        &LmSettings::default(),
    )
    .unwrap();

    assert!(r.cost < 0.1, "lm numerical jac cost = {}", r.cost);
}

#[test]
fn lm_linear_exact() {
    // Linear problem: should converge in very few iterations
    let a = Matrix::new([[1.0_f64, 0.0], [1.0, 1.0], [1.0, 2.0]]);
    let b = Vector::from_array([1.0, 2.0, 4.0]);

    let r = least_squares_lm(
        |x: &Vector<f64, 2>| a.vecmul(x) - b,
        |_: &Vector<f64, 2>| a,
        &Vector::from_array([0.0, 0.0]),
        &LmSettings::default(),
    )
    .unwrap();

    assert_near(r.x[0], 5.0 / 6.0, 1e-6, "lm linear c0");
    assert_near(r.x[1], 3.0 / 2.0, 1e-6, "lm linear c1");
}

// ═══════════════════════════════════════════════════════════════════
// Error type tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn error_display() {
    let e = OptimError::MaxIterations;
    assert_eq!(format!("{}", e), "maximum iterations exceeded");
    let e = OptimError::BracketInvalid;
    assert_eq!(format!("{}", e), "bracket endpoints must have opposite signs");
}

// ═══════════════════════════════════════════════════════════════════
// Settings defaults
// ═══════════════════════════════════════════════════════════════════

#[test]
fn settings_defaults() {
    let rs: RootSettings<f64> = RootSettings::default();
    assert_eq!(rs.max_iter, 100);

    let bs: BfgsSettings<f64> = BfgsSettings::default();
    assert_eq!(bs.max_iter, 200);

    let gs: GaussNewtonSettings<f64> = GaussNewtonSettings::default();
    assert_eq!(gs.max_iter, 100);

    let ls: LmSettings<f64> = LmSettings::default();
    assert_eq!(ls.max_iter, 100);
}
