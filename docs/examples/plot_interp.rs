// Interpolation comparison: sin(x) with 6 knots.
// Prints JSON with knot points and 200 evaluation points for four methods:
//   {"kx":[...], "ky":[...], "x":[...], "y_true":[...],
//    "y_linear":[...], "y_hermite":[...], "y_lagrange":[...], "y_spline":[...]}
// Used by docs/gen_plots.py to build docs/includes/plot_interp.html.

use numeris::interp::{CubicSpline, HermiteInterp, LagrangeInterp, LinearInterp};

fn fmt_arr(v: &[f64]) -> String {
    let inner: Vec<String> = v.iter().map(|x| format!("{x:.6}")).collect();
    format!("[{}]", inner.join(","))
}

fn main() {
    // 6 uniformly-spaced knots on [0, 2π]
    let tau = 2.0 * std::f64::consts::PI;
    let kx: [f64; 6] = core::array::from_fn(|i| tau * i as f64 / 5.0);
    let ky: [f64; 6] = core::array::from_fn(|i| kx[i].sin());
    let kd: [f64; 6] = core::array::from_fn(|i| kx[i].cos()); // sin' = cos

    let linear = LinearInterp::new(kx, ky).unwrap();
    let hermite = HermiteInterp::new(kx, ky, kd).unwrap();
    let lagrange = LagrangeInterp::new(kx, ky).unwrap();
    let spline = CubicSpline::new(kx, ky).unwrap();

    const N: usize = 200;
    let mut x_vals = vec![0.0_f64; N];
    let mut y_true = vec![0.0_f64; N];
    let mut y_linear = vec![0.0_f64; N];
    let mut y_hermite = vec![0.0_f64; N];
    let mut y_lagrange = vec![0.0_f64; N];
    let mut y_spline = vec![0.0_f64; N];

    for i in 0..N {
        let x = tau * i as f64 / (N - 1) as f64;
        x_vals[i] = x;
        y_true[i] = x.sin();
        y_linear[i] = linear.eval(x);
        y_hermite[i] = hermite.eval(x);
        y_lagrange[i] = lagrange.eval(x);
        y_spline[i] = spline.eval(x);
    }

    println!(
        "{{\"kx\":{},\"ky\":{},\"x\":{},\"y_true\":{},\"y_linear\":{},\"y_hermite\":{},\"y_lagrange\":{},\"y_spline\":{}}}",
        fmt_arr(&kx),
        fmt_arr(&ky),
        fmt_arr(&x_vals),
        fmt_arr(&y_true),
        fmt_arr(&y_linear),
        fmt_arr(&y_hermite),
        fmt_arr(&y_lagrange),
        fmt_arr(&y_spline)
    );
}
