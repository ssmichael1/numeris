// Harmonic oscillator solved with RKTS54 dense output.
// Prints JSON with 300 uniformly-sampled points on [0, 4π]:
//   {"t":[...], "x":[...], "v":[...]}
// Used by docs/gen_plots.py to build docs/includes/plot_ode.html.

use numeris::ode::{AdaptiveSettings, RKAdaptive, RKTS54};
use numeris::Vector;

fn fmt_arr(v: &[f64]) -> String {
    let inner: Vec<String> = v.iter().map(|x| format!("{x:.6}")).collect();
    format!("[{}]", inner.join(","))
}

fn main() {
    let tau = 4.0 * std::f64::consts::PI;
    let y0 = Vector::from_array([1.0_f64, 0.0]);

    let settings = AdaptiveSettings {
        dense_output: true,
        ..AdaptiveSettings::default()
    };

    let sol = RKTS54::integrate(
        0.0,
        tau,
        &y0,
        |_t, y| Vector::from_array([y[1], -y[0]]),
        &settings,
    )
    .expect("ODE integration failed");

    const N: usize = 300;
    let mut t_vals = vec![0.0_f64; N];
    let mut x_vals = vec![0.0_f64; N];
    let mut v_vals = vec![0.0_f64; N];

    for i in 0..N {
        let t = tau * (i as f64) / (N - 1) as f64;
        let y = RKTS54::interpolate(t, &sol).expect("interpolation failed");
        t_vals[i] = t;
        x_vals[i] = y[0];
        v_vals[i] = y[1];
    }

    println!(
        "{{\"t\":{},\"x\":{},\"v\":{}}}",
        fmt_arr(&t_vals),
        fmt_arr(&x_vals),
        fmt_arr(&v_vals)
    );
}
