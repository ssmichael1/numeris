//! Generate Plotly HTML snippets for MkDocs documentation.
//!
//! Replaces the separate plot_ode / plot_interp examples and the gen_plots.py
//! Python script. Run with:
//!
//! ```sh
//! cargo run --example gen_plots --features all
//! ```
//!
//! Writes interactive Plotly HTML snippets to `docs/includes/`.

use numeris::control::{butterworth_lowpass, lag_compensator, lead_compensator, BiquadCascade};
use numeris::interp::{CubicSpline, HermiteInterp, LagrangeInterp, LinearInterp};
use numeris::ode::{AdaptiveSettings, RKAdaptive, Rosenbrock, RKTS54, RODAS4};
use numeris::stats::{
    Beta, Binomial, ContinuousDistribution, DiscreteDistribution, Gamma, Normal, Poisson,
};
use numeris::Vector;

use std::f64::consts::PI;
use std::fs;
use std::path::Path;

// ─── helpers ──────────────────────────────────────────────────────────────

fn fmt_arr(v: &[f64]) -> String {
    let inner: Vec<String> = v.iter().map(|x| format!("{x:.6}")).collect();
    format!("[{}]", inner.join(","))
}

/// Shared axis and font styling applied to every plot layout.
///
/// Takes a partial layout JSON object (must start with `{` and end with `}`)
/// and injects the common decoration keys. The caller provides title, axis
/// titles, and any plot-specific overrides; this function adds:
///
/// - Serif font family at 15 px for body, 17 px for axis titles, 18 px bold
///   for the plot title
/// - Axis lines + mirror + ticks on all four sides
/// - Light gridlines with dash pattern
/// - Consistent legend and margin defaults (caller can override)
fn decorate_layout(
    title: &str,
    x_title: &str,
    y_title: &str,
    extra: &str, // additional JSON key-value pairs (with leading comma)
) -> String {
    // We build the JSON manually to avoid pulling in serde.
    format!(
        concat!(
            "{{",
            "\"title\":{{\"text\":\"{title}\",\"font\":{{\"family\":\"Georgia, 'Times New Roman', serif\",\"size\":18}}}},",
            "\"font\":{{\"family\":\"Georgia, 'Times New Roman', serif\",\"size\":15}},",
            "\"xaxis\":{{",
            "\"title\":{{\"text\":\"{x_title}\",\"font\":{{\"size\":16}}}},",
            "\"showline\":true,\"linewidth\":1,\"linecolor\":\"black\",",
            "\"mirror\":true,",
            "\"ticks\":\"outside\",\"ticklen\":5,\"tickwidth\":1,\"tickcolor\":\"black\",",
            "\"showgrid\":true,\"gridwidth\":1,\"gridcolor\":\"rgba(180,180,180,0.35)\",\"griddash\":\"dot\"",
            "{x_extra}",
            "}},",
            "\"yaxis\":{{",
            "\"title\":{{\"text\":\"{y_title}\",\"font\":{{\"size\":16}}}},",
            "\"showline\":true,\"linewidth\":1,\"linecolor\":\"black\",",
            "\"mirror\":true,",
            "\"ticks\":\"outside\",\"ticklen\":5,\"tickwidth\":1,\"tickcolor\":\"black\",",
            "\"showgrid\":true,\"gridwidth\":1,\"gridcolor\":\"rgba(180,180,180,0.35)\",\"griddash\":\"dot\"",
            "{y_extra}",
            "}},",
            "\"legend\":{{\"orientation\":\"h\",\"y\":-0.22,\"font\":{{\"size\":14}}}},",
            "\"margin\":{{\"t\":55,\"b\":70,\"l\":70,\"r\":30}},",
            "\"plot_bgcolor\":\"white\",\"paper_bgcolor\":\"white\"",
            "{extra}",
            "}}",
        ),
        title = title,
        x_title = x_title,
        y_title = y_title,
        x_extra = "",
        y_extra = "",
        extra = extra,
    )
}

/// Like `decorate_layout` but with per-axis extra JSON fragments.
fn decorate_layout_ex(
    title: &str,
    x_title: &str,
    x_extra: &str,
    y_title: &str,
    y_extra: &str,
    extra: &str,
) -> String {
    format!(
        concat!(
            "{{",
            "\"title\":{{\"text\":\"{title}\",\"font\":{{\"family\":\"Georgia, 'Times New Roman', serif\",\"size\":18}}}},",
            "\"font\":{{\"family\":\"Georgia, 'Times New Roman', serif\",\"size\":15}},",
            "\"xaxis\":{{",
            "\"title\":{{\"text\":\"{x_title}\",\"font\":{{\"size\":16}}}},",
            "\"showline\":true,\"linewidth\":1,\"linecolor\":\"black\",",
            "\"mirror\":true,",
            "\"ticks\":\"outside\",\"ticklen\":5,\"tickwidth\":1,\"tickcolor\":\"black\",",
            "\"showgrid\":true,\"gridwidth\":1,\"gridcolor\":\"rgba(180,180,180,0.35)\",\"griddash\":\"dot\"",
            "{x_extra}",
            "}},",
            "\"yaxis\":{{",
            "\"title\":{{\"text\":\"{y_title}\",\"font\":{{\"size\":16}}}},",
            "\"showline\":true,\"linewidth\":1,\"linecolor\":\"black\",",
            "\"mirror\":true,",
            "\"ticks\":\"outside\",\"ticklen\":5,\"tickwidth\":1,\"tickcolor\":\"black\",",
            "\"showgrid\":true,\"gridwidth\":1,\"gridcolor\":\"rgba(180,180,180,0.35)\",\"griddash\":\"dot\"",
            "{y_extra}",
            "}},",
            "\"legend\":{{\"orientation\":\"h\",\"y\":-0.22,\"font\":{{\"size\":14}}}},",
            "\"margin\":{{\"t\":55,\"b\":70,\"l\":70,\"r\":30}},",
            "\"plot_bgcolor\":\"white\",\"paper_bgcolor\":\"white\"",
            "{extra}",
            "}}",
        ),
        title = title,
        x_title = x_title,
        x_extra = x_extra,
        y_title = y_title,
        y_extra = y_extra,
        extra = extra,
    )
}

fn plotly_snippet(div_id: &str, traces_json: &str, layout_json: &str, height: u32) -> String {
    format!(
        "<div id=\"{div_id}\" style=\"width:100%;height:{height}px;\"></div>\n\
         <script>\n\
         window.addEventListener('load',function(){{Plotly.newPlot('{div_id}',{traces_json},{layout_json},{{responsive:true}})}});\n\
         </script>\n"
    )
}

fn write_snippet(dir: &Path, name: &str, html: &str) {
    let path = dir.join(format!("{name}.html"));
    fs::write(&path, html).unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
    println!("  → {}", path.display());
}

// ─── ODE plot ─────────────────────────────────────────────────────────────

fn make_ode_plot() -> String {
    let tau = 4.0 * PI;
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
    let mut t = vec![0.0; N];
    let mut x = vec![0.0; N];
    let mut v = vec![0.0; N];
    for i in 0..N {
        let ti = tau * i as f64 / (N - 1) as f64;
        let yi = RKTS54::interpolate(ti, &sol).unwrap();
        t[i] = ti;
        x[i] = yi[0];
        v[i] = yi[1];
    }

    let traces = format!(
        "[{{\"type\":\"scatter\",\"mode\":\"lines\",\"name\":\"position x(t)\",\
          \"x\":{},\"y\":{},\"line\":{{\"width\":2.5}}}},\
         {{\"type\":\"scatter\",\"mode\":\"lines\",\"name\":\"velocity v(t)\",\
          \"x\":{},\"y\":{},\"line\":{{\"width\":2.5,\"dash\":\"dash\"}}}}]",
        fmt_arr(&t),
        fmt_arr(&x),
        fmt_arr(&t),
        fmt_arr(&v),
    );

    let layout = decorate_layout(
        "Harmonic Oscillator — RKTS54 Dense Output",
        "t",
        "Value",
        "",
    );

    plotly_snippet("plot-ode", &traces, &layout, 420)
}

// ─── Interpolation plot ───────────────────────────────────────────────────

fn make_interp_plot() -> String {
    let tau = 2.0 * PI;
    let kx: [f64; 6] = core::array::from_fn(|i| tau * i as f64 / 5.0);
    let ky: [f64; 6] = core::array::from_fn(|i| kx[i].sin());
    let kd: [f64; 6] = core::array::from_fn(|i| kx[i].cos());

    let linear = LinearInterp::new(kx, ky).unwrap();
    let hermite = HermiteInterp::new(kx, ky, kd).unwrap();
    let lagrange = LagrangeInterp::new(kx, ky).unwrap();
    let spline = CubicSpline::new(kx, ky).unwrap();

    const N: usize = 200;
    let mut xv = vec![0.0; N];
    let mut y_true = vec![0.0; N];
    let mut y_lin = vec![0.0; N];
    let mut y_her = vec![0.0; N];
    let mut y_lag = vec![0.0; N];
    let mut y_spl = vec![0.0; N];
    for i in 0..N {
        let xi = tau * i as f64 / (N - 1) as f64;
        xv[i] = xi;
        y_true[i] = xi.sin();
        y_lin[i] = linear.eval(xi);
        y_her[i] = hermite.eval(xi);
        y_lag[i] = lagrange.eval(xi);
        y_spl[i] = spline.eval(xi);
    }

    let traces = format!(
        "[{{\"type\":\"scatter\",\"mode\":\"lines\",\"name\":\"sin(x) exact\",\
          \"x\":{},\"y\":{},\"line\":{{\"width\":2.5,\"color\":\"rgba(120,120,120,0.6)\",\"dash\":\"dot\"}}}},\
         {{\"type\":\"scatter\",\"mode\":\"lines\",\"name\":\"Linear\",\
          \"x\":{},\"y\":{},\"line\":{{\"width\":2}}}},\
         {{\"type\":\"scatter\",\"mode\":\"lines\",\"name\":\"Hermite\",\
          \"x\":{},\"y\":{},\"line\":{{\"width\":2}}}},\
         {{\"type\":\"scatter\",\"mode\":\"lines\",\"name\":\"Lagrange\",\
          \"x\":{},\"y\":{},\"line\":{{\"width\":2}}}},\
         {{\"type\":\"scatter\",\"mode\":\"lines\",\"name\":\"Cubic Spline\",\
          \"x\":{},\"y\":{},\"line\":{{\"width\":2}}}},\
         {{\"type\":\"scatter\",\"mode\":\"markers\",\"name\":\"knots\",\
          \"x\":{},\"y\":{},\"marker\":{{\"size\":9,\"color\":\"black\",\"symbol\":\"diamond\"}}}}]",
        fmt_arr(&xv), fmt_arr(&y_true),
        fmt_arr(&xv), fmt_arr(&y_lin),
        fmt_arr(&xv), fmt_arr(&y_her),
        fmt_arr(&xv), fmt_arr(&y_lag),
        fmt_arr(&xv), fmt_arr(&y_spl),
        fmt_arr(&kx), fmt_arr(&ky),
    );

    let layout = decorate_layout(
        "Interpolation Methods on sin(x) — 6 Knots",
        "x",
        "y",
        "",
    );

    plotly_snippet("plot-interp", &traces, &layout, 440)
}

// ─── Control: Butterworth frequency response ──────────────────────────────

fn biquad_cascade_freq_response<const N: usize>(
    cascade: &BiquadCascade<f64, N>,
    freq: f64,
    fs: f64,
) -> f64 {
    let omega = 2.0 * PI * freq / fs;
    let (sin_w, cos_w) = omega.sin_cos();
    let cos_2w = 2.0 * cos_w * cos_w - 1.0;
    let sin_2w = 2.0 * sin_w * cos_w;

    let mut mag_sq = 1.0;
    for section in &cascade.sections {
        let (b, a) = section.coefficients();
        let nr = b[0] + b[1] * cos_w + b[2] * cos_2w;
        let ni = -b[1] * sin_w - b[2] * sin_2w;
        let dr = a[0] + a[1] * cos_w + a[2] * cos_2w;
        let di = -a[1] * sin_w - a[2] * sin_2w;
        mag_sq *= (nr * nr + ni * ni) / (dr * dr + di * di);
    }
    mag_sq.sqrt()
}

fn make_control_plot() -> String {
    let fs = 8000.0;
    let fc = 1000.0;

    let bw2: BiquadCascade<f64, 1> = butterworth_lowpass(2, fc, fs).unwrap();
    let bw4: BiquadCascade<f64, 2> = butterworth_lowpass(4, fc, fs).unwrap();
    let bw6: BiquadCascade<f64, 3> = butterworth_lowpass(6, fc, fs).unwrap();

    const N: usize = 500;
    let mut freqs = vec![0.0; N];
    let mut db2 = vec![0.0; N];
    let mut db4 = vec![0.0; N];
    let mut db6 = vec![0.0; N];

    let f_min: f64 = 10.0;
    let f_max: f64 = 3900.0;
    for i in 0..N {
        let f = f_min * (f_max / f_min).powf(i as f64 / (N - 1) as f64);
        freqs[i] = f;
        db2[i] = 20.0 * biquad_cascade_freq_response(&bw2, f, fs).log10();
        db4[i] = 20.0 * biquad_cascade_freq_response(&bw4, f, fs).log10();
        db6[i] = 20.0 * biquad_cascade_freq_response(&bw6, f, fs).log10();
    }

    let traces = format!(
        "[{{\"type\":\"scatter\",\"mode\":\"lines\",\"name\":\"2nd order\",\
          \"x\":{},\"y\":{},\"line\":{{\"width\":2.5}}}},\
         {{\"type\":\"scatter\",\"mode\":\"lines\",\"name\":\"4th order\",\
          \"x\":{},\"y\":{},\"line\":{{\"width\":2.5}}}},\
         {{\"type\":\"scatter\",\"mode\":\"lines\",\"name\":\"6th order\",\
          \"x\":{},\"y\":{},\"line\":{{\"width\":2.5}}}}]",
        fmt_arr(&freqs), fmt_arr(&db2),
        fmt_arr(&freqs), fmt_arr(&db4),
        fmt_arr(&freqs), fmt_arr(&db6),
    );

    let layout = decorate_layout_ex(
        "Butterworth Lowpass — f<sub>c</sub> = 1 kHz, f<sub>s</sub> = 8 kHz",
        "Frequency (Hz)",
        ",\"type\":\"log\"",
        "Magnitude (dB)",
        ",\"range\":[-80,5]",
        &format!(
            ",\"shapes\":[{{\"type\":\"line\",\"x0\":{f_min},\"x1\":{f_max},\
             \"y0\":-3,\"y1\":-3,\"line\":{{\"dash\":\"dot\",\"color\":\"rgba(160,80,80,0.5)\",\"width\":1.5}}}}]"
        ),
    );

    plotly_snippet("plot-control", &traces, &layout, 420)
}

// ─── Control: Lead/Lag compensator Bode ───────────────────────────────────

fn biquad_freq_response(b: &[f64; 3], a: &[f64; 3], freq: f64, fs: f64) -> (f64, f64) {
    let omega = 2.0 * PI * freq / fs;
    let (sin_w, cos_w) = omega.sin_cos();
    let cos_2w = 2.0 * cos_w * cos_w - 1.0;
    let sin_2w = 2.0 * sin_w * cos_w;
    let nr = b[0] + b[1] * cos_w + b[2] * cos_2w;
    let ni = -b[1] * sin_w - b[2] * sin_2w;
    let dr = a[0] + a[1] * cos_w + a[2] * cos_2w;
    let di = -a[1] * sin_w - a[2] * sin_2w;
    let mag = ((nr * nr + ni * ni) / (dr * dr + di * di)).sqrt();
    let phase = (ni.atan2(nr) - di.atan2(dr)).to_degrees();
    (mag, phase)
}

fn make_lead_lag_plot() -> String {
    let fs = 1000.0;
    let lead = lead_compensator(std::f64::consts::FRAC_PI_4, 50.0, 1.0, fs).unwrap();
    let lag = lag_compensator(10.0, 5.0, fs).unwrap();

    let (b_lead, a_lead) = lead.coefficients();
    let (b_lag, a_lag) = lag.coefficients();

    const N: usize = 400;
    let f_min: f64 = 0.1;
    let f_max: f64 = 490.0;
    let mut freqs = vec![0.0; N];
    let mut lead_db = vec![0.0; N];
    let mut lead_ph = vec![0.0; N];
    let mut lag_db = vec![0.0; N];
    let mut lag_ph = vec![0.0; N];

    for i in 0..N {
        let f = f_min * (f_max / f_min).powf(i as f64 / (N - 1) as f64);
        freqs[i] = f;
        let (m, p) = biquad_freq_response(&b_lead, &a_lead, f, fs);
        lead_db[i] = 20.0 * m.log10();
        lead_ph[i] = p;
        let (m, p) = biquad_freq_response(&b_lag, &a_lag, f, fs);
        lag_db[i] = 20.0 * m.log10();
        lag_ph[i] = p;
    }

    // Magnitude plot
    let traces_mag = format!(
        "[{{\"type\":\"scatter\",\"mode\":\"lines\",\"name\":\"Lead (45° @ 50 Hz)\",\
          \"x\":{},\"y\":{},\"line\":{{\"width\":2.5}}}},\
         {{\"type\":\"scatter\",\"mode\":\"lines\",\"name\":\"Lag (10× DC @ 5 Hz)\",\
          \"x\":{},\"y\":{},\"line\":{{\"width\":2.5}}}}]",
        fmt_arr(&freqs), fmt_arr(&lead_db),
        fmt_arr(&freqs), fmt_arr(&lag_db),
    );
    let layout_mag = decorate_layout_ex(
        "Lead / Lag Compensators — Magnitude",
        "Frequency (Hz)",
        ",\"type\":\"log\"",
        "Magnitude (dB)",
        "",
        "",
    );

    // Phase plot
    let traces_ph = format!(
        "[{{\"type\":\"scatter\",\"mode\":\"lines\",\"name\":\"Lead phase\",\
          \"x\":{},\"y\":{},\"line\":{{\"width\":2.5}}}},\
         {{\"type\":\"scatter\",\"mode\":\"lines\",\"name\":\"Lag phase\",\
          \"x\":{},\"y\":{},\"line\":{{\"width\":2.5}}}}]",
        fmt_arr(&freqs), fmt_arr(&lead_ph),
        fmt_arr(&freqs), fmt_arr(&lag_ph),
    );
    let layout_ph = decorate_layout_ex(
        "Lead / Lag Compensators — Phase",
        "Frequency (Hz)",
        ",\"type\":\"log\"",
        "Phase (°)",
        "",
        "",
    );

    let mut html = plotly_snippet("plot-lead-lag-mag", &traces_mag, &layout_mag, 380);
    html.push_str(&plotly_snippet(
        "plot-lead-lag-phase",
        &traces_ph,
        &layout_ph,
        380,
    ));
    html
}

// ─── ODE: Van der Pol (stiff, RODAS4) ─────────────────────────────────────

fn make_vanderpol_plot() -> String {
    let mu = 20.0_f64;
    let y0 = Vector::from_array([2.0, 0.0]);
    let t_end = 120.0;

    let settings = AdaptiveSettings {
        abs_tol: 1e-8,
        rel_tol: 1e-8,
        max_steps: 100_000,
        dense_output: true,
        ..AdaptiveSettings::default()
    };

    let sol = RODAS4::integrate(
        0.0,
        t_end,
        &y0,
        |_t, y| {
            Vector::from_array([y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]])
        },
        |_t, y| {
            numeris::Matrix::new([
                [0.0, 1.0],
                [-2.0 * mu * y[0] * y[1] - 1.0, mu * (1.0 - y[0] * y[0])],
            ])
        },
        &settings,
    )
    .expect("Van der Pol integration failed");

    // Downsample accepted step points to a fixed grid for a manageable HTML size.
    // The adaptive solver clusters points at sharp transitions; we keep enough
    // resolution by picking the nearest stored point for each output sample.
    let ds = sol.dense.as_ref().expect("no dense output");
    let n_out = 2000usize;
    let mut tv = Vec::with_capacity(n_out);
    let mut xv = Vec::with_capacity(n_out);
    let mut idx = 0usize;
    for i in 0..n_out {
        let t_want = t_end * i as f64 / (n_out - 1) as f64;
        // advance index to nearest stored point
        while idx + 1 < ds.t.len() && ds.t[idx + 1] <= t_want {
            idx += 1;
        }
        tv.push(t_want);
        xv.push(ds.y[idx][0]);
    }

    let traces = format!(
        "[{{\"type\":\"scatter\",\"mode\":\"lines\",\"name\":\"y₁(t)\",\
          \"x\":{},\"y\":{},\"line\":{{\"width\":2}}}}]",
        fmt_arr(&tv),
        fmt_arr(&xv),
    );

    let layout = decorate_layout(
        "Van der Pol Oscillator (μ = 20) — RODAS4",
        "t",
        "y₁",
        "",
    );

    plotly_snippet("plot-vanderpol", &traces, &layout, 420)
}

// ─── Stats: continuous PDF plots ──────────────────────────────────────────

/// Helper: build a single line trace JSON fragment.
fn line_trace(name: &str, x: &[f64], y: &[f64]) -> String {
    format!(
        "{{\"type\":\"scatter\",\"mode\":\"lines\",\"name\":\"{name}\",\
         \"x\":{},\"y\":{},\"line\":{{\"width\":2.5}}}}",
        fmt_arr(x),
        fmt_arr(y),
    )
}

/// Helper: build a bar trace JSON fragment (for PMFs).
fn bar_trace(name: &str, x: &[f64], y: &[f64]) -> String {
    format!(
        "{{\"type\":\"bar\",\"name\":\"{name}\",\
         \"x\":{},\"y\":{},\"opacity\":0.7}}",
        fmt_arr(x),
        fmt_arr(y),
    )
}

fn make_normal_pdf_plot() -> String {
    let dists: Vec<(Normal<f64>, &str)> = vec![
        (Normal::new(0.0, 1.0).unwrap(), "μ=0, σ=1"),
        (Normal::new(0.0, 2.0).unwrap(), "μ=0, σ=2"),
        (Normal::new(2.0, 0.7).unwrap(), "μ=2, σ=0.7"),
    ];

    const N: usize = 300;
    let x_min = -6.0_f64;
    let x_max = 6.0_f64;
    let mut xv: Vec<f64> = (0..N)
        .map(|i| x_min + (x_max - x_min) * i as f64 / (N - 1) as f64)
        .collect();
    // ensure exact 0
    xv[N / 2] = 0.0;

    let traces: Vec<String> = dists
        .iter()
        .map(|(d, name)| {
            let yv: Vec<f64> = xv.iter().map(|&x| d.pdf(x)).collect();
            line_trace(name, &xv, &yv)
        })
        .collect();

    let layout = decorate_layout("Normal Distribution — PDF", "x", "f(x)", "");
    plotly_snippet("plot-normal-pdf", &format!("[{}]", traces.join(",")), &layout, 400)
}

fn make_gamma_pdf_plot() -> String {
    let dists: Vec<(Gamma<f64>, &str)> = vec![
        (Gamma::new(1.0, 1.0).unwrap(), "α=1, β=1 (Exp)"),
        (Gamma::new(2.0, 1.0).unwrap(), "α=2, β=1"),
        (Gamma::new(5.0, 1.0).unwrap(), "α=5, β=1"),
        (Gamma::new(2.0, 2.0).unwrap(), "α=2, β=2"),
    ];

    const N: usize = 300;
    let x_min = 0.01_f64;
    let x_max = 12.0_f64;
    let xv: Vec<f64> = (0..N)
        .map(|i| x_min + (x_max - x_min) * i as f64 / (N - 1) as f64)
        .collect();

    let traces: Vec<String> = dists
        .iter()
        .map(|(d, name)| {
            let yv: Vec<f64> = xv.iter().map(|&x| d.pdf(x)).collect();
            line_trace(name, &xv, &yv)
        })
        .collect();

    let layout = decorate_layout("Gamma Distribution — PDF", "x", "f(x)", "");
    plotly_snippet("plot-gamma-pdf", &format!("[{}]", traces.join(",")), &layout, 400)
}

fn make_beta_pdf_plot() -> String {
    let dists: Vec<(Beta<f64>, &str)> = vec![
        (Beta::new(0.5, 0.5).unwrap(), "α=0.5, β=0.5"),
        (Beta::new(2.0, 2.0).unwrap(), "α=2, β=2"),
        (Beta::new(2.0, 5.0).unwrap(), "α=2, β=5"),
        (Beta::new(5.0, 2.0).unwrap(), "α=5, β=2"),
    ];

    const N: usize = 300;
    let eps = 0.005;
    let xv: Vec<f64> = (0..N)
        .map(|i| eps + (1.0 - 2.0 * eps) * i as f64 / (N - 1) as f64)
        .collect();

    let traces: Vec<String> = dists
        .iter()
        .map(|(d, name)| {
            let yv: Vec<f64> = xv.iter().map(|&x| d.pdf(x).min(8.0)).collect();
            line_trace(name, &xv, &yv)
        })
        .collect();

    let layout = decorate_layout_ex(
        "Beta Distribution — PDF",
        "x",
        "",
        "f(x)",
        ",\"range\":[0,4.5]",
        "",
    );
    plotly_snippet("plot-beta-pdf", &format!("[{}]", traces.join(",")), &layout, 400)
}

// ─── Stats: discrete PMF plots ───────────────────────────────────────────

fn make_binomial_pmf_plot() -> String {
    let dists: Vec<(Binomial<f64>, &str)> = vec![
        (Binomial::new(10, 0.3).unwrap(), "n=10, p=0.3"),
        (Binomial::new(10, 0.5).unwrap(), "n=10, p=0.5"),
        (Binomial::new(20, 0.7).unwrap(), "n=20, p=0.7"),
    ];

    let k_max = 21u64;
    let kv: Vec<f64> = (0..=k_max).map(|k| k as f64).collect();

    let traces: Vec<String> = dists
        .iter()
        .map(|(d, name)| {
            let yv: Vec<f64> = (0..=k_max).map(|k| d.pmf(k)).collect();
            bar_trace(name, &kv, &yv)
        })
        .collect();

    let layout = decorate_layout_ex(
        "Binomial Distribution — PMF",
        "k",
        "",
        "P(X = k)",
        "",
        ",\"barmode\":\"group\"",
    );
    plotly_snippet("plot-binomial-pmf", &format!("[{}]", traces.join(",")), &layout, 400)
}

fn make_poisson_pmf_plot() -> String {
    let dists: Vec<(Poisson<f64>, &str)> = vec![
        (Poisson::new(1.0).unwrap(), "λ = 1"),
        (Poisson::new(4.0).unwrap(), "λ = 4"),
        (Poisson::new(10.0).unwrap(), "λ = 10"),
    ];

    let k_max = 20u64;
    let kv: Vec<f64> = (0..=k_max).map(|k| k as f64).collect();

    let traces: Vec<String> = dists
        .iter()
        .map(|(d, name)| {
            let yv: Vec<f64> = (0..=k_max).map(|k| d.pmf(k)).collect();
            bar_trace(name, &kv, &yv)
        })
        .collect();

    let layout = decorate_layout_ex(
        "Poisson Distribution — PMF",
        "k",
        "",
        "P(X = k)",
        "",
        ",\"barmode\":\"group\"",
    );
    plotly_snippet("plot-poisson-pmf", &format!("[{}]", traces.join(",")), &layout, 400)
}

fn make_continuous_cdf_plot() -> String {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let gamma = Gamma::new(3.0, 1.0).unwrap();
    let beta = Beta::new(2.0, 5.0).unwrap();

    const N: usize = 300;

    // Normal CDF on [-4, 4]
    let xn: Vec<f64> = (0..N)
        .map(|i| -4.0 + 8.0 * i as f64 / (N - 1) as f64)
        .collect();
    let yn: Vec<f64> = xn.iter().map(|&x| normal.cdf(x)).collect();

    // Gamma CDF on [0, 10]
    let xg: Vec<f64> = (0..N)
        .map(|i| 0.01 + 10.0 * i as f64 / (N - 1) as f64)
        .collect();
    let yg: Vec<f64> = xg.iter().map(|&x| gamma.cdf(x)).collect();

    // Beta CDF on [0, 1]
    let xb: Vec<f64> = (0..N)
        .map(|i| 0.005 + 0.99 * i as f64 / (N - 1) as f64)
        .collect();
    let yb: Vec<f64> = xb.iter().map(|&x| beta.cdf(x)).collect();

    let traces = format!(
        "[{},{},{}]",
        line_trace("Normal(0, 1)", &xn, &yn),
        line_trace("Gamma(3, 1)", &xg, &yg),
        line_trace("Beta(2, 5)", &xb, &yb),
    );

    let layout = decorate_layout(
        "Continuous Distributions — CDF",
        "x",
        "F(x)",
        "",
    );
    plotly_snippet("plot-continuous-cdf", &traces, &layout, 400)
}

// ─── main ─────────────────────────────────────────────────────────────────

fn main() {
    let includes = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/includes");
    fs::create_dir_all(&includes).expect("failed to create docs/includes/");

    println!("Generating Plotly HTML snippets...");

    write_snippet(&includes, "plot_ode", &make_ode_plot());
    write_snippet(&includes, "plot_vanderpol", &make_vanderpol_plot());
    write_snippet(&includes, "plot_interp", &make_interp_plot());
    write_snippet(&includes, "plot_control", &make_control_plot());
    write_snippet(&includes, "plot_lead_lag", &make_lead_lag_plot());
    write_snippet(&includes, "plot_normal_pdf", &make_normal_pdf_plot());
    write_snippet(&includes, "plot_gamma_pdf", &make_gamma_pdf_plot());
    write_snippet(&includes, "plot_beta_pdf", &make_beta_pdf_plot());
    write_snippet(&includes, "plot_binomial_pmf", &make_binomial_pmf_plot());
    write_snippet(&includes, "plot_poisson_pmf", &make_poisson_pmf_plot());
    write_snippet(&includes, "plot_continuous_cdf", &make_continuous_cdf_plot());

    println!("Done.");
}
