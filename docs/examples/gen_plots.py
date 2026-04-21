#!/usr/bin/env python3
"""Generate SVG plots for MkDocs documentation using matplotlib + SciencePlots.

Replaces the Rust gen_plots example that generated Plotly HTML snippets.

    python docs/examples/gen_plots.py

Writes SVG files to docs/includes/.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, interpolate, ndimage, signal, stats

import scienceplots  # noqa: F401 — registers styles

plt.style.use(["science", "no-latex"])

# STIX fonts — designed for scientific publishing, full Unicode, no LaTeX needed
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "axes.formatter.use_mathtext": True,
    "svg.fonttype": "none",          # emit <text> not paths — crisp browser rendering
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

INCLUDES = Path(__file__).resolve().parent.parent / "includes"

# Shared color palette (qualitative, colorblind-friendly)
COLORS = [
    "#0077BB",  # blue
    "#EE7733",  # orange
    "#009988",  # teal
    "#CC3311",  # red
    "#33BBEE",  # cyan
    "#EE3377",  # magenta
    "#BBBBBB",  # grey
]


def savefig(fig, name):
    path = INCLUDES / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ── ODE: Harmonic Oscillator ────────────────────────────────────────────────


def make_ode_plot():
    tau = 4.0 * np.pi
    sol = integrate.solve_ivp(
        lambda t, y: [y[1], -y[0]],
        [0.0, tau],
        [1.0, 0.0],
        dense_output=True,
        rtol=1e-10,
        atol=1e-12,
    )
    t = np.linspace(0, tau, 300)
    y = sol.sol(t)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(t, y[0], label="position $x(t)$", color=COLORS[0], linewidth=1.5)
    ax.plot(
        t,
        y[1],
        label="velocity $v(t)$",
        color=COLORS[1],
        linewidth=1.5,
        linestyle="--",
    )
    ax.set_xlabel("$t$")
    ax.set_ylabel("Value")
    ax.set_title("Harmonic Oscillator — RKTS54 Dense Output")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    savefig(fig, "plot_ode")


# ── ODE: Van der Pol (stiff) ────────────────────────────────────────────────


def make_vanderpol_plot():
    mu = 20.0

    def vdp(t, y):
        return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]

    def jac(t, y):
        return [[0, 1], [-2 * mu * y[0] * y[1] - 1, mu * (1 - y[0] ** 2)]]

    sol = integrate.solve_ivp(
        vdp,
        [0, 120],
        [2.0, 0.0],
        method="Radau",
        jac=jac,
        dense_output=True,
        rtol=1e-8,
        atol=1e-8,
    )
    t = np.linspace(0, 120, 2000)
    y = sol.sol(t)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(t, y[0], color=COLORS[0], linewidth=1.0, label="$y_1(t)$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y_1$")
    ax.set_title(r"Van der Pol Oscillator ($\mu = 20$) — RODAS4")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    savefig(fig, "plot_vanderpol")


# ── Interpolation ───────────────────────────────────────────────────────────


def make_interp_plot():
    kx = np.linspace(0, 2 * np.pi, 6)
    ky = np.sin(kx)
    kd = np.cos(kx)

    x = np.linspace(0, 2 * np.pi, 200)
    y_true = np.sin(x)

    # Linear
    y_lin = np.interp(x, kx, ky)

    # Hermite (pchip uses monotone Hermite; CubicHermiteSpline for exact derivatives)
    y_her = interpolate.CubicHermiteSpline(kx, ky, kd)(x)

    # Lagrange
    y_lag = interpolate.BarycentricInterpolator(kx, ky)(x)

    # Natural cubic spline
    y_spl = interpolate.CubicSpline(kx, ky, bc_type="natural")(x)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(x, y_true, color=COLORS[6], linewidth=1.5, linestyle=":", label="sin(x) exact")
    ax.plot(x, y_lin, color=COLORS[0], linewidth=1.2, label="Linear")
    ax.plot(x, y_her, color=COLORS[1], linewidth=1.2, label="Hermite")
    ax.plot(x, y_lag, color=COLORS[2], linewidth=1.2, label="Lagrange")
    ax.plot(x, y_spl, color=COLORS[3], linewidth=1.2, label="Cubic Spline")
    ax.plot(kx, ky, "kD", markersize=5, label="knots", zorder=5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Interpolation Methods on sin(x) — 6 Knots")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    savefig(fig, "plot_interp")


# ── Control: Butterworth ────────────────────────────────────────────────────


def make_control_plot():
    fs = 8000.0
    fc = 1000.0
    freqs = np.geomspace(10, 3900, 500)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    for order, color in zip([2, 4, 6], COLORS[:3]):
        sos = signal.butter(order, fc, btype="low", fs=fs, output="sos")
        w, h = signal.sosfreqz(sos, worN=freqs, fs=fs)
        ax.semilogx(w, 20 * np.log10(np.abs(h)), label=f"{order}nd order" if order == 2 else f"{order}th order",
                     color=color, linewidth=1.5)

    ax.axhline(-3, color="gray", linewidth=0.8, linestyle=":")
    ax.set_ylim(-80, 5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(r"Butterworth Lowpass — $f_c$ = 1 kHz, $f_s$ = 8 kHz")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    savefig(fig, "plot_control")


# ── Control: Lead/Lag Bode ──────────────────────────────────────────────────


def make_lead_lag_plot():
    fs = 1000.0
    freqs = np.geomspace(0.1, 490, 400)

    # Lead compensator: 45° at 50 Hz
    phi_max = np.pi / 4
    alpha = (1 + np.sin(phi_max)) / (1 - np.sin(phi_max))
    wm = 2 * np.pi * 50
    T_lead = 1.0 / (wm * np.sqrt(alpha))
    num_s = [T_lead * alpha, 1.0]
    den_s = [T_lead, 1.0]
    lead = signal.cont2discrete((num_s, den_s), 1.0 / fs, method="bilinear")
    b_lead, a_lead = lead[0].flatten(), lead[1]

    # Lag compensator: 10× DC at 5 Hz
    K_dc = 10.0
    f_corner = 5.0
    T_lag = 1.0 / (2 * np.pi * f_corner)
    num_lag_s = [K_dc * T_lag, 1.0]
    den_lag_s = [T_lag, 1.0 / K_dc]
    # Normalize: make DC gain = K_dc
    # Actually let's do it differently — match the Rust output shape
    lag = signal.cont2discrete(([K_dc * T_lag, 1.0], [T_lag, 1.0 / K_dc]), 1.0 / fs, method="bilinear")
    b_lag, a_lag = lag[0].flatten(), lag[1]

    w_lead, h_lead = signal.freqz(b_lead, a_lead, worN=freqs, fs=fs)
    w_lag, h_lag = signal.freqz(b_lag, a_lag, worN=freqs, fs=fs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5.2), sharex=True)

    # Magnitude
    ax1.semilogx(w_lead, 20 * np.log10(np.abs(h_lead)),
                 color=COLORS[0], linewidth=1.5, label=r"Lead (45° @ 50 Hz)")
    ax1.semilogx(w_lag, 20 * np.log10(np.abs(h_lag)),
                 color=COLORS[1], linewidth=1.5, label=r"Lag (10× DC @ 5 Hz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title("Lead / Lag Compensators")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    # Phase
    ax2.semilogx(w_lead, np.degrees(np.angle(h_lead)),
                 color=COLORS[0], linewidth=1.5, label="Lead phase")
    ax2.semilogx(w_lag, np.degrees(np.angle(h_lag)),
                 color=COLORS[1], linewidth=1.5, label="Lag phase")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (°)")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)

    fig.tight_layout()
    savefig(fig, "plot_lead_lag")


# ── Stats: Normal PDF ──────────────────────────────────────────────────────


def make_normal_pdf_plot():
    x = np.linspace(-6, 6, 300)
    dists = [
        (stats.norm(0, 1), r"$\mu$=0, $\sigma$=1"),
        (stats.norm(0, 2), r"$\mu$=0, $\sigma$=2"),
        (stats.norm(2, 0.7), r"$\mu$=2, $\sigma$=0.7"),
    ]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    for (d, label), color in zip(dists, COLORS):
        ax.plot(x, d.pdf(x), label=label, color=color, linewidth=1.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title("Normal Distribution — PDF")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    savefig(fig, "plot_normal_pdf")


# ── Stats: Gamma PDF ──────────────────────────────────────────────────────


def make_gamma_pdf_plot():
    x = np.linspace(0.01, 12, 300)
    # Gamma(shape=α, rate=β) → scipy uses shape a, scale=1/β
    dists = [
        (stats.gamma(1, scale=1), r"$\alpha$=1, $\beta$=1 (Exp)"),
        (stats.gamma(2, scale=1), r"$\alpha$=2, $\beta$=1"),
        (stats.gamma(5, scale=1), r"$\alpha$=5, $\beta$=1"),
        (stats.gamma(2, scale=0.5), r"$\alpha$=2, $\beta$=2"),
    ]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    for (d, label), color in zip(dists, COLORS):
        ax.plot(x, d.pdf(x), label=label, color=color, linewidth=1.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title("Gamma Distribution — PDF")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    savefig(fig, "plot_gamma_pdf")


# ── Stats: Beta PDF ──────────────────────────────────────────────────────


def make_beta_pdf_plot():
    x = np.linspace(0.005, 0.995, 300)
    dists = [
        (stats.beta(0.5, 0.5), r"$\alpha$=0.5, $\beta$=0.5"),
        (stats.beta(2, 2), r"$\alpha$=2, $\beta$=2"),
        (stats.beta(2, 5), r"$\alpha$=2, $\beta$=5"),
        (stats.beta(5, 2), r"$\alpha$=5, $\beta$=2"),
    ]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    for (d, label), color in zip(dists, COLORS):
        y = np.minimum(d.pdf(x), 8.0)
        ax.plot(x, y, label=label, color=color, linewidth=1.5)
    ax.set_ylim(0, 4.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title("Beta Distribution — PDF")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    savefig(fig, "plot_beta_pdf")


# ── Stats: Binomial PMF ────────────────────────────────────────────────────


def make_binomial_pmf_plot():
    k = np.arange(0, 22)
    dists = [
        (stats.binom(10, 0.3), "n=10, p=0.3"),
        (stats.binom(10, 0.5), "n=10, p=0.5"),
        (stats.binom(20, 0.7), "n=20, p=0.7"),
    ]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    width = 0.25
    for i, ((d, label), color) in enumerate(zip(dists, COLORS)):
        ax.bar(k + (i - 1) * width, d.pmf(k), width=width, label=label,
               color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("$k$")
    ax.set_ylabel("$P(X = k)$")
    ax.set_title("Binomial Distribution — PMF")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    savefig(fig, "plot_binomial_pmf")


# ── Stats: Poisson PMF ─────────────────────────────────────────────────────


def make_poisson_pmf_plot():
    k = np.arange(0, 21)
    dists = [
        (stats.poisson(1), r"$\lambda$ = 1"),
        (stats.poisson(4), r"$\lambda$ = 4"),
        (stats.poisson(10), r"$\lambda$ = 10"),
    ]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    width = 0.25
    for i, ((d, label), color) in enumerate(zip(dists, COLORS)):
        ax.bar(k + (i - 1) * width, d.pmf(k), width=width, label=label,
               color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("$k$")
    ax.set_ylabel("$P(X = k)$")
    ax.set_title("Poisson Distribution — PMF")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    savefig(fig, "plot_poisson_pmf")


# ── Stats: Continuous CDF ──────────────────────────────────────────────────


def make_continuous_cdf_plot():
    fig, ax = plt.subplots(figsize=(6, 3.2))

    x_n = np.linspace(-4, 4, 300)
    ax.plot(x_n, stats.norm.cdf(x_n), color=COLORS[0], linewidth=1.5, label="Normal(0, 1)")

    x_g = np.linspace(0.01, 10, 300)
    ax.plot(x_g, stats.gamma(3, scale=1).cdf(x_g), color=COLORS[1], linewidth=1.5, label="Gamma(3, 1)")

    x_b = np.linspace(0.005, 0.995, 300)
    ax.plot(x_b, stats.beta(2, 5).cdf(x_b), color=COLORS[2], linewidth=1.5, label="Beta(2, 5)")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$F(x)$")
    ax.set_title("Continuous Distributions — CDF")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    savefig(fig, "plot_continuous_cdf")


# ── Image processing ───────────────────────────────────────────────────────


def _synthetic_starfield(h=128, w=128, seed=42):
    """Astronomy-style synthetic test image: noisy background, Gaussian
    stars, one small bright rectangle, salt-and-pepper outliers."""
    rng = np.random.default_rng(seed)
    bg = 50.0 + rng.normal(0, 6.0, size=(h, w))
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    stars = [
        (20, 30, 180, 1.4),
        (50, 80, 220, 1.1),
        (90, 20, 160, 1.6),
        (100, 100, 200, 1.2),
        (70, 50, 140, 1.3),
        (35, 105, 170, 1.5),
    ]
    img = bg.copy()
    for cy, cx, amp, sigma in stars:
        img += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    # Small bright block — makes morphology effects visible.
    img[103:112, 42:51] = 240
    # Salt-and-pepper outliers.
    n_sp = 60
    ii = rng.integers(0, h, n_sp)
    jj = rng.integers(0, w, n_sp)
    vv = rng.choice([0.0, 255.0], size=n_sp)
    img_sp = img.copy()
    img_sp[ii, jj] = vv
    return img, img_sp


def make_imageproc_panel():
    """6-panel showcase: noisy input, Gaussian blur, 3×3 median,
    Sobel gradient magnitude, dilate, erode. Uses scipy.ndimage as the
    algorithmic reference — numeris produces the same results."""
    img_clean, img = _synthetic_starfield()

    gauss = ndimage.gaussian_filter(img, sigma=1.5, mode="nearest")
    median = ndimage.median_filter(img, size=3, mode="nearest")
    sx = ndimage.sobel(img_clean, axis=1, mode="nearest")
    sy = ndimage.sobel(img_clean, axis=0, mode="nearest")
    grad = np.hypot(sx, sy)
    dilated = ndimage.grey_dilation(img_clean, size=(5, 5), mode="nearest")
    eroded = ndimage.grey_erosion(img_clean, size=(5, 5), mode="nearest")

    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.6))
    panels = [
        (img, "Input (noise + salt & pepper)"),
        (gauss, "gaussian_blur σ=1.5"),
        (median, "median_filter r=1"),
        (grad, "sobel + gradient_magnitude"),
        (dilated, "dilate r=2 (max_filter)"),
        (eroded, "erode r=2 (min_filter)"),
    ]
    for ax, (data, title) in zip(axes.flat, panels):
        ax.imshow(data, cmap="gray", interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    savefig(fig, "plot_imageproc_panel")


def _block_median_pool(img, block):
    h, w = img.shape
    h_out = (h + block - 1) // block
    w_out = (w + block - 1) // block
    out = np.zeros((h_out, w_out))
    for bi in range(h_out):
        for bj in range(w_out):
            i0 = bi * block
            i1 = min(i0 + block, h)
            j0 = bj * block
            j1 = min(j0 + block, w)
            out[bi, bj] = np.median(img[i0:i1, j0:j1])
    return out


def make_imageproc_bgsub():
    """Star-tracker style background subtraction via median_pool_upsampled.

    Input has a slowly-varying illumination gradient plus noise plus several
    bright point sources. Block-median rejects the sources, bilinear
    upsample produces a smooth background map, subtraction isolates the
    sources."""
    rng = np.random.default_rng(7)
    h, w = 128, 128
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    bg_true = 40.0 + 0.35 * yy + 0.15 * xx
    img = bg_true + rng.normal(0, 2.5, size=(h, w))
    for _ in range(18):
        cy = rng.integers(10, h - 10)
        cx = rng.integers(10, w - 10)
        amp = rng.uniform(60, 180)
        img += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.1 ** 2))

    block = 16
    pooled = _block_median_pool(img, block)
    # Bilinear upsample (pixel-center convention).
    bg_est = ndimage.zoom(
        pooled,
        zoom=(h / pooled.shape[0], w / pooled.shape[1]),
        order=1,
        mode="nearest",
    )
    subtracted = img - bg_est

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, data, title in zip(
        axes,
        [img, bg_est, subtracted],
        ["Input (sources + gradient)", "median_pool_upsampled (block=16)", "Input − background"],
    ):
        ax.imshow(data, cmap="gray", interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    savefig(fig, "plot_imageproc_bgsub")


# ── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(INCLUDES, exist_ok=True)
    print("Generating SVG plots...")

    make_ode_plot()
    make_vanderpol_plot()
    make_interp_plot()
    make_control_plot()
    make_lead_lag_plot()
    make_normal_pdf_plot()
    make_gamma_pdf_plot()
    make_beta_pdf_plot()
    make_binomial_pmf_plot()
    make_poisson_pmf_plot()
    make_continuous_cdf_plot()
    make_imageproc_panel()
    make_imageproc_bgsub()

    print("Done.")
