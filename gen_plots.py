#!/usr/bin/env python3
"""Generate Plotly HTML snippets for MkDocs documentation.

Runs the three Rust plot examples, parses their JSON output, and writes
interactive Plotly HTML snippets to docs/includes/.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
INCLUDES = Path(__file__).parent / "includes"


def run_example(name: str, features: str) -> dict:
    result = subprocess.run(
        ["cargo", "run", "--example", name, "--features", features, "--release"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# ODE plot — harmonic oscillator dense output
# ---------------------------------------------------------------------------

def make_ode_plot(data: dict) -> str:
    t = data["t"]
    x = data["x"]
    v = data["v"]

    traces = json.dumps([
        {
            "type": "scatter",
            "mode": "lines",
            "name": "position x(t)",
            "x": t,
            "y": x,
            "line": {"width": 2},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "velocity v(t)",
            "x": t,
            "y": v,
            "line": {"width": 2, "dash": "dash"},
        },
    ])

    layout = json.dumps({
        "title": "Harmonic oscillator — RKTS54 dense output (300 pts on [0, 4π])",
        "xaxis": {"title": "t"},
        "yaxis": {"title": "value"},
        "legend": {"orientation": "h", "y": -0.2},
        "margin": {"t": 50, "b": 60},
    })

    return (
        '<div id="plot-ode" style="width:100%;height:400px;"></div>\n'
        "<script>\n"
        f"window.addEventListener('load',function(){{Plotly.newPlot('plot-ode',{traces},{layout},{{responsive:true}})}});\n"
        "</script>\n"
    )


# ---------------------------------------------------------------------------
# Interpolation plot — sin(x) method comparison
# ---------------------------------------------------------------------------

def make_interp_plot(data: dict) -> str:
    traces = json.dumps([
        {
            "type": "scatter",
            "mode": "lines",
            "name": "sin(x) true",
            "x": data["x"],
            "y": data["y_true"],
            "line": {"width": 2, "color": "#888"},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Linear",
            "x": data["x"],
            "y": data["y_linear"],
            "line": {"width": 2},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Hermite",
            "x": data["x"],
            "y": data["y_hermite"],
            "line": {"width": 2},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Lagrange",
            "x": data["x"],
            "y": data["y_lagrange"],
            "line": {"width": 2},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Cubic Spline",
            "x": data["x"],
            "y": data["y_spline"],
            "line": {"width": 2},
        },
        {
            "type": "scatter",
            "mode": "markers",
            "name": "knots",
            "x": data["kx"],
            "y": data["ky"],
            "marker": {"size": 8, "color": "#333"},
        },
    ])

    layout = json.dumps({
        "title": "Interpolation methods on sin(x) — 6 knots, 200 eval points",
        "xaxis": {"title": "x"},
        "yaxis": {"title": "y"},
        "legend": {"orientation": "h", "y": -0.25},
        "margin": {"t": 50, "b": 80},
    })

    return (
        '<div id="plot-interp" style="width:100%;height:420px;"></div>\n'
        "<script>\n"
        f"window.addEventListener('load',function(){{Plotly.newPlot('plot-interp',{traces},{layout},{{responsive:true}})}});\n"
        "</script>\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    INCLUDES.mkdir(parents=True, exist_ok=True)

    print("Running plot_ode example...")
    ode_data = run_example("plot_ode", "ode")
    (INCLUDES / "plot_ode.html").write_text(make_ode_plot(ode_data))
    print("  → docs/includes/plot_ode.html")

    print("Running plot_interp example...")
    interp_data = run_example("plot_interp", "interp")
    (INCLUDES / "plot_interp.html").write_text(make_interp_plot(interp_data))
    print("  → docs/includes/plot_interp.html")

    print("Done.")


if __name__ == "__main__":
    main()
