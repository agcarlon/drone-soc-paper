from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from measurement import (  # noqa: E402
    PlanarAdvectionDiffusionDecayPlume,
    SlicedScreenedAdvectionDiffusionPlume,
)


def _make_models() -> tuple[PlanarAdvectionDiffusionDecayPlume, SlicedScreenedAdvectionDiffusionPlume]:
    common = dict(
        source=np.array([0.0, 0.0], dtype=float),
        q=1.0,
        wind=np.array([0.8, 0.0], dtype=float),
        sigma_e=1e-3,
        ds=1.6,
        tau_s=30.0,
        eps_c=5e-2,
    )
    return (
        PlanarAdvectionDiffusionDecayPlume(**common),
        SlicedScreenedAdvectionDiffusionPlume(**common),
    )


def _log10_field(values: np.ndarray) -> np.ndarray:
    clipped = np.maximum(values, 1e-14)
    return np.log10(clipped)


def _bench_model(model, coords: np.ndarray, n_reps: int = 25) -> dict[str, float]:
    theta = np.asarray(model.source, dtype=float)
    t = 0.0

    start = time.perf_counter()
    for _ in range(n_reps):
        model.get_pollutant_concentration(coords, t, theta)
    conc_ms = 1e3 * (time.perf_counter() - start) / n_reps

    start = time.perf_counter()
    for _ in range(n_reps):
        model.get_concentration_gradient_wrt_theta(coords, t, theta)
    grad_ms = 1e3 * (time.perf_counter() - start) / n_reps

    return {
        "concentration_ms": conc_ms,
        "gradient_ms": grad_ms,
    }


def main() -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.unicode_minus": False,
    })

    planar, sliced = _make_models()
    models = [
        (r"Planar $2$-D plume", planar),
        (r"Sliced $3$-D plume", sliced),
    ]

    x_bounds = (-8.0, 8.0)
    y_bounds = (-4.5, 4.5)
    n_x = 401
    n_y = 241

    xs = np.linspace(*x_bounds, n_x)
    ys = np.linspace(*y_bounds, n_y)
    X, Y = np.meshgrid(xs, ys)
    coords = np.column_stack([X.ravel(), Y.ravel()])

    fields: dict[str, np.ndarray] = {}
    grad_norms: dict[str, np.ndarray] = {}
    bench: dict[str, dict[str, float]] = {}

    for name, model in models:
        conc = model.get_pollutant_concentration(coords, 0.0, model.source).reshape(n_y, n_x)
        grad = model.get_concentration_gradient_wrt_theta(coords, 0.0, model.source).reshape(n_y, n_x, 2)
        fields[name] = conc
        grad_norms[name] = np.linalg.norm(grad, axis=-1)
        bench[name] = _bench_model(model, coords)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2), constrained_layout=True)

    conc_logs = np.stack([_log10_field(fields[name]) for name, _ in models], axis=0)
    conc_vmin = float(np.nanmin(conc_logs))
    conc_vmax = float(np.nanmax(conc_logs))

    grad_logs = np.stack([_log10_field(grad_norms[name]) for name, _ in models], axis=0)
    grad_vmin = float(np.nanmin(grad_logs))
    grad_vmax = float(np.nanmax(grad_logs))
    shared_cmap = "viridis"

    conc_levels = np.linspace(conc_vmin, conc_vmax, 13)
    grad_levels = np.linspace(grad_vmin, grad_vmax, 13)
    conc_norm = BoundaryNorm(conc_levels, ncolors=plt.get_cmap(shared_cmap).N, clip=True)
    grad_norm = BoundaryNorm(grad_levels, ncolors=plt.get_cmap(shared_cmap).N, clip=True)
    conc_ticks = conc_levels
    grad_ticks = grad_levels

    for ax, (name, _model) in zip(axes[0, :], models):
        conc_log = _log10_field(fields[name])
        im = ax.contourf(
            X,
            Y,
            conc_log,
            levels=conc_levels,
            cmap=shared_cmap,
            norm=conc_norm,
            extend="both",
        )
        cs = ax.contour(
            X,
            Y,
            conc_log,
            levels=conc_levels,
            colors="white",
            linewidths=0.8,
            alpha=0.9,
        )
        ax.clabel(cs, fmt="%.1f", fontsize=9, inline=True, inline_spacing=3)
        ax.scatter([0.0], [0.0], c="white", s=30, edgecolors="black", linewidths=0.6, zorder=5)
        ax.annotate("", xy=(2.0, 3.8), xytext=(0.6, 3.8), arrowprops=dict(arrowstyle="->", color="white", lw=2.0))
        ax.text(0.1, 4.0, r"$\mathbf{w}$", color="white", fontsize=12, va="center")
        ax.set_title(name, fontsize=14)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")

    cbar = fig.colorbar(im, ax=axes[0, :], shrink=0.96)
    cbar.set_label(r"$\log_{10} C$")
    cbar.set_ticks(conc_ticks)

    for ax, (name, _model) in zip(axes[1, :], models):
        grad_log = _log10_field(grad_norms[name])
        im = ax.contourf(
            X,
            Y,
            grad_log,
            levels=grad_levels,
            cmap=shared_cmap,
            norm=grad_norm,
            extend="both",
        )
        cs = ax.contour(
            X,
            Y,
            grad_log,
            levels=grad_levels,
            colors="white",
            linewidths=0.8,
            alpha=0.9,
        )
        ax.clabel(cs, fmt="%.1f", fontsize=9, inline=True, inline_spacing=3)
        ax.scatter([0.0], [0.0], c="white", s=30, edgecolors="black", linewidths=0.6, zorder=5)
        ax.set_title(rf"{name} sensitivity", fontsize=14)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")

    cbar = fig.colorbar(im, ax=axes[1, :], shrink=0.96)
    cbar.set_label(r"$\log_{10}\|\nabla_{\theta} C\|$")
    cbar.set_ticks(grad_ticks)

    pdf_path = OUT_DIR / "plume_model_comparison.pdf"
    png_path = OUT_DIR / "plume_model_comparison.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {pdf_path}")
    print(f"Saved figure to {png_path}")
    print()
    print("Average vectorized evaluation times on a 401x241 grid:")
    for name, stats in bench.items():
        print(
            f"  {name:20s} "
            f"concentration={stats['concentration_ms']:.3f} ms, "
            f"theta-gradient={stats['gradient_ms']:.3f} ms"
        )
    print()
    print("Edge-case setup:")
    print("  source=(0, 0), wind=(0.8, 0.0), ds=1.6, tau_s=30.0, eps_c=0.05")


if __name__ == "__main__":
    main()
