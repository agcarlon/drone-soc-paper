"""
Plot baseline comparison outcomes: one-body HJB vs greedy EIG vs patrol.

Outputs:
  baseline_comparison.pdf           — success rate and median energy vs SNR₀
  baseline_distance_vs_energy.pdf   — 2×2 distance-to-source vs cumulative energy grid

Usage:
    python3 plotting/plot_greedy_eig_comparison.py
    python3 plotting/plot_greedy_eig_comparison.py --results-dir results/greedy_eig_comparison
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from plotting.plotting import PAPER_COLORS, PAPER_STYLE  # noqa: E402

plt.rcParams.update(PAPER_STYLE)

PAPER_SNR0_LEVELS = (50.0, 100.0, 150.0, 200.0)
METHODS = ("order1", "greedy_eig", "patrol")
STYLES = {
    "order1": {"label": "One-body HJB", "color": PAPER_COLORS[0], "marker": "o"},
    "greedy_eig": {"label": "Greedy one-step EIG", "color": PAPER_COLORS[1], "marker": "s"},
    "patrol": {"label": "Non-adaptive patrol", "color": PAPER_COLORS[2], "marker": "^"},
}


def _wilson_interval(count: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    count = np.asarray(count, dtype=float)
    n = np.asarray(n, dtype=float)
    p = count / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denom
    return center - half, center + half


def _plot_summary(df: pd.DataFrame, out_dir: str, paper_copy: str | None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4))

    ax = axes[0]
    for method in METHODS:
        sub = df[df["method"] == method].sort_values("snr0")
        if sub.empty:
            continue
        style = STYLES[method]
        n = sub["num_trials"].to_numpy(dtype=float)
        count = np.rint(sub["success_rate"].to_numpy(dtype=float) * n)
        low, high = _wilson_interval(count, n)
        center = sub["success_rate"].to_numpy(dtype=float)
        ax.errorbar(
            sub["snr0"],
            center,
            yerr=np.maximum(0.0, np.vstack([center - low, high - center])),
            color=style["color"],
            marker=style["marker"],
            capsize=3.0,
            linewidth=2.0,
            label=style["label"],
        )
    ax.set_xlabel(r"Target initial SNR$_0$")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.95)

    ax = axes[1]
    for method in METHODS:
        sub = df[df["method"] == method].sort_values("snr0")
        if sub.empty:
            continue
        style = STYLES[method]
        ax.plot(
            sub["snr0"],
            sub["median_energy"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            label=style["label"],
        )
        ax.fill_between(
            sub["snr0"],
            sub["q10_energy"],
            sub["q90_energy"],
            color=style["color"],
            alpha=0.16,
        )
    ax.set_xlabel(r"Target initial SNR$_0$")
    ax.set_ylabel(
        r"Cumulative control-effort $\mathcal{E}$ [m$^2$ s$^{-1}$]"
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "baseline_comparison.png")
    out_pdf = os.path.join(out_dir, "baseline_comparison.pdf")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    if paper_copy:
        os.makedirs(os.path.dirname(paper_copy), exist_ok=True)
        fig.savefig(paper_copy, bbox_inches="tight")
        print(f"Saved: {paper_copy}")

    plt.close(fig)


def _load_tradeoff_curves(snr_dir: Path) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    curves: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {m: [] for m in METHODS}
    for seed_dir in sorted(p for p in snr_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")):
        for method in METHODS:
            cfg_path = seed_dir / method / "run_config.json"
            res_path = seed_dir / method / "run_results.npz"
            if not (cfg_path.exists() and res_path.exists()):
                continue

            with cfg_path.open() as f:
                cfg = json.load(f)
            data = np.load(res_path, allow_pickle=True)
            if "energy_cumulative_by_cycle" not in data or "mus" not in data:
                continue

            mus = np.asarray(data["mus"], dtype=float)
            theta_true = np.asarray(cfg["problem"]["pollutant"]["source"], dtype=float)
            distances = np.linalg.norm(mus - theta_true[None, :], axis=1)
            energies = np.asarray(data["energy_cumulative_by_cycle"], dtype=float)

            n = min(len(distances), len(energies))
            if n == 0:
                continue
            curves[method].append((energies[:n], distances[:n]))
    return curves


def _interpolate_common_effort(
    curves: List[Tuple[np.ndarray, np.ndarray]], *, num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    if not curves:
        return np.empty((0,), dtype=float), np.empty((0, 0), dtype=float)
    cleaned = []
    for energy, distance in curves:
        order = np.argsort(energy)
        energy_sorted = np.asarray(energy, dtype=float)[order]
        distance_sorted = np.asarray(distance, dtype=float)[order]
        unique_energy, unique_idx = np.unique(energy_sorted, return_index=True)
        cleaned.append((unique_energy, distance_sorted[unique_idx]))
    lower = max(float(energy[0]) for energy, _ in cleaned)
    upper = min(float(energy[-1]) for energy, _ in cleaned)
    if upper < lower:
        raise ValueError("Tradeoff curves have no common effort range.")
    grid = np.linspace(lower, upper, max(2, int(num_points)))
    return grid, np.stack([np.interp(grid, energy, distance) for energy, distance in cleaned])


def _plot_distance_vs_energy_axis(ax, snr_dir: Path) -> bool:
    curves_by_method = _load_tradeoff_curves(snr_dir)
    if not any(curves_by_method[m] for m in METHODS):
        return False

    snr_label = snr_dir.name.replace("snr0_", "")
    for method in METHODS:
        effort_grid, distances = _interpolate_common_effort(curves_by_method[method])
        if effort_grid.size == 0:
            continue
        x_med = effort_grid
        y_med = np.median(distances, axis=0)
        y_q10 = np.percentile(distances, 10.0, axis=0)
        y_q90 = np.percentile(distances, 90.0, axis=0)
        style = STYLES[method]
        ax.plot(x_med, y_med, color=style["color"], linewidth=2.0, label=style["label"])
        ax.fill_between(x_med, y_q10, y_q90, color=style["color"], alpha=0.18)

    ax.set_title(rf"$\mathrm{{SNR}}_0 = {snr_label}$")
    ax.set_xlabel(
        r"Cumulative control-effort $\mathcal{E}$ [m$^2$ s$^{-1}$]"
    )
    ax.set_ylabel("Source-location error [m]")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    return True


def _plot_distance_vs_energy_grid(
    results_dir: Path,
    *,
    snr0_levels: tuple[float, ...],
    out_dir: str,
    paper_copy: str | None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    plotted = 0
    for ax, snr0 in zip(axes.ravel(), snr0_levels):
        snr_dir = results_dir / f"snr0_{float(snr0):.1f}"
        if not snr_dir.exists():
            ax.set_visible(False)
            continue
        if _plot_distance_vs_energy_axis(ax, snr_dir):
            plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise FileNotFoundError("No run_results.npz inputs found for the requested SNR levels.")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, framealpha=0.95, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "baseline_distance_vs_energy.png")
    out_pdf = os.path.join(out_dir, "baseline_distance_vs_energy.pdf")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    if paper_copy:
        os.makedirs(os.path.dirname(paper_copy), exist_ok=True)
        fig.savefig(paper_copy, bbox_inches="tight")
        print(f"Saved: {paper_copy}")

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot greedy-EIG comparison outcomes")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(ROOT, "results", "greedy_eig_comparison"),
    )
    parser.add_argument(
        "--paper-copy",
        default=os.path.join(ROOT, "paper", "figures", "baseline_comparison.pdf"),
    )
    parser.add_argument(
        "--distance-paper-copy",
        default=os.path.join(ROOT, "paper", "figures", "baseline_distance_vs_energy.pdf"),
    )
    parser.add_argument(
        "--figures-dir",
        default=os.path.join(ROOT, "paper", "figures"),
        help="Directory for per-SNR distance-vs-energy paper copies",
    )
    parser.add_argument(
        "--snr0",
        type=float,
        nargs="*",
        default=None,
        help="SNR₀ level(s) for distance-vs-energy plots; default: all paper levels",
    )
    parser.add_argument(
        "--no-distance-energy",
        action="store_true",
        help="Skip per-SNR distance-vs-energy plots",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    csv_path = results_dir / "summary_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results not found at {csv_path}.\n"
            "Run:  python3 experiments/exp_greedy_eig_comparison.py --num-seeds 50 "
            "--methods order1 greedy_eig"
        )

    df = pd.read_csv(csv_path)
    required = {
        "method", "snr0", "num_trials", "success_rate",
        "median_energy", "q10_energy", "q90_energy",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"No baseline summary rows found in {csv_path}")
    df["snr0"] = df["snr0"].astype(float)
    _plot_summary(df, str(results_dir), args.paper_copy or None)

    if not args.no_distance_energy:
        snr0_levels = tuple(float(s) for s in (args.snr0 if args.snr0 else PAPER_SNR0_LEVELS))
        _plot_distance_vs_energy_grid(
            results_dir,
            snr0_levels=snr0_levels,
            out_dir=str(results_dir),
            paper_copy=args.distance_paper_copy or None,
        )


if __name__ == "__main__":
    main()
