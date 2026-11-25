"""
Plot absolute belief error vs cumulative control effort for one-body vs pairwise runs.

For each saved run, the script pairs

    d_k = ||mu_k - theta_true||

with the cumulative-by-cycle control effort stored in `run_results.npz`.
Curves are interpolated over the effort range shared by all runs of a method
before pointwise medians and percentile bands are computed.

Usage:
    python3 plotting/plot_order1_vs_order2_distance_vs_energy.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from plotting.plotting import PAPER_STYLE  # noqa: E402
plt.rcParams.update(PAPER_STYLE)
def _load_tradeoff_curves(results_dir: Path) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    curves: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {"order1": [], "order2": []}
    for seed_dir in sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")):
        for method in curves:
            cfg_path = seed_dir / method / "run_config.json"
            res_path = seed_dir / method / "run_results.npz"
            if not (cfg_path.exists() and res_path.exists()):
                continue

            with open(cfg_path) as f:
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
    """Interpolate curves on their shared effort range before aggregation."""
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
    distances = np.stack([np.interp(grid, energy, distance) for energy, distance in cleaned])
    return grid, distances


def _plot_distance_vs_energy(ax, results_dir: Path) -> None:
    curves_by_method = _load_tradeoff_curves(results_dir)
    missing = [method for method, curves in curves_by_method.items() if not curves]
    if missing:
        raise FileNotFoundError(
            f"No complete run_config.json/run_results.npz pairs for {missing} under {results_dir}"
        )
    styles = {
        "order1": {"color": "C0", "label": "One-body"},
        "order2": {"color": "C2", "label": "Pairwise"},
    }

    for method in ("order1", "order2"):
        effort_grid, distances = _interpolate_common_effort(curves_by_method[method])
        if effort_grid.size == 0:
            continue
        x_med = effort_grid
        y_med = np.median(distances, axis=0)
        y_q10 = np.percentile(distances, 10.0, axis=0)
        y_q90 = np.percentile(distances, 90.0, axis=0)
        style = styles[method]
        ax.plot(x_med, y_med, color=style["color"], linewidth=2.0, label=style["label"])
        ax.fill_between(x_med, y_q10, y_q90, color=style["color"], alpha=0.18)

    ax.set_xlabel(r"Cumulative control-effort proxy $\mathcal{E}$ [m$^2$ s$^{-1}$]")
    ax.set_ylabel(r"Distance to source $\|\mu_\theta-\theta_{\mathrm{true}}\|$ [m]")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one-body vs pairwise belief error vs cumulative energy")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(ROOT, "results", "paper_pairwise_showcase"),
        help="Results directory containing seed_*/order{1,2}/run_results.npz",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path; default is <results-dir>/pairwise_showcase_distance_vs_energy.png",
    )
    parser.add_argument(
        "--paper-copy",
        default=os.path.join(ROOT, "paper", "figures", "pairwise_showcase_distance_vs_energy.pdf"),
        help="Optional PDF copy for the paper; use an empty string to disable.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    _plot_distance_vs_energy(ax, results_dir)
    fig.tight_layout()

    out_path = Path(args.out) if args.out is not None else results_dir / "pairwise_showcase_distance_vs_energy.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    print(f"Saved: {pdf_path}")

    if args.paper_copy:
        paper_path = Path(args.paper_copy)
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_path, bbox_inches="tight")
        print(f"Saved: {paper_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
