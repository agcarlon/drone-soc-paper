"""
Plot summary metrics for the pairwise-showcase discount sweep.

Usage:
    python3 plotting/plot_discount_sweep_summary.py
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import numpy as np

from experiments.compute_paper_statistics import BOOTSTRAP_SEED, _bca_interval  # noqa: E402
from experiments.paper_result_filters import accepted_fixed_point_rows  # noqa: E402
from plotting.plotting import PAPER_STYLE  # noqa: E402
plt.rcParams.update(PAPER_STYLE)
METRICS = (
    ("travel_distance", r"Travel distance [m]"),
    ("energy", r"Control-effort proxy $\mathcal{E}$ [m$^2$ s$^{-1}$]"),
    ("runtime_sec", r"Runtime [s]"),
)


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _group_values(rows: List[Dict[str, str]], metric: str) -> Dict[str, Dict[float, np.ndarray]]:
    grouped: Dict[str, Dict[float, List[float]]] = {}
    for row in rows:
        method = str(row["method"])
        discount = float(row["discount"])
        value = float(row[metric])
        grouped.setdefault(method, {}).setdefault(discount, []).append(value)

    out: Dict[str, Dict[float, np.ndarray]] = {}
    for method, by_discount in grouped.items():
        out[method] = {
            discount: np.asarray(values, dtype=float)
            for discount, values in by_discount.items()
        }
    return out


def _plot_metric(
    ax,
    rows: List[Dict[str, str]],
    metric: str,
    ylabel: str,
    *,
    rng: np.random.Generator,
) -> None:
    grouped = _group_values(rows, metric)
    styles = {
        "order1": {"color": "C0", "label": "One-body", "marker": "o", "linestyle": "-"},
        "order2": {"color": "C2", "label": "Pairwise", "marker": "s", "linestyle": "--"},
    }

    for method in ("order1", "order2"):
        if method not in grouped:
            continue
        discounts = np.asarray(sorted(grouped[method].keys()), dtype=float)
        intervals = np.asarray(
            [_bca_interval(grouped[method][r], rng=rng) for r in discounts],
            dtype=float,
        )
        med = intervals[:, 0]
        style = styles[method]
        ax.plot(
            discounts,
            med,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            color=style["color"],
            label=style["label"],
        )
        ax.errorbar(
            discounts,
            med,
            yerr=np.maximum(
                0.0, np.vstack([med - intervals[:, 1], intervals[:, 2] - med])
            ),
            fmt="none",
            capsize=3.0,
            linewidth=1.2,
            color=style["color"],
        )

    ax.set_xlabel(r"Discount rate $r$ [s$^{-1}$]")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({float(row["discount"]) for row in rows}))
    ax.grid(True, alpha=0.3)


def plot_discount_sweep_summary(
    *,
    csv_path: Path,
    out_path: Path,
    paper_copy: Path | None,
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing per-trial results CSV: {csv_path}")

    rows = accepted_fixed_point_rows(_read_rows(csv_path))
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")
    methods = {str(row["method"]) for row in rows}
    if not {"order1", "order2"}.issubset(methods):
        raise ValueError(f"Both one-body and pairwise rows are required in {csv_path}")
    discounts_by_method = {
        method: {float(row["discount"]) for row in rows if row["method"] == method}
        for method in ("order1", "order2")
    }
    if discounts_by_method["order1"] != discounts_by_method["order2"]:
        raise ValueError(f"One-body and pairwise discount grids differ in {csv_path}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    for ax, (metric, ylabel) in zip(axes, METRICS):
        _plot_metric(ax, rows, metric, ylabel, rng=rng)
    axes[0].legend(loc="best", framealpha=0.95)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    print(f"Saved: {pdf_path}")

    if paper_copy is not None:
        paper_copy.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_copy, bbox_inches="tight")
        print(f"Saved: {paper_copy}")

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cooperative discount sweep summary")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(ROOT, "results", "paper_pairwise_discount_sweep"),
        help="Directory containing per_trial_results.csv",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path; default is <results-dir>/discount_sweep_summary.png",
    )
    parser.add_argument(
        "--paper-copy",
        default=os.path.join(ROOT, "paper", "figures", "discount_sweep_summary.pdf"),
        help="Optional PDF copy for the paper; use an empty string to disable.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    csv_path = results_dir / "per_trial_results.csv"
    out_path = Path(args.out) if args.out is not None else results_dir / "discount_sweep_summary.png"
    paper_copy = Path(args.paper_copy) if args.paper_copy else None

    plot_discount_sweep_summary(csv_path=csv_path, out_path=out_path, paper_copy=paper_copy)


if __name__ == "__main__":
    main()
