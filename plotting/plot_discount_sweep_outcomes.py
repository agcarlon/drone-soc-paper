"""
Plot aggregate outcome metrics for the pairwise-showcase discount sweep.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd

from experiments.compute_paper_statistics import BOOTSTRAP_SEED, _bca_interval  # noqa: E402
from experiments.paper_result_filters import fixed_point_policy_accepted  # noqa: E402
from plotting.plotting import PAPER_STYLE  # noqa: E402
plt.rcParams.update(PAPER_STYLE)


def _wilson_interval(count: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    count = np.asarray(count, dtype=float)
    n = np.asarray(n, dtype=float)
    p = count / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denom
    return center - half, center + half


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot discount-sweep aggregate outcomes")
    parser.add_argument("--results-dir", default=os.path.join(ROOT, "results", "paper_pairwise_discount_sweep"))
    parser.add_argument("--paper-copy", default=os.path.join(ROOT, "paper", "figures", "discount_sweep_outcomes.pdf"))
    args = parser.parse_args()

    csv_path = os.path.join(args.results_dir, "per_trial_results.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing per-trial results CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"discount", "method", "stop_reason", "final_error", "final_trace_sigma"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    df = df[df.apply(lambda row: fixed_point_policy_accepted(row.to_dict()), axis=1)]
    if df.empty:
        raise ValueError(f"No fixed-point-accepted rows found in {csv_path}")
    df["confidence_stop"] = df["stop_reason"].astype(str).str.startswith("confidence_criterion_met").astype(float)

    discounts = sorted(float(v) for v in df["discount"].unique())
    methods = ("order1", "order2")
    missing_groups = [
        (discount, method)
        for discount in discounts
        for method in methods
        if df[(df["discount"] == discount) & (df["method"] == method)].empty
    ]
    if missing_groups:
        raise ValueError(f"Missing discount/method groups in {csv_path}: {missing_groups}")
    styles = {
        "order1": {"label": "One-body", "color": "C0", "marker": "o", "linestyle": "-"},
        "order2": {"label": "Pairwise", "color": "C2", "marker": "s", "linestyle": "--"},
    }
    metrics = [
        ("confidence_stop", "Diagnostic-stop rate", False),
        ("final_error", r"Final error [m]", True),
        ("final_trace_sigma", r"Final $\mathrm{tr}(\Sigma)$ [m$^2$]", True),
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.4))
    for ax, (col, ylabel, logy) in zip(axes, metrics):
        for method in methods:
            center = []
            ci_low = []
            ci_high = []
            for discount in discounts:
                vals = df[(df["discount"] == discount) & (df["method"] == method)][col].to_numpy(dtype=float)
                if col == "confidence_stop":
                    count = float(np.sum(vals))
                    n = float(len(vals))
                    low, high = _wilson_interval(np.array([count]), np.array([n]))
                    center.append(float(count / n))
                    ci_low.append(float(low[0]))
                    ci_high.append(float(high[0]))
                else:
                    estimate, low, high = _bca_interval(vals, rng=rng)
                    center.append(estimate)
                    ci_low.append(low)
                    ci_high.append(high)
            style = styles[method]
            center_arr = np.asarray(center, dtype=float)
            low_arr = np.asarray(ci_low, dtype=float)
            high_arr = np.asarray(ci_high, dtype=float)
            ax.plot(
                discounts,
                center_arr,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                label=style["label"],
            )
            ax.errorbar(
                discounts,
                center_arr,
                yerr=np.maximum(
                    0.0, np.vstack([center_arr - low_arr, high_arr - center_arr])
                ),
                fmt="none",
                color=style["color"],
                capsize=3.0,
                linewidth=1.2,
            )
        ax.set_xlabel(r"Discount rate $r$ [s$^{-1}$]")
        ax.set_ylabel(ylabel)
        ax.set_xticks(discounts)
        ax.grid(True, alpha=0.3)
        if logy:
            ax.set_yscale("log")
    axes[0].set_ylim(-0.04, 1.04)
    axes[0].legend(framealpha=0.95)
    fig.tight_layout()

    os.makedirs(args.results_dir, exist_ok=True)
    out_png = os.path.join(args.results_dir, "discount_sweep_outcomes.png")
    out_pdf = os.path.join(args.results_dir, "discount_sweep_outcomes.pdf")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    if args.paper_copy:
        os.makedirs(os.path.dirname(args.paper_copy), exist_ok=True)
        fig.savefig(args.paper_copy, bbox_inches="tight")
        print(f"Saved: {args.paper_copy}")
    plt.close(fig)


if __name__ == "__main__":
    main()
