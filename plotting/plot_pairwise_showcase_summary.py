"""
Plot robust-accuracy summaries for the pairwise showcase full run.

Default input:
    results/paper_pairwise_showcase/per_trial_results.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt  # noqa: E402
from plotting.plotting import PAPER_COLORS, PAPER_STYLE, styled_boxplot  # noqa: E402

plt.rcParams.update(PAPER_STYLE)

THRESHOLDS = np.array([1.0, 2.0, 5.0, 10.0], dtype=float)
LABELS = {"order1": "One-body", "order2": "Pairwise"}
COLORS = {"order1": PAPER_COLORS[0], "order2": PAPER_COLORS[1]}


def _wilson_interval(count: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    count = np.asarray(count, dtype=float)
    n = np.asarray(n, dtype=float)
    p = count / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denom
    return center - half, center + half


def _success_by_threshold(df: pd.DataFrame, method: str) -> np.ndarray:
    group = df[df["method"] == method]
    if group.empty:
        return np.full_like(THRESHOLDS, np.nan, dtype=float)
    errors = group["final_error"].to_numpy(dtype=float)
    return np.array([np.mean(errors <= tau) for tau in THRESHOLDS], dtype=float)


def _success_ci_by_threshold(df: pd.DataFrame, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group = df[df["method"] == method]
    if group.empty:
        nan = np.full_like(THRESHOLDS, np.nan, dtype=float)
        return nan, nan, nan
    errors = group["final_error"].to_numpy(dtype=float)
    n = np.full_like(THRESHOLDS, len(errors), dtype=float)
    count = np.array([np.sum(errors <= tau) for tau in THRESHOLDS], dtype=float)
    low, high = _wilson_interval(count, n)
    estimate = count / n
    return estimate, low, high


def plot_pairwise_showcase_summary(csv_path: str, out_dir: str) -> list[str]:
    df = pd.read_csv(csv_path)
    required = {"method", "final_error", "final_trace_sigma"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    methods = [m for m in ("order1", "order2") if m in set(df["method"].astype(str))]
    if set(methods) != {"order1", "order2"}:
        raise ValueError(f"Both one-body and pairwise rows are required in {csv_path}")

    colors = [COLORS[m] for m in methods]
    labels = [LABELS[m] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))

    data_error = [df[df["method"] == m]["final_error"].to_numpy(dtype=float) for m in methods]
    styled_boxplot(axes[0], data_error, labels, colors=colors, log_y=True,
                   ylabel=r"Distance to source [m]", title="Localization error")

    data_trace = [df[df["method"] == m]["final_trace_sigma"].to_numpy(dtype=float) for m in methods]
    styled_boxplot(axes[1], data_trace, labels, colors=colors, log_y=True,
                   ylabel=r"$\mathrm{tr}(\Sigma)$ [m$^2$]", title="Posterior uncertainty")

    for method in methods:
        center, low, high = _success_ci_by_threshold(df, method)
        axes[2].errorbar(
            THRESHOLDS,
            center,
            yerr=np.maximum(0.0, np.vstack([center - low, high - center])),
            marker="o",
            linewidth=2.0,
            capsize=3.0,
            color=COLORS[method],
            label=LABELS.get(method, method),
        )
    axes[2].set_xscale("log")
    axes[2].set_xticks(THRESHOLDS)
    axes[2].set_xticklabels([f"{tau:g}" for tau in THRESHOLDS])
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_xlabel("Error threshold [m]")
    axes[2].set_ylabel("Success rate")
    axes[2].set_title("Robust accuracy")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    outputs = [
        os.path.join(out_dir, "pairwise_showcase_summary.png"),
        os.path.join(out_dir, "pairwise_showcase_summary.pdf"),
    ]
    for out_path in outputs:
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pairwise showcase robust-accuracy summaries")
    parser.add_argument(
        "--csv",
        default=os.path.join(ROOT, "results", "paper_pairwise_showcase", "per_trial_results.csv"),
    )
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "paper", "figures"))
    args = parser.parse_args()

    out_dir = args.out_dir
    outputs = plot_pairwise_showcase_summary(args.csv, out_dir)
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
