"""
Plot team-size scaling results (§5.7).

Three panels:
  1. Diagnostic-stop rate vs M
  2. Median final localisation error vs M with q10--q90 band
  3. Median control-effort proxy per drone vs M with q10--q90 band

Usage:
    python3 plotting/plot_scaling_M.py
    python3 plotting/plot_scaling_M.py --results-dir results/scaling_M
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from plotting.plotting import PAPER_COLORS, PAPER_STYLE  # noqa: E402

plt.rcParams.update(PAPER_STYLE)

_COLOR = PAPER_COLORS[0]


def _wilson_interval(count: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    count = np.asarray(count, dtype=float)
    n = np.asarray(n, dtype=float)
    p = count / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denom
    return center - half, center + half


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot M-scaling outcomes")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(ROOT, "results", "scaling_M"),
    )
    parser.add_argument(
        "--paper-copy",
        default=os.path.join(ROOT, "paper", "figures", "scaling_M.pdf"),
    )
    args = parser.parse_args()

    csv_path = os.path.join(args.results_dir, "per_trial_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Results not found at {csv_path}.\n"
            "Run:  python3 experiments/exp_scaling_M.py --num-seeds 50"
        )

    df = pd.read_csv(csv_path)
    required = {"num_drones", "final_error", "energy_per_drone", "stopped"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"No scaling rows found in {csv_path}")
    df["num_drones"] = df["num_drones"].astype(int)
    df["final_error"] = df["final_error"].astype(float)
    df["energy_per_drone"] = df["energy_per_drone"].astype(float)
    df["stopped"] = df["stopped"].astype(int).astype(bool)

    M_values = sorted(df["num_drones"].unique())
    if M_values != [2, 3, 4]:
        raise ValueError(f"Expected scaling results for M=2,3,4; found {M_values}")

    stop_rate, stop_low, stop_high, med_err, q10_err, q90_err = [], [], [], [], [], []
    med_epd, q10_epd, q90_epd = [], [], []

    for M in M_values:
        sub = df[df["num_drones"] == M]
        count = float(sub["stopped"].sum())
        n = float(len(sub))
        low, high = _wilson_interval(np.array([count]), np.array([n]))
        stop_rate.append(float(count / n))
        stop_low.append(float(low[0]))
        stop_high.append(float(high[0]))
        med_err.append(float(sub["final_error"].median()))
        q10_err.append(float(np.percentile(sub["final_error"], 10)))
        q90_err.append(float(np.percentile(sub["final_error"], 90)))
        med_epd.append(float(sub["energy_per_drone"].median()))
        q10_epd.append(float(np.percentile(sub["energy_per_drone"], 10)))
        q90_epd.append(float(np.percentile(sub["energy_per_drone"], 90)))

    M_arr = np.array(M_values)
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.4))

    # --- Panel 1: diagnostic-stop rate ---
    ax = axes[0]
    stop_rate_arr = np.asarray(stop_rate, dtype=float)
    stop_low_arr = np.asarray(stop_low, dtype=float)
    stop_high_arr = np.asarray(stop_high, dtype=float)
    ax.bar(
        M_arr,
        stop_rate_arr,
        color=_COLOR,
        alpha=0.75,
        width=0.55,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.errorbar(
        M_arr,
        stop_rate_arr,
        yerr=np.maximum(
            0.0,
            np.vstack(
                [stop_rate_arr - stop_low_arr, stop_high_arr - stop_rate_arr]
            ),
        ),
        fmt="none",
        ecolor="black",
        elinewidth=1.1,
        capsize=3.0,
        zorder=4,
    )
    ax.set_xlabel(r"Team size $M$")
    ax.set_ylabel("Diagnostic-stop rate")
    ax.set_xticks(M_arr)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    # --- Panel 2: final localisation error ---
    ax = axes[1]
    ax.plot(M_arr, med_err, color=_COLOR, marker="o", linewidth=2.0, zorder=3)
    ax.fill_between(M_arr, q10_err, q90_err, color=_COLOR, alpha=0.16)
    ax.set_xlabel(r"Team size $M$")
    ax.set_ylabel(r"Final error [m]")
    ax.set_xticks(M_arr)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: control-effort proxy per drone ---
    ax = axes[2]
    ax.plot(M_arr, med_epd, color=_COLOR, marker="o", linewidth=2.0, zorder=3)
    ax.fill_between(M_arr, q10_epd, q90_epd, color=_COLOR, alpha=0.16)
    ax.set_xlabel(r"Team size $M$")
    ax.set_ylabel(r"Control-effort proxy per drone $\mathcal{E}/M$ [m$^2$s$^{-1}$]")
    ax.set_xticks(M_arr)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    out_png = os.path.join(args.results_dir, "scaling_M.png")
    out_pdf = os.path.join(args.results_dir, "scaling_M.pdf")
    os.makedirs(args.results_dir, exist_ok=True)
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
