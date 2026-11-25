"""
Plot wind-aligned vs no-alignment single-drone comparison from saved CSV.

Usage:
    python3 plotting/plot_wind_aligned_comparison.py
    python3 plotting/plot_wind_aligned_comparison.py --csv results/wind_aligned_comparison/wind_aligned_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from plotting.plotting import PAPER_COLORS, PAPER_STYLE, styled_boxplot  # noqa: E402

plt.rcParams.update(PAPER_STYLE)


def plot_wind_aligned_comparison(rows_wa: list[dict], rows_nwa: list[dict], out_path: str) -> None:
    if not rows_wa or not rows_nwa:
        raise ValueError("Both wind-aligned and free-angle rows are required")
    if any(int(row["num_plan_calls"]) <= 0 for row in rows_wa + rows_nwa):
        raise ValueError("Every wind-alignment row must contain at least one planning call")
    tick_labels = [
        r"Wind-aligned" + "\n" + r"(5-D HJB)",
        r"No alignment" + "\n" + r"(6-D HJB)",
    ]
    colors = [PAPER_COLORS[0], PAPER_COLORS[1]]

    avg_t_wa = [float(r["total_plan_time"]) / int(r["num_plan_calls"]) for r in rows_wa]
    avg_t_nwa = [float(r["total_plan_time"]) / int(r["num_plan_calls"]) for r in rows_nwa]
    median_wa = float(np.median(avg_t_wa))
    if not np.isfinite(median_wa) or median_wa <= 0.0:
        raise ValueError("Wind-aligned planning times must have a positive finite median")
    t_ratio = float(np.median(avg_t_nwa)) / median_wa

    specs = [
        (
            [float(r["dist_final"]) for r in rows_wa],
            [float(r["dist_final"]) for r in rows_nwa],
            r"Distance to source [m]",
            r"Localization accuracy",
            True,
        ),
        (
            [float(r["energy"]) for r in rows_wa],
            [float(r["energy"]) for r in rows_nwa],
            r"$\mathcal{E}$ [m$^2$s$^{-1}$]",
            r"Control-effort proxy",
            False,
        ),
        (
            avg_t_wa,
            avg_t_nwa,
            r"Avg.\ planning time per call [s]",
            rf"Planning cost ($\times${t_ratio:.1f} without alignment)",
            False,
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
    for ax, (data_wa, data_nwa, ylabel, title, log_y) in zip(axes, specs):
        styled_boxplot(
            ax, [data_wa, data_nwa], tick_labels,
            colors=colors, log_y=log_y, ylabel=ylabel, title=title,
        )

    fig.suptitle(r"One-body HJB: wind-aligned vs no alignment")
    fig.tight_layout()

    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    for path in [str(out_path_obj), str(out_path_obj.with_suffix(".png"))]:
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default=os.path.join(ROOT, "results", "wind_aligned_comparison", "wind_aligned_comparison.csv"),
        help="CSV produced by exp_wind_aligned_comparison.py",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "paper", "figures", "wind_aligned_comparison.pdf"),
        help="Output PDF path; a PNG with the same stem is also written",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    rows_wa = [r for r in rows if r["wind_aligned"].strip().lower() in ("true", "1")]
    rows_nwa = [r for r in rows if r["wind_aligned"].strip().lower() in ("false", "0")]

    if not rows_wa or not rows_nwa:
        raise ValueError("CSV must contain both wind_aligned=True and wind_aligned=False rows")

    plot_wind_aligned_comparison(rows_wa, rows_nwa, args.out)


if __name__ == "__main__":
    main()
