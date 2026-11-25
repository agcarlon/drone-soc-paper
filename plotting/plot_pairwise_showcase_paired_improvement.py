"""Plot paired pairwise-planning improvements for the showcase experiment.

The plot compares each seed against itself:

    error improvement  = final_error(one-body) - final_error(pairwise)
    energy improvement = energy(one-body) - energy(pairwise)

Positive values therefore favor pairwise planning. The localization axis uses a
symmetric log scale because a few large failures otherwise hide the paired
cloud near the origin.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FixedFormatter  # noqa: E402
from plotting.plotting import PAPER_STYLE  # noqa: E402

plt.rcParams.update(PAPER_STYLE)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
        "axes.unicode_minus": False,
    }
)


def _load_paired_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"seed", "method", "final_error", "energy"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    duplicate = df.duplicated(subset=["seed", "method"], keep=False)
    if duplicate.any():
        raise ValueError(
            f"Duplicate seed/method rows in {csv_path}: {int(duplicate.sum())} rows"
        )

    pivot = df.pivot(index="seed", columns="method")
    if "order1" not in pivot["final_error"] or "order2" not in pivot["final_error"]:
        raise ValueError("CSV must contain paired one-body and pairwise rows")

    paired = pd.DataFrame(
        {
            "seed": pivot.index,
            "error_improvement": pivot["final_error"]["order1"] - pivot["final_error"]["order2"],
            "energy_improvement": pivot["energy"]["order1"] - pivot["energy"]["order2"],
            "order1_error": pivot["final_error"]["order1"],
            "order2_error": pivot["final_error"]["order2"],
            "order1_energy": pivot["energy"]["order1"],
            "order2_energy": pivot["energy"]["order2"],
        }
    ).dropna()
    return paired


def _plot(ax: plt.Axes, paired: pd.DataFrame) -> None:
    x = paired["error_improvement"].to_numpy(dtype=float)
    y = paired["energy_improvement"].to_numpy(dtype=float)

    better_both = (x > 0.0) & (y > 0.0)
    better_error = (x > 0.0) & (y <= 0.0)
    better_energy = (x <= 0.0) & (y > 0.0)
    worse_both = (x <= 0.0) & (y <= 0.0)

    groups = [
        (better_both, "Both improved", "C2"),
        (better_error, "Accuracy only", "C0"),
        (better_energy, "Energy only", "C1"),
        (worse_both, "Neither", "0.45"),
    ]
    for mask, label, color in groups:
        if not np.any(mask):
            continue
        ax.scatter(
            x[mask],
            y[mask],
            s=32,
            color=color,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.5,
            label=f"{label} ({int(mask.sum())})",
        )

    ax.axvline(0.0, color="black", linewidth=0.9, alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.75)
    ax.set_xscale("symlog", linthresh=2.0, linscale=0.9)
    tick_values = [-100, -30, -10, -3, 0, 3, 10, 30, 100]
    ax.xaxis.set_major_locator(FixedLocator(tick_values))
    ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in tick_values]))
    ax.set_xlabel(r"Final-distance difference $d_{\mathrm{1b}}-d_{\mathrm{2b}}$ [m]")
    ax.set_ylabel(
        r"Control-effort difference $\mathcal{E}_{\mathrm{1b}}-\mathcal{E}_{\mathrm{2b}}$ "
        r"[m$^2$ s$^{-1}$]"
    )
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(loc="best", framealpha=0.95, fontsize=8)

    med_x = float(np.median(x))
    med_y = float(np.median(y))
    ax.plot(med_x, med_y, marker="D", color="black", markersize=5, label="_nolegend_")
    ax.annotate(
        r"median",
        xy=(med_x, med_y),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=8,
        color="black",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paired pairwise-planning improvements for the showcase")
    parser.add_argument(
        "--csv",
        default=os.path.join(ROOT, "results", "paper_pairwise_showcase", "per_trial_results.csv"),
        help="Per-trial CSV with paired one-body/pairwise results",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "results", "paper_pairwise_showcase", "pairwise_showcase_paired_improvement.png"),
        help="Output PNG path. A PDF with the same stem is also written.",
    )
    parser.add_argument(
        "--paper-copy",
        default=os.path.join(ROOT, "paper", "figures", "pairwise_showcase_paired_improvement.pdf"),
        help="Optional PDF copy for the paper; use an empty string to disable.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    paired = _load_paired_metrics(csv_path)
    if paired.empty:
        raise ValueError("No paired one-body/pairwise seeds found")

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.1))
    _plot(ax, paired)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
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
