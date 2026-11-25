"""
Plot the pairwise showcase experiment setting (geometry, no-fly zones, prior box).

This script does not require any pre-computed data — it builds the scenario
directly and calls the shared plotting utility.

Usage:
    python3 plotting/plot_pairwise_showcase_setting.py
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.exp_pairwise_showcase import (  # noqa: E402
    SOURCE_X_BOUNDS,
    SOURCE_Y_BOUNDS,
    RuntimeOverrides,
    _build_case,
    _sample_scenario,
)
from plotting.plotting import plot_cooperative_experiment_setting  # noqa: E402


def plot_setting(
    *,
    seed: int = 0,
    variant: str = "prior_tuned",
    discount: float = 0.3,
    out_path: str,
) -> None:
    runtime = RuntimeOverrides(
        t_end=600.0,
        dt_meas=30.0,
        dx=60.0,
        dx_secondary=120.0,
        num_belief_discr=3,
        fp_max_iter=12,
    )
    scenario = _sample_scenario(seed, variant)
    par, problem, state, _ = _build_case(
        scenario=scenario,
        discount=discount,
        out_dir=os.path.dirname(out_path),
        variant=variant,
        runtime=runtime,
    )
    plot_cooperative_experiment_setting(
        par=par,
        problem=problem,
        s=state,
        source_box_x=SOURCE_X_BOUNDS,
        source_box_y=SOURCE_Y_BOUNDS,
        out_path=out_path,
    )
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Scenario seed (determines initial drone positions shown)",
    )
    parser.add_argument(
        "--variant",
        choices=("primary", "fallback", "collision_tuned", "prior_tuned"),
        default="prior_tuned",
    )
    parser.add_argument("--discount", type=float, default=0.3)
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "paper", "figures", "pairwise_showcase_setting.pdf"),
        help="Output PDF path",
    )
    args = parser.parse_args()
    plot_setting(
        seed=args.seed,
        variant=args.variant,
        discount=args.discount,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
