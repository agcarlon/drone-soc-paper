"""
Paper experiment: one-body vs. pairwise-planning showcase.

Uses the prior-tuned variant of the cooperative scenario (3 drones, no-fly
zones, wind-aligned HJB, r=0.3 s^-1) and runs paired one-body/pairwise trials over
--num-seeds randomised source positions.

Output (results/paper_pairwise_showcase/):
  per_trial_results.csv  — one row per (seed, method)
  summary_results.csv    — aggregated statistics

Run plots:
  python3 plotting/plot_pairwise_showcase_summary.py
  python3 plotting/plot_pairwise_showcase_paired_improvement.py
  python3 plotting/plot_order1_vs_order2_distance_vs_energy.py
  python3 plotting/plot_pairwise_showcase_seed_suite.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.exp_pairwise_showcase import (  # noqa: E402
    DEFAULT_METHODS,
    RuntimeOverrides,
    _aggregate,
    _initial_condition_summary,
    _run_method,
    _sample_scenario,
    _write_csv,
)
from plotting.plot_pairwise_showcase_summary import plot_pairwise_showcase_summary  # noqa: E402
from experiments.result_io import load_json_if_complete  # noqa: E402


DEFAULT_OUT_DIR = os.path.join(ROOT, "results", "paper_pairwise_showcase")
DEFAULT_VARIANT = "prior_tuned"


def _metrics_path(out_dir: Path, seed: int, method: str) -> Path:
    return out_dir / f"seed_{seed:03d}" / method / "metrics.json"


def _load_metrics(path: Path) -> Dict[str, Any] | None:
    return load_json_if_complete(
        path,
        ("method", "seed", "final_error", "final_trace_sigma", "energy", "runtime_sec"),
    )


def run_paper_pairwise_showcase(
    *,
    out_dir: str,
    seeds: Sequence[int],
    methods: Sequence[str],
    discount: float,
    overwrite: bool,
) -> List[Dict[str, Any]]:
    if not seeds:
        raise ValueError("At least one seed is required")
    if not methods:
        raise ValueError("At least one method is required")
    unknown_methods = set(methods) - set(DEFAULT_METHODS)
    if unknown_methods:
        raise ValueError(f"Unknown methods: {sorted(unknown_methods)}")
    if discount <= 0.0:
        raise ValueError("Discount rate must be positive")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    runtime = RuntimeOverrides(
        t_end=600.0,
        dt_meas=30.0,
        dx=60.0,
        dx_secondary=120.0,
        num_belief_discr=3,
        fp_max_iter=12,
    )

    initial_summary = _initial_condition_summary(
        seed=int(seeds[0]),
        discount=float(discount),
        out_dir=str(out_path),
        variant=DEFAULT_VARIANT,
        runtime=runtime,
    )
    print(
        "Paper pairwise showcase:",
        f"variant={DEFAULT_VARIANT}",
        f"num_seeds={len(seeds)}",
        f"methods={list(methods)}",
        f"discount={discount}",
        f"output={out_path}",
        flush=True,
    )
    print(
        "Initial condition check:",
        f"source=({initial_summary['source_x']:.1f}, {initial_summary['source_y']:.1f})",
        f"q={initial_summary['q']:.1f}",
        f"snr={[round(float(x), 3) for x in initial_summary['initial_snr_linear']]}",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        scenario = _sample_scenario(int(seed), DEFAULT_VARIANT)
        for method in methods:
            method_dir = out_path / f"seed_{int(seed):03d}" / str(method)
            metrics_path = _metrics_path(out_path, int(seed), str(method))
            run_results_path = method_dir / "run_results.npz"
            if metrics_path.exists() and run_results_path.exists() and not overwrite:
                row = _load_metrics(metrics_path)
                if row is not None and row["method"] == method and int(row["seed"]) == int(seed):
                    print(f"Skipping existing seed={seed}, method={method}", flush=True)
                else:
                    row = None
                    print(f"Cached metrics are incomplete for seed={seed}, method={method}; rerunning.", flush=True)
            else:
                row = None
            if row is None:
                method_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"Running seed={seed}, method={method}, "
                    f"source=({scenario.source_pos[0]:.1f}, {scenario.source_pos[1]:.1f}), q={scenario.q:.1f}",
                    flush=True,
                )
                row = _run_method(
                    scenario=scenario,
                    discount=float(discount),
                    method=str(method),
                    out_dir=str(method_dir),
                    variant=DEFAULT_VARIANT,
                    make_plots=False,
                    make_contours=False,
                    runtime=runtime,
                )
                print(
                    f"  Result: error={row['final_error']:.2f}, "
                    f"trace={row['final_trace_sigma']:.2f}, "
                    f"q={row['final_q']:.2f}, "
                    f"energy={row['energy']:.2f}, "
                    f"stopped={row['stopped']}, "
                    f"boundary={100.0 * row['boundary_dwell_fraction']:.2f}%",
                    flush=True,
                )
            rows.append(row)
            _write_csv(str(out_path / "per_trial_results.csv"), rows)
            _write_csv(str(out_path / "summary_results.csv"), _aggregate(rows))

    if {"order1", "order2"}.issubset(set(methods)):
        plot_pairwise_showcase_summary(str(out_path / "per_trial_results.csv"), str(out_path))
    else:
        print("Skipping paired summary figure because both methods were not requested.", flush=True)
    print(f"Saved per-trial results to {out_path / 'per_trial_results.csv'}", flush=True)
    print(f"Saved summary results to {out_path / 'summary_results.csv'}", flush=True)
    print(f"Saved summary figures to {out_path}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper pairwise-planning showcase experiment")
    parser.add_argument("--num-seeds", type=int, default=100, help="Number of seeds starting from 0")
    parser.add_argument("--seed-offset", type=int, default=0, help="First seed index")
    parser.add_argument("--discount", type=float, default=0.3)
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Rerun cases with existing metrics.json")
    args = parser.parse_args()

    seeds = tuple(range(int(args.seed_offset), int(args.seed_offset) + int(args.num_seeds)))
    run_paper_pairwise_showcase(
        out_dir=str(args.out_dir),
        seeds=seeds,
        methods=tuple(str(m) for m in args.methods),
        discount=float(args.discount),
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
