"""
Paper experiment: baseline comparisons (one-body HJB vs two baselines).

Compares three controllers on the baseline scenario (Table simulation_scenarios):
  order1      — infinite-horizon one-body HJB planner (wind-aligned)
  greedy_eig  — myopic one-step EIG maximizer (same plume model and belief)
  patrol      — passive sinusoidal patrol

Usage:
    python3 experiments/exp_greedy_eig_comparison.py --num-seeds 50

Output (results/greedy_eig_comparison/):
  per_trial_results.csv  — one row per (method, seed, SNR₀)
  summary_results.csv    — aggregated statistics

Run plots:
  python3 plotting/plot_greedy_eig_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from drone_soc import find_pollutant_source  # noqa: E402
from experiments.baseline_scenario import (  # noqa: E402
    BaselineScenario,
    BaselineScenarioSpec,
    PatrolOnlyPlanner,
    _cumulative_distance,
    _pairwise_min_distance,
    _sample_scenario,
    _snr_dir,
    make_baseline_case,
)
from greedy_eig import GreedyEIGPlanner  # noqa: E402
from infinite_horizon import InfiniteHorizonSingleDronePlanner  # noqa: E402
from experiments.fixed_point_diagnostics import fixed_point_metrics  # noqa: E402
from experiments.result_io import (  # noqa: E402
    load_json_if_complete,
    write_csv_atomic,
    write_json_atomic,
)

DEFAULT_OUT_DIR = os.path.join(ROOT, "results", "greedy_eig_comparison")
DEFAULT_METHODS = ("order1", "greedy_eig", "patrol")
SUCCESS_THRESHOLDS_M = (50.0,)


def _metrics_path(out_dir: str, snr0: float, seed: int, method: str) -> Path:
    return Path(_snr_dir(out_dir, snr0)) / f"seed_{seed:03d}" / method / "metrics.json"


def _cached_patrol_matches(
    metrics_path: Path,
    *,
    expected_period: float,
    expected_y_scale: float,
) -> bool:
    """Check that a cached passive run uses the requested patrol parameters."""
    config_path = metrics_path.parent / "run_config.json"
    try:
        with config_path.open() as f:
            config = json.load(f)
        patrol = config["problem"]["patrol"]
        planner_type = config["planner"]["type"]
        return (
            planner_type == "PatrolOnlyPlanner"
            and np.isclose(float(patrol["period"]), float(expected_period))
            and np.isclose(float(patrol["y_scale"]), float(expected_y_scale))
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _boundary_summary(traj: np.ndarray, domain: np.ndarray) -> tuple[int, float]:
    positions = np.asarray(traj, dtype=float).transpose(2, 0, 1)
    lower = np.asarray(domain, dtype=float)[:, 0]
    upper = np.asarray(domain, dtype=float)[:, 1]
    touches = np.any(
        np.isclose(positions, lower[None, None, :])
        | np.isclose(positions, upper[None, None, :]),
        axis=2,
    )
    return int(np.sum(touches)), float(np.mean(np.any(touches, axis=1)))


def _run_method(
    method: str,
    scenario: BaselineScenario,
    spec: BaselineScenarioSpec,
) -> Dict[str, Any]:
    method_dir = os.path.join(
        _snr_dir(spec.output_dir, scenario.snr0_target),
        f"seed_{scenario.seed:03d}",
        method,
    )
    os.makedirs(method_dir, exist_ok=True)

    patrol_period = (
        spec.passive_patrol_period if method == "patrol" else spec.patrol_period
    )
    patrol_y_scale = (
        spec.passive_patrol_y_scale if method == "patrol" else spec.patrol_y_scale
    )
    par, problem, s, fp_solver = make_baseline_case(
        scenario,
        output_dir=method_dir,
        patrol_period=patrol_period,
        patrol_y_scale=patrol_y_scale,
    )

    t0 = time.perf_counter()
    if method == "order1":
        planner = InfiniteHorizonSingleDronePlanner(wind_aligned=True, fp_solver=fp_solver)
    elif method == "greedy_eig":
        planner = GreedyEIGPlanner(num_restarts=8)
    elif method == "patrol":
        planner = PatrolOnlyPlanner()
    else:
        raise ValueError(f"Unknown method: {method!r}")

    traj, log = find_pollutant_source(par, problem, s, planner=planner)
    runtime_sec = float(time.perf_counter() - t0)

    final_belief = log["beliefs"][-1]
    final_error = float(np.linalg.norm(final_belief.mu - problem.pollutant.source))
    success = bool(final_error <= float(spec.success_radius))
    min_pair_distance = _pairwise_min_distance(traj)
    collision_limit = float(problem.collision.d_min) if problem.collision is not None else 0.0
    safe = bool(min_pair_distance >= collision_limit)
    energy = float(log.get("energy_total", 0.0))
    travel_distance = _cumulative_distance(traj)
    stop_reason = str(log.get("stop_reason", "reached_time_horizon"))
    stopped = str(stop_reason).startswith("confidence_criterion_met")
    boundary_touch_count, boundary_dwell_fraction = _boundary_summary(
        traj, problem.domain
    )

    row = {
        "method": method,
        "seed": int(scenario.seed),
        "snr0_target": float(scenario.snr0_target),
        "source_x": float(problem.pollutant.source[0]),
        "source_y": float(problem.pollutant.source[1]),
        "q": float(problem.pollutant.q),
        "stopped": int(stopped),
        "success": int(success),
        "safe": int(safe),
        "collision_failure": int(not safe),
        "min_pair_distance": float(min_pair_distance),
        "boundary_touch_count": int(boundary_touch_count),
        "boundary_dwell_fraction": float(boundary_dwell_fraction),
        "patrol_period": float(patrol_period),
        "patrol_y_scale": float(patrol_y_scale),
        "travel_distance": float(travel_distance),
        "energy": float(energy),
        "stop_time": float(log["times"][-1]) if log["times"] else float(par.t_end),
        "final_error": float(final_error),
        "final_trace_sigma": float(np.trace(final_belief.sigma)),
        "runtime_sec": float(runtime_sec),
        "stop_reason": stop_reason,
    }
    if method == "order1":
        row.update(fixed_point_metrics(fp_solver, method_dir))
    metrics_path = Path(method_dir) / "metrics.json"
    write_json_atomic(metrics_path, row)
    return row


def _aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: dict[tuple[str, float], list[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["method"]), float(row["snr0_target"]))
        by_key.setdefault(key, []).append(row)

    summary: List[Dict[str, Any]] = []
    for (method, snr0), group in sorted(by_key.items()):
        errors = np.array([float(g["final_error"]) for g in group], dtype=float)
        energies = np.array([float(g["energy"]) for g in group], dtype=float)
        summary.append(
            {
                "method": method,
                "snr0": float(snr0),
                "num_trials": len(group),
                "success_rate": float(np.mean([g["success"] for g in group])),
                "stop_rate": float(np.mean([g["stopped"] for g in group])),
                "safe_rate": float(np.mean([g["safe"] for g in group])),
                "median_final_error": float(np.median(errors)),
                "q10_final_error": float(np.quantile(errors, 0.1)),
                "q90_final_error": float(np.quantile(errors, 0.9)),
                "median_energy": float(np.median(energies)),
                "q10_energy": float(np.quantile(energies, 0.1)),
                "q90_energy": float(np.quantile(energies, 0.9)),
                "median_runtime_sec": float(np.median([g["runtime_sec"] for g in group])),
            }
        )
    return summary


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    write_csv_atomic(path, rows)


def run_comparison(
    *,
    out_dir: str,
    seeds: Sequence[int],
    snr0_levels: Sequence[float],
    methods: Sequence[str],
    overwrite: bool,
    passive_patrol_period: float = 40.0,
    passive_patrol_y_scale: float = 0.99,
) -> List[Dict[str, Any]]:
    if not seeds:
        raise ValueError("At least one seed is required")
    if not snr0_levels:
        raise ValueError("At least one SNR level is required")
    if not methods:
        raise ValueError("At least one method is required")
    unknown_methods = set(methods) - set(DEFAULT_METHODS)
    if unknown_methods:
        raise ValueError(f"Unknown methods: {sorted(unknown_methods)}")
    if any(float(snr0) <= 0.0 for snr0 in snr0_levels):
        raise ValueError("SNR levels must be positive")
    if passive_patrol_period <= 0.0:
        raise ValueError("Passive patrol period must be positive")
    if not 0.0 < passive_patrol_y_scale <= 1.0:
        raise ValueError("Passive patrol y-scale must lie in (0, 1]")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    spec = BaselineScenarioSpec(
        output_dir=out_dir,
        success_radius=50.0,
        passive_patrol_period=float(passive_patrol_period),
        passive_patrol_y_scale=float(passive_patrol_y_scale),
    )

    all_rows: List[Dict[str, Any]] = []
    total = len(snr0_levels) * len(seeds) * len(methods)
    print(
        f"Greedy-EIG comparison: SNR0={list(snr0_levels)}, {len(seeds)} seeds, "
        f"methods={list(methods)}, {total} runs total",
        flush=True,
    )

    for snr0 in snr0_levels:
        for seed in seeds:
            for method in methods:
                mpath = _metrics_path(out_dir, snr0, seed, method)
                run_results_path = mpath.parent / "run_results.npz"
                cached = mpath.exists() and run_results_path.exists() and not overwrite
                if cached and method == "patrol" and not _cached_patrol_matches(
                    mpath,
                    expected_period=spec.passive_patrol_period,
                    expected_y_scale=spec.passive_patrol_y_scale,
                ):
                    cached = False
                    print(
                        f"Cached patrol configuration does not match the requested "
                        f"parameters for SNR0={snr0}, seed={seed}; rerunning.",
                        flush=True,
                    )
                if cached:
                    row = load_json_if_complete(
                        mpath,
                        ("method", "seed", "snr0_target", "final_error", "energy", "runtime_sec"),
                    )
                    cached = row is not None and (
                        row["method"] == method
                        and int(row["seed"]) == int(seed)
                        and np.isclose(float(row["snr0_target"]), float(snr0))
                    )
                    if cached:
                        all_rows.append(row)
                        print(f"Loaded SNR0={snr0}, seed={seed}, method={method}", flush=True)
                        continue
                    print(f"Cached metrics are incomplete for SNR0={snr0}, seed={seed}, method={method}; rerunning.", flush=True)

                scenario = _sample_scenario(seed=int(seed), snr0_target=float(snr0), spec=spec)
                print(
                    f"Running SNR0={snr0}, seed={seed}, method={method}, "
                    f"source=({scenario.source_pos[0]:.1f},{scenario.source_pos[1]:.1f})",
                    flush=True,
                )
                row = _run_method(method=str(method), scenario=scenario, spec=spec)
                all_rows.append(row)
                print(
                    f"  error={row['final_error']:.2f}, energy={row['energy']:.1f}, "
                    f"success={row['success']}, runtime={row['runtime_sec']:.1f}s",
                    flush=True,
                )

        _write_csv(out_path / "per_trial_results.csv", all_rows)
        _write_csv(out_path / "summary_results.csv", _aggregate(all_rows))

    print(f"Results saved to {out_path}", flush=True)
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline comparison: one-body HJB vs greedy EIG vs patrol")
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--snr0-levels", nargs="+", type=float, default=[50.0, 100.0, 150.0, 200.0])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--passive-patrol-period", "--patrol-period",
        dest="passive_patrol_period", type=float, default=40.0,
    )
    parser.add_argument(
        "--passive-patrol-y-scale", "--patrol-y-scale",
        dest="passive_patrol_y_scale", type=float, default=0.99,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seeds = list(range(int(args.seed_offset), int(args.seed_offset) + int(args.num_seeds)))
    run_comparison(
        out_dir=str(args.out_dir),
        seeds=seeds,
        snr0_levels=tuple(float(s) for s in args.snr0_levels),
        methods=tuple(str(m) for m in args.methods),
        overwrite=bool(args.overwrite),
        passive_patrol_period=float(args.passive_patrol_period),
        passive_patrol_y_scale=float(args.passive_patrol_y_scale),
    )
    print("\nRun  python3 plotting/plot_greedy_eig_comparison.py  to generate the figure.")


if __name__ == "__main__":
    main()
