"""
Paper experiment: §5.7 — team-size scaling M ∈ {2, 3, 4}.

Uses the pairwise HJB surrogate with the cooperative scenario.
The offline HJB solve is always 7-D (4 relative position + 3 belief axes),
independent of M; only the simulation and online policy evaluation grow with M
(linearly, via pairwise superposition). Pairwise planning is used here because one-body
suffers from a herding artefact: all drones follow the same single-drone policy
and cluster in the same region, causing error to increase with M.

Usage:
    python3 experiments/exp_scaling_M.py --num-seeds 50

Output (results/scaling_M/):
  per_trial_results.csv  — one row per (num_drones, seed)
  summary_results.csv    — stop rate, error, energy per drone aggregated by M

Run plots:
  python3 plotting/plot_scaling_M.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from drone_soc import (  # noqa: E402
    Belief,
    CircularNoFlyZone,
    Drone,
    DroneProblemState,
    Params,
    ProblemSpec,
    SigmoidCollisionAvoidance,
    SinusoidalPatrolling,
    find_pollutant_source,
)
from experiments.realistic_pairwise_showcase import (  # noqa: E402
    PRIMARY_TRUE_Q,
    PRIMARY_WIND,
    SOURCE_X_BOUNDS,
    SOURCE_Y_BOUNDS,
)
from fixed_point import AndersonAccelerationFixedPointSolver  # noqa: E402
from infinite_horizon import InfiniteHorizonTwoDronesPlanner  # noqa: E402
from experiments.fixed_point_diagnostics import fixed_point_metrics  # noqa: E402
from experiments.result_io import (  # noqa: E402
    load_json_if_complete,
    write_csv_atomic,
    write_json_atomic,
)
from measurement import SlicedScreenedAdvectionDiffusionPlume  # noqa: E402

DEFAULT_OUT_DIR = os.path.join(ROOT, "results", "scaling_M")

_DRONE_POSITIONS: dict[int, list[list[float]]] = {
    2: [[-220.0, 90.0], [-220.0, -90.0]],
    3: [[-220.0, 90.0], [-220.0, 0.0], [-220.0, -90.0]],
    4: [[-220.0, 90.0], [-220.0, 30.0], [-220.0, -30.0], [-220.0, -90.0]],
}


@dataclass(frozen=True)
class Scenario:
    seed: int
    source_pos: np.ndarray
    q: float


def _sample_scenario(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    source_pos = np.array(
        [rng.uniform(*SOURCE_X_BOUNDS), rng.uniform(*SOURCE_Y_BOUNDS)], dtype=float
    )
    return Scenario(seed=int(seed), source_pos=source_pos, q=float(PRIMARY_TRUE_Q))


def _build_case(
    *,
    scenario: Scenario,
    num_drones: int,
    discount: float,
    out_dir: str,
    t_end: float = 600.0,
    dt_meas: float = 30.0,
    dx: float = 60.0,
    dx_secondary: float = 120.0,
    num_belief_discr: int = 3,
    fp_max_iter: int = 12,
):
    wind = PRIMARY_WIND.copy()
    domain = np.array([[-360.0, 360.0], [-360.0, 360.0]], dtype=float)
    v_max = 10.0
    noise_std = 2e-5

    par = Params(
        meas_times=np.linspace(0.0, t_end, int(t_end / dt_meas) + 1),
        t_end=t_end,
        dt=0.1,
        dt_hjb=1.0,
        dx=dx,
        dx_secondary=dx_secondary,
        num_belief_discr=num_belief_discr,
        v_max=v_max,
        noise_std=noise_std,
        dyn_noise_std=0.3,
        tol_sigma=40.0,
        tol_residual=8.0,
        tol_q_log_var=2.0,
        discount=float(discount),
        save_videos=False,
        update_mode="full_batch",
        output_dir=out_dir,
        stop_when_confident=True,
        min_stop_updates=6,
        confident_streak=2,
        seed=int(scenario.seed),
        verbose_planning=False,
    )

    pollutant = SlicedScreenedAdvectionDiffusionPlume(
        source=np.array(scenario.source_pos, copy=True),
        q=float(scenario.q),
        ds=90.0,
        tau_s=2600.0,
        wind=wind,
        sigma_e=noise_std,
        eps_c=22.0,
    )

    no_fly_zones = [
        CircularNoFlyZone(center=np.array([-95.0, 0.0]), radius=55.0, smooth_width=40.0, max_value=1.0),
        CircularNoFlyZone(center=np.array([35.0, 120.0]), radius=50.0, smooth_width=40.0, max_value=1.0),
        CircularNoFlyZone(center=np.array([35.0, -120.0]), radius=50.0, smooth_width=40.0, max_value=1.0),
    ]
    patrol = SinusoidalPatrolling(wind=wind, v_max=v_max, period=220.0, y_scale=0.82)
    collision = SigmoidCollisionAvoidance(d_min=45.0, d_max=90.0, max_value=80.0, confidence=0.99)

    problem = ProblemSpec(
        pollutant=pollutant,
        domain=domain,
        wind=wind,
        no_fly_zones=no_fly_zones,
        c1=1e-4,
        c2=1e-5,
        c3=1e-2,
        tau_col=1.0,
        c4=1.8,
        patrol=patrol,
        collision=collision,
    )

    drone_positions = np.array(_DRONE_POSITIONS[num_drones], dtype=float)
    prior = Belief(
        mu=np.array([255.0, 0.0], dtype=float),
        sigma=np.diag([90.0**2, 110.0**2]),
        q_log=float(np.log(45.0)),
        q_log_var=1.0,
        q_log_bounds=(float(np.log(10.0)), float(np.log(500.0))),
    )
    state = DroneProblemState(
        prior=prior.copy(),
        belief=prior.copy(),
        drones=[Drone(x0=pos, v_max=par.v_max, patrol_dir_fn=problem.patrol) for pos in drone_positions],
    )
    fp_solver = AndersonAccelerationFixedPointSolver(m=3, tol=1e-3, max_iter=int(fp_max_iter), verbose=False)
    return par, problem, state, fp_solver


def _is_confidence_stop(stop_reason: Any) -> bool:
    return str(stop_reason).startswith("confidence_criterion_met")


def _cumulative_distance(traj: np.ndarray) -> float:
    diffs = np.diff(traj, axis=-1)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _run_case(
    *,
    scenario: Scenario,
    num_drones: int,
    discount: float,
    out_dir: str,
) -> Dict[str, Any]:
    par, problem, state, fp_solver = _build_case(
        scenario=scenario,
        num_drones=num_drones,
        discount=discount,
        out_dir=out_dir,
    )
    planner = InfiniteHorizonTwoDronesPlanner(wind_aligned=True, fp_solver=fp_solver)

    t0 = time.perf_counter()
    traj, log = find_pollutant_source(par, problem, state, planner=planner)
    runtime_sec = float(time.perf_counter() - t0)

    final_belief = log["beliefs"][-1]
    final_error = float(np.linalg.norm(final_belief.mu - problem.pollutant.source))
    final_trace = float(np.trace(final_belief.sigma))
    energy_total = float(log.get("energy_total", 0.0))
    travel_distance = float(_cumulative_distance(traj))
    stop_reason = str(log.get("stop_reason", "reached_time_horizon"))

    row = {
        "num_drones": int(num_drones),
        "seed": int(scenario.seed),
        "discount": float(discount),
        "stopped": int(_is_confidence_stop(stop_reason)),
        "success_5m": int(final_error <= 5.0),
        "success_10m": int(final_error <= 10.0),
        "final_error": float(final_error),
        "final_trace_sigma": float(final_trace),
        "energy_total": float(energy_total),
        "energy_per_drone": float(energy_total / num_drones),
        "travel_distance": float(travel_distance),
        "travel_distance_per_drone": float(travel_distance / num_drones),
        "runtime_sec": float(runtime_sec),
        "stop_reason": stop_reason,
    }
    row.update(fixed_point_metrics(fp_solver, out_dir))
    return row


def _aggregate(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_M: dict[int, list[Dict[str, Any]]] = {}
    for row in rows:
        by_M.setdefault(int(row["num_drones"]), []).append(row)

    summary: List[Dict[str, Any]] = []
    for M, group in sorted(by_M.items()):
        errors = np.array([float(g["final_error"]) for g in group], dtype=float)
        energies_per_drone = np.array([float(g["energy_per_drone"]) for g in group], dtype=float)
        traces = np.array([float(g["final_trace_sigma"]) for g in group], dtype=float)
        summary.append(
            {
                "num_drones": M,
                "num_trials": len(group),
                "stop_rate": float(np.mean([g["stopped"] for g in group])),
                "success_5m_rate": float(np.mean([g["success_5m"] for g in group])),
                "success_10m_rate": float(np.mean([g["success_10m"] for g in group])),
                "median_final_error": float(np.median(errors)),
                "q10_final_error": float(np.quantile(errors, 0.1)),
                "q90_final_error": float(np.quantile(errors, 0.9)),
                "median_trace_sigma": float(np.median(traces)),
                "median_energy_per_drone": float(np.median(energies_per_drone)),
                "q10_energy_per_drone": float(np.quantile(energies_per_drone, 0.1)),
                "q90_energy_per_drone": float(np.quantile(energies_per_drone, 0.9)),
                "median_runtime_sec": float(np.median([g["runtime_sec"] for g in group])),
            }
        )
    return summary


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    write_csv_atomic(path, rows)


def run_scaling_study(
    *,
    out_dir: str,
    seeds: Sequence[int],
    M_values: Sequence[int],
    discount: float,
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    if not seeds:
        raise ValueError("At least one seed is required")
    if not M_values:
        raise ValueError("At least one team size is required")
    unsupported = set(M_values) - set(_DRONE_POSITIONS)
    if unsupported:
        raise ValueError(f"No drone positions are defined for team sizes {sorted(unsupported)}")
    if len(set(M_values)) != len(M_values):
        raise ValueError("Team sizes must be unique")
    if discount <= 0.0:
        raise ValueError("Discount rate must be positive")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []

    total = len(M_values) * len(seeds)
    print(
        f"Scaling study: M={list(M_values)}, {len(seeds)} seeds each, {total} total runs",
        flush=True,
    )

    for M in M_values:
        for seed in seeds:
            case_dir = str(out_path / f"M{M}" / f"seed_{seed:03d}")
            os.makedirs(case_dir, exist_ok=True)
            metrics_path = Path(case_dir) / "metrics.json"
            if metrics_path.exists() and not overwrite:
                row = load_json_if_complete(
                    metrics_path,
                    ("num_drones", "seed", "final_error", "energy_per_drone", "runtime_sec"),
                )
                if row is not None and int(row["num_drones"]) == int(M) and int(row["seed"]) == int(seed):
                    all_rows.append(row)
                    print(
                        f"Loaded M={M}, seed={seed}: error={row['final_error']:.2f}, "
                        f"energy/drone={row['energy_per_drone']:.1f}",
                        flush=True,
                    )
                    continue
                print(f"Cached metrics are incomplete for M={M}, seed={seed}; rerunning.", flush=True)

            scenario = _sample_scenario(seed)
            print(
                f"Running M={M}, seed={seed}, source=({scenario.source_pos[0]:.1f}, {scenario.source_pos[1]:.1f})",
                flush=True,
            )
            row = _run_case(
                scenario=scenario,
                num_drones=int(M),
                discount=float(discount),
                out_dir=case_dir,
            )
            write_json_atomic(metrics_path, row)
            all_rows.append(row)
            print(
                f"  M={M}, seed={seed}: error={row['final_error']:.2f}, "
                f"energy/drone={row['energy_per_drone']:.1f}, "
                f"stopped={row['stopped']}, runtime={row['runtime_sec']:.1f}s",
                flush=True,
            )

        _write_csv(out_path / "per_trial_results.csv", all_rows)
        _write_csv(out_path / "summary_results.csv", _aggregate(all_rows))

    print(f"Saved results to {out_path}", flush=True)
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-M scaling experiment for the pairwise planner")
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--M-values", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--discount", type=float, default=0.3)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seeds = list(range(int(args.seed_offset), int(args.seed_offset) + int(args.num_seeds)))
    run_scaling_study(
        out_dir=str(args.out_dir),
        seeds=seeds,
        M_values=tuple(int(m) for m in args.M_values),
        discount=float(args.discount),
        overwrite=bool(args.overwrite),
    )
    print("\nRun  python3 plotting/plot_scaling_M.py  to generate the figure.")


if __name__ == "__main__":
    main()
