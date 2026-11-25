"""
Core library for the pairwise-planning showcase scenario.

Provides scenario sampling, case construction, metrics collection, and
aggregation functions shared by exp_paper_pairwise_showcase.py and
exp_paper_discount_sweep.py.

When run directly, performs a single-seed diagnostic followed by a mini-pilot,
which is useful for verifying scenario health before a full paper-scale run.

Usage:
    python3 experiments/exp_pairwise_showcase.py --skip-pilot
    python3 experiments/exp_pairwise_showcase.py --force-pilot --pilot-seeds 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from drone_soc import find_pollutant_source  # noqa: E402
from experiments.realistic_pairwise_showcase import (  # noqa: E402
    SOURCE_X_BOUNDS,
    SOURCE_Y_BOUNDS,
    RealisticPairwiseShowcaseSpec,
    make_realistic_pairwise_showcase,
)
from infinite_horizon import InfiniteHorizonSingleDronePlanner, InfiniteHorizonTwoDronesPlanner  # noqa: E402
from experiments.fixed_point_diagnostics import fixed_point_metrics  # noqa: E402
from experiments.result_io import write_csv_atomic, write_json_atomic  # noqa: E402
from _drone_soc_internal import spatial_grid_resolution_metadata  # noqa: E402
from plotting.plotting import (  # noqa: E402
    default_after_measurement_callback,
    plot_belief_convergence_with_intensity,
    plot_belief_distribution_ellipses,
    plot_confidence_ellipses_with_wind,
    plot_path,
    plot_running_cost_breakdown,
)


DEFAULT_METHODS = ("order1", "order2")
SUCCESS_THRESHOLDS = (1.0, 2.0, 5.0, 10.0)


@dataclass(frozen=True)
class Scenario:
    seed: int
    source_pos: np.ndarray
    q: float


@dataclass(frozen=True)
class RuntimeOverrides:
    t_end: float = 600.0
    dt_meas: float = 30.0
    dx: float = 60.0
    dx_secondary: float = 120.0
    num_belief_discr: int = 4
    fp_max_iter: int = 25
    dt_hjb: float | None = None  # if set, overrides dt_hjb from make_realistic_pairwise_showcase


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    write_csv_atomic(path, rows)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    write_json_atomic(path, payload)


def _variant_true_q(variant: str) -> float:
    spec = RealisticPairwiseShowcaseSpec(variant=variant)
    par, problem, _, _ = make_realistic_pairwise_showcase(spec)
    del par
    return float(problem.pollutant.q)


def _sample_scenario(seed: int, variant: str) -> Scenario:
    rng = np.random.default_rng(seed)
    source_pos = np.array(
        [
            rng.uniform(*SOURCE_X_BOUNDS),
            rng.uniform(*SOURCE_Y_BOUNDS),
        ],
        dtype=float,
    )
    return Scenario(seed=int(seed), source_pos=source_pos, q=_variant_true_q(variant))


def _pairwise_min_distance(traj: np.ndarray) -> float:
    min_d = np.inf
    for k in range(traj.shape[-1]):
        pos = traj[:, :, k]
        for i in range(pos.shape[0]):
            for j in range(i + 1, pos.shape[0]):
                min_d = min(min_d, float(np.linalg.norm(pos[i] - pos[j])))
    return float(min_d)


def _cumulative_distance(traj: np.ndarray) -> float:
    diffs = np.diff(traj, axis=-1)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _boundary_summary(traj: np.ndarray, domain: np.ndarray, atol: float = 1e-8) -> tuple[int, float]:
    lower = domain[:, 0][None, :, None]
    upper = domain[:, 1][None, :, None]
    on_edge = np.any(np.isclose(traj, lower, atol=atol) | np.isclose(traj, upper, atol=atol), axis=1)
    count = int(np.count_nonzero(on_edge))
    fraction = float(count / max(1, on_edge.size))
    return count, fraction


def _min_no_fly_margin(traj: np.ndarray, problem) -> float:
    if not problem.no_fly_zones:
        return float("nan")
    positions = traj.reshape(traj.shape[0], 2, -1).transpose(0, 2, 1).reshape(-1, 2)
    margins = []
    for zone in problem.no_fly_zones:
        d = np.linalg.norm(positions - np.asarray(zone.center, dtype=float)[None, :], axis=1)
        margins.append(float(np.min(d - float(zone.radius))))
    return float(np.min(margins))


def _is_confidence_stop(stop_reason: Any) -> bool:
    return str(stop_reason).startswith("confidence_criterion_met")


def _first_informative_measurement_time(
    *,
    log: Dict[str, Any],
    par: Params,
    problem: ProblemSpec,
    snr_threshold: float | None,
) -> float:
    if snr_threshold is None:
        return float("nan")
    for t, positions in zip(
        log.get("measurement_times", []),
        log.get("measurement_positions", []),
    ):
        concentrations = np.asarray(
            problem.pollutant.get_pollutant_concentration(
                np.asarray(positions, dtype=float),
                float(t),
                theta=np.asarray(problem.pollutant.source, dtype=float),
                intensity=float(problem.pollutant.q),
            ),
            dtype=float,
        )
        if float(np.max(np.abs(concentrations)) / par.noise_std) >= snr_threshold:
            return float(t)
    return float("nan")


def _build_case(
    *,
    scenario: Scenario,
    discount: float,
    out_dir: str,
    variant: str,
    runtime: RuntimeOverrides,
):
    spec = RealisticPairwiseShowcaseSpec(
        discount=float(discount),
        output_dir=out_dir,
        variant=variant,
        t_end=float(runtime.t_end),
        dt_meas=float(runtime.dt_meas),
        dx=float(runtime.dx),
        dx_secondary=float(runtime.dx_secondary),
        num_belief_discr=int(runtime.num_belief_discr),
        fp_max_iter=int(runtime.fp_max_iter),
    )
    par, problem, state, fp_solver = make_realistic_pairwise_showcase(spec)
    if runtime.dt_hjb is not None:
        par.dt_hjb = float(runtime.dt_hjb)
    problem.pollutant.source = np.array(scenario.source_pos, copy=True)
    problem.pollutant.q = float(scenario.q)
    par.seed = int(scenario.seed)
    par.output_dir = out_dir
    par.save_videos = False
    par.verbose_planning = False
    if hasattr(fp_solver, "verbose"):
        fp_solver.verbose = False
    return par, problem, state, fp_solver


def _initial_condition_summary(
    *,
    seed: int,
    discount: float,
    out_dir: str,
    variant: str,
    runtime: RuntimeOverrides,
) -> Dict[str, Any]:
    scenario = _sample_scenario(seed, variant)
    par, problem, state, _ = _build_case(
        scenario=scenario,
        discount=discount,
        out_dir=out_dir,
        variant=variant,
        runtime=runtime,
    )
    starts = np.asarray(state.drone_positions, dtype=float)
    concentrations = np.asarray(
        problem.pollutant.get_pollutant_concentration(
            starts,
            0.0,
            theta=np.asarray(problem.pollutant.source, dtype=float),
            intensity=float(problem.pollutant.q),
        ),
        dtype=float,
    )
    snr_linear = concentrations / float(par.noise_std)
    snr_db = 20.0 * np.log10(np.maximum(snr_linear, 1e-30))
    inside = np.all(
        (starts[:, 0] > problem.domain[0, 0])
        & (starts[:, 0] < problem.domain[0, 1])
        & (starts[:, 1] > problem.domain[1, 0])
        & (starts[:, 1] < problem.domain[1, 1])
    )
    summary = {
        "variant": variant,
        "seed": int(seed),
        "source_x": float(scenario.source_pos[0]),
        "source_y": float(scenario.source_pos[1]),
        "q": float(problem.pollutant.q),
        "domain": np.asarray(problem.domain, dtype=float).tolist(),
        "source_box_x": list(SOURCE_X_BOUNDS),
        "source_box_y": list(SOURCE_Y_BOUNDS),
        "prior_mu": np.asarray(state.prior.mu, dtype=float).tolist(),
        "prior_sigma": np.asarray(state.prior.sigma, dtype=float).tolist(),
        "prior_q_median": float(np.exp(state.prior.q_log)) if state.prior.q_log is not None else None,
        "prior_q_log_var": float(state.prior.q_log_var) if state.prior.q_log_var is not None else None,
        "wind": np.asarray(problem.wind, dtype=float).tolist(),
        "drone_starts": starts.tolist(),
        "initial_concentration": concentrations.tolist(),
        "initial_snr_linear": snr_linear.tolist(),
        "initial_snr_db": snr_db.tolist(),
        "all_initial_drones_strictly_inside_domain": bool(inside),
        "t_end": float(par.t_end),
        "dt_meas": float(runtime.dt_meas),
        "dx": float(par.dx),
        "dx_secondary": float(par.dx_secondary),
        "num_belief_discr": int(par.num_belief_discr),
        "fp_max_iter": int(runtime.fp_max_iter),
        "no_fly_zones": [
            {"center": np.asarray(zone.center, dtype=float).tolist(), "radius": float(zone.radius)}
            for zone in (problem.no_fly_zones or [])
        ],
    }
    _write_json(os.path.join(out_dir, "initial_conditions.json"), summary)
    return summary


def _run_method(
    *,
    scenario: Scenario,
    discount: float,
    method: str,
    out_dir: str,
    variant: str,
    make_plots: bool,
    make_contours: bool,
    runtime: RuntimeOverrides,
    informative_snr_threshold: float | None = None,
) -> Dict[str, Any]:
    par, problem, state, fp_solver = _build_case(
        scenario=scenario,
        discount=discount,
        out_dir=out_dir,
        variant=variant,
        runtime=runtime,
    )
    if method == "order1":
        planner = InfiniteHorizonSingleDronePlanner(wind_aligned=True, fp_solver=fp_solver)
    elif method == "order2":
        planner = InfiniteHorizonTwoDronesPlanner(wind_aligned=True, fp_solver=fp_solver)
    else:
        raise ValueError(f"Unknown method: {method}")

    t0 = time.perf_counter()
    traj, log = find_pollutant_source(
        par,
        problem,
        state,
        planner=planner,
        on_after_measurement=default_after_measurement_callback if make_contours else None,
    )
    runtime_sec = float(time.perf_counter() - t0)

    if make_plots:
        plot_path(traj, log, par, problem, state)
        plot_belief_convergence_with_intensity(log, par, problem, out_dir=out_dir)
        plot_belief_distribution_ellipses(log, par, problem, out_dir=out_dir)
        plot_confidence_ellipses_with_wind(log, par, problem, out_dir=out_dir)
        plot_running_cost_breakdown(traj, log, par, problem, drone_idx=None, out_dir=out_dir)

    final_belief = log["beliefs"][-1]
    final_error = float(np.linalg.norm(final_belief.mu - problem.pollutant.source))
    final_trace_sigma = float(np.trace(final_belief.sigma))
    final_q = float(np.exp(final_belief.q_log)) if final_belief.q_log is not None else float("nan")
    final_q_abs_error = float(abs(final_q - problem.pollutant.q)) if np.isfinite(final_q) else float("nan")
    final_q_log_var = float(final_belief.q_log_var) if final_belief.q_log_var is not None else float("nan")
    residual_scores = log.get("residual_scores", [])
    final_residual = float(residual_scores[-1]) if len(residual_scores) else float("nan")
    fp_metrics = fixed_point_metrics(fp_solver, out_dir)
    time_to_informative = _first_informative_measurement_time(
        log=log,
        par=par,
        problem=problem,
        snr_threshold=informative_snr_threshold,
    )

    min_pair_distance = _pairwise_min_distance(traj)
    collision_limit = float(problem.collision.d_min) if problem.collision is not None else 0.0
    min_no_fly_margin = _min_no_fly_margin(traj, problem)
    boundary_touch_count, boundary_dwell_fraction = _boundary_summary(traj, np.asarray(problem.domain, dtype=float))
    stop_reason = str(log.get("stop_reason", "reached_time_horizon"))

    row = {
        "variant": variant,
        "discount": float(discount),
        "method": method,
        "seed": int(scenario.seed),
        "source_x": float(problem.pollutant.source[0]),
        "source_y": float(problem.pollutant.source[1]),
        "q": float(problem.pollutant.q),
        "initial_q_median": float(np.exp(state.prior.q_log)) if state.prior.q_log is not None else float("nan"),
        "t_end": float(par.t_end),
        "dt_meas": float(runtime.dt_meas),
        "dx": float(par.dx),
        "dx_secondary": float(par.dx_secondary),
        "num_belief_discr": int(par.num_belief_discr),
        "fp_max_iter": int(runtime.fp_max_iter),
        "stopped": int(_is_confidence_stop(stop_reason)),
        "success_1m": int(final_error <= 1.0),
        "success_2m": int(final_error <= 2.0),
        "success_5m": int(final_error <= 5.0),
        "success_10m": int(final_error <= 10.0),
        "safe": int(min_pair_distance >= collision_limit),
        "collision_failure": int(min_pair_distance < collision_limit),
        "min_pair_distance": float(min_pair_distance),
        "min_no_fly_margin": float(min_no_fly_margin),
        "no_fly_violation": int(np.isfinite(min_no_fly_margin) and min_no_fly_margin < 0.0),
        "boundary_touch_count": int(boundary_touch_count),
        "boundary_dwell_fraction": float(boundary_dwell_fraction),
        "travel_distance": float(_cumulative_distance(traj)),
        "energy": float(log.get("energy_total", 0.0)),
        "runtime_sec": float(runtime_sec),
        "stop_time": float(log["times"][-1]) if len(log["times"]) else float(par.t_end),
        "final_error": float(final_error),
        "final_trace_sigma": float(final_trace_sigma),
        "final_residual": float(final_residual),
        "informative_snr_threshold": (
            float(informative_snr_threshold)
            if informative_snr_threshold is not None
            else float("nan")
        ),
        "time_to_first_informative_measurement": float(time_to_informative),
        "final_q": float(final_q),
        "final_q_abs_error": float(final_q_abs_error),
        "final_q_log_var": float(final_q_log_var),
        "stop_reason": stop_reason,
    }
    grid_meta = spatial_grid_resolution_metadata(par, problem)
    row.update({
        "primary_grid_nodes_x": int(grid_meta["primary"]["node_counts"][0]),
        "primary_grid_nodes_y": int(grid_meta["primary"]["node_counts"][1]),
        "primary_grid_spacing_x": float(grid_meta["primary"]["actual_spacings"][0]),
        "primary_grid_spacing_y": float(grid_meta["primary"]["actual_spacings"][1]),
        "secondary_grid_nodes_x": int(grid_meta["secondary"]["node_counts"][0]),
        "secondary_grid_nodes_y": int(grid_meta["secondary"]["node_counts"][1]),
        "secondary_grid_spacing_x": float(grid_meta["secondary"]["actual_spacings"][0]),
        "secondary_grid_spacing_y": float(grid_meta["secondary"]["actual_spacings"][1]),
    })
    row.update(fp_metrics)
    _write_json(os.path.join(out_dir, "metrics.json"), row)
    return row


def _aggregate(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    summary: List[Dict[str, Any]] = []
    for method, group in sorted(by_method.items()):
        errors = np.array([float(g["final_error"]) for g in group], dtype=float)
        traces = np.array([float(g["final_trace_sigma"]) for g in group], dtype=float)
        summary.append(
            {
                "method": method,
                "num_trials": len(group),
                "stop_rate": float(np.mean([g["stopped"] for g in group])),
                "success_1m_rate": float(np.mean([g["success_1m"] for g in group])),
                "success_2m_rate": float(np.mean([g["success_2m"] for g in group])),
                "success_5m_rate": float(np.mean([g["success_5m"] for g in group])),
                "success_10m_rate": float(np.mean([g["success_10m"] for g in group])),
                "safe_rate": float(np.mean([g["safe"] for g in group])),
                "no_fly_violation_rate": float(np.mean([g["no_fly_violation"] for g in group])),
                "median_boundary_dwell_fraction": float(np.median([g["boundary_dwell_fraction"] for g in group])),
                "max_boundary_dwell_fraction": float(np.max([g["boundary_dwell_fraction"] for g in group])),
                "median_final_error": float(np.median(errors)),
                "q90_final_error": float(np.quantile(errors, 0.9)),
                "median_trace_sigma": float(np.median(traces)),
                "q90_trace_sigma": float(np.quantile(traces, 0.9)),
                "median_q_abs_error": float(np.median([g["final_q_abs_error"] for g in group])),
                "median_energy": float(np.median([g["energy"] for g in group])),
                "median_travel_distance": float(np.median([g["travel_distance"] for g in group])),
                "median_runtime_sec": float(np.median([g["runtime_sec"] for g in group])),
            }
        )
    return summary


def _diagnostic_is_credible(rows: Sequence[Dict[str, Any]], initial_summary: Dict[str, Any]) -> bool:
    """Return True when the single-seed diagnostic passes basic sanity checks.

    Checks: initial SNR in a reasonable range, no collision or no-fly
    violations, negligible boundary dwell, and pairwise planning no worse than one-body planning.
    """
    by_method = {str(row["method"]): row for row in rows}
    order1 = by_method.get("order1")
    order2 = by_method.get("order2")
    if order1 is None or order2 is None:
        return False
    snr = np.asarray(initial_summary["initial_snr_linear"], dtype=float)
    snr_ok = bool(np.min(snr) >= 0.4 and np.max(snr) <= 2.0)
    safe_ok = all(int(row["safe"]) and not int(row["no_fly_violation"]) for row in rows)
    boundary_ok = all(float(row["boundary_dwell_fraction"]) < 0.02 for row in rows)
    order2_not_worse = bool(
        float(order2["final_error"]) <= float(order1["final_error"])
        or float(order2["final_trace_sigma"]) <= float(order1["final_trace_sigma"])
    )
    energy_ok = float(order2["energy"]) <= 1.5 * max(float(order1["energy"]), 1e-12)
    return bool(snr_ok and safe_ok and boundary_ok and order2_not_worse and energy_ok)


def _run_diagnostic(
    *,
    seed: int,
    discount: float,
    out_dir: str,
    variant: str,
    methods: Sequence[str],
    make_plots: bool,
    make_contours: bool,
    runtime: RuntimeOverrides,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    scenario = _sample_scenario(seed, variant)
    for method in methods:
        method_dir = os.path.join(out_dir, variant, f"seed_{seed:03d}", method)
        os.makedirs(method_dir, exist_ok=True)
        print(
            f"Running diagnostic variant={variant}, method={method}, seed={seed}, "
            f"source=({scenario.source_pos[0]:.1f}, {scenario.source_pos[1]:.1f}), q={scenario.q:.1f}",
            flush=True,
        )
        row = _run_method(
            scenario=scenario,
            discount=discount,
            method=method,
            out_dir=method_dir,
            variant=variant,
            make_plots=make_plots,
            make_contours=make_contours,
            runtime=runtime,
        )
        rows.append(row)
        print(
            f"  Result: error={row['final_error']:.2f}, trace={row['final_trace_sigma']:.2f}, "
            f"q={row['final_q']:.2f}, energy={row['energy']:.2f}, "
            f"boundary_dwell={100.0 * row['boundary_dwell_fraction']:.2f}%, "
            f"no_fly_margin={row['min_no_fly_margin']:.2f}, reason={row['stop_reason']}",
            flush=True,
        )
    _write_csv(os.path.join(out_dir, variant, "diagnostic_per_trial_results.csv"), rows)
    _write_csv(os.path.join(out_dir, variant, "diagnostic_summary_results.csv"), _aggregate(rows))
    return rows


def _run_pilot(
    *,
    num_seeds: int,
    discount: float,
    out_dir: str,
    variant: str,
    runtime: RuntimeOverrides,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in range(num_seeds):
        scenario = _sample_scenario(seed, variant)
        for method in DEFAULT_METHODS:
            method_dir = os.path.join(out_dir, variant, "mini_pilot", f"seed_{seed:03d}", method)
            os.makedirs(method_dir, exist_ok=True)
            print(
                f"Running pilot variant={variant}, method={method}, seed={seed}, "
                f"source=({scenario.source_pos[0]:.1f}, {scenario.source_pos[1]:.1f}), q={scenario.q:.1f}",
                flush=True,
            )
            row = _run_method(
                scenario=scenario,
                discount=discount,
                method=method,
                out_dir=method_dir,
                variant=variant,
                make_plots=False,
                make_contours=False,
                runtime=runtime,
            )
            rows.append(row)
            _write_csv(os.path.join(out_dir, variant, "mini_pilot", "per_trial_results.csv"), rows)
            _write_csv(os.path.join(out_dir, variant, "mini_pilot", "summary_results.csv"), _aggregate(rows))
            print(
                f"  Result: error={row['final_error']:.2f}, trace={row['final_trace_sigma']:.2f}, "
                f"energy={row['energy']:.2f}, boundary_dwell={100.0 * row['boundary_dwell_fraction']:.2f}%",
                flush=True,
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the pairwise-planning showcase scenario")
    parser.add_argument("--seed", type=int, default=0, help="Diagnostic seed")
    parser.add_argument("--discount", type=float, default=0.3)
    parser.add_argument(
        "--variant",
        choices=("primary", "fallback", "collision_tuned", "prior_tuned"),
        default="primary",
    )
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "results", "pairwise_showcase"))
    parser.add_argument("--pilot-seeds", type=int, default=10)
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--t-end", type=float, default=600.0)
    parser.add_argument("--dt-meas", type=float, default=30.0)
    parser.add_argument("--dx", type=float, default=60.0)
    parser.add_argument("--dx-secondary", type=float, default=120.0)
    parser.add_argument("--num-belief-discr", type=int, default=4)
    parser.add_argument("--fp-max-iter", type=int, default=25)
    parser.add_argument("--skip-pilot", action="store_true")
    parser.add_argument("--pilot-only", action="store_true", help="Skip diagnostic and run the mini-pilot directly")
    parser.add_argument("--force-pilot", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Skip diagnostic PDF plot suite")
    parser.add_argument("--contours", action="store_true", help="Save prior/likelihood/posterior contours")
    args = parser.parse_args()

    base_out = os.path.abspath(args.out_dir)
    variant_out = os.path.join(base_out, args.variant)
    os.makedirs(variant_out, exist_ok=True)
    runtime = RuntimeOverrides(
        t_end=float(args.t_end),
        dt_meas=float(args.dt_meas),
        dx=float(args.dx),
        dx_secondary=float(args.dx_secondary),
        num_belief_discr=int(args.num_belief_discr),
        fp_max_iter=int(args.fp_max_iter),
    )

    initial_summary = _initial_condition_summary(
        seed=int(args.seed),
        discount=float(args.discount),
        out_dir=variant_out,
        variant=str(args.variant),
        runtime=runtime,
    )
    print(
        "Initial condition:",
        f"variant={args.variant}",
        f"source=({initial_summary['source_x']:.1f}, {initial_summary['source_y']:.1f})",
        f"q={initial_summary['q']:.1f}",
        f"wind={initial_summary['wind']}",
        flush=True,
    )
    for idx, (conc, snr_lin, snr_db) in enumerate(
        zip(
            initial_summary["initial_concentration"],
            initial_summary["initial_snr_linear"],
            initial_summary["initial_snr_db"],
        ),
        start=1,
    ):
        print(f"  drone {idx}: concentration={conc:.6g}, snr={snr_lin:.3f}, snr_db={snr_db:.2f}", flush=True)

    diagnostic_ok = False
    if not args.pilot_only:
        diagnostic_rows = _run_diagnostic(
            seed=int(args.seed),
            discount=float(args.discount),
            out_dir=base_out,
            variant=str(args.variant),
            methods=tuple(str(m) for m in args.methods),
            make_plots=not bool(args.no_plots),
            make_contours=bool(args.contours),
            runtime=runtime,
        )
        diagnostic_ok = _diagnostic_is_credible(diagnostic_rows, initial_summary)
        print(f"Diagnostic gate: {'passed' if diagnostic_ok else 'failed'}", flush=True)
    if not args.skip_pilot and (args.pilot_only or diagnostic_ok or args.force_pilot):
        _run_pilot(
            num_seeds=int(args.pilot_seeds),
            discount=float(args.discount),
            out_dir=base_out,
            variant=str(args.variant),
            runtime=runtime,
        )
    elif not args.skip_pilot:
        print("Skipping mini-pilot: diagnostic gate failed. Use --force-pilot to run it anyway.", flush=True)


if __name__ == "__main__":
    main()
