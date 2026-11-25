"""Generate the standard diagnostic plot suite for one saved showcase seed.

The simulation itself is not rerun.  The script reconstructs the log objects
from ``run_results.npz`` and then calls the existing plotting utilities in
``plotting.py`` for visual consistency with the rest of the paper figures.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from _drone_soc_internal import covariance_from_eigen  # noqa: E402
from drone_soc import Belief  # noqa: E402
from experiments.exp_pairwise_showcase import RuntimeOverrides, _build_case, _sample_scenario  # noqa: E402
from plotting.plotting import (  # noqa: E402
    plot_belief_convergence_with_intensity,
    plot_belief_distribution_ellipses,
    plot_confidence_ellipses_with_wind,
    plot_drone_paths,
    plot_running_cost_breakdown,
)


_REQUIRED_RESULT_ARRAYS = {
    "controls",
    "eig_angle",
    "eig_vals_log",
    "energy_cumulative_by_cycle",
    "energy_cycles",
    "energy_steps",
    "meas_times",
    "mus",
    "q_log_vars",
    "q_logs",
    "residual_scores",
    "times",
    "traj",
}


def _load_results(path: Path) -> np.lib.npyio.NpzFile:
    data = np.load(path, allow_pickle=True)
    missing = _REQUIRED_RESULT_ARRAYS - set(data.files)
    if missing:
        data.close()
        raise ValueError(f"Saved run results are missing arrays {sorted(missing)}: {path}")
    return data


def _reconstruct_log(data: np.lib.npyio.NpzFile, q_log_bounds) -> dict:
    mus = np.asarray(data["mus"], dtype=float)
    eig_vals_log = np.asarray(data["eig_vals_log"], dtype=float)
    eig_angle = np.asarray(data["eig_angle"], dtype=float)
    q_logs = np.asarray(data["q_logs"], dtype=float)
    q_log_vars = np.asarray(data["q_log_vars"], dtype=float)

    beliefs = []
    for k in range(len(mus)):
        sigma = covariance_from_eigen(eig_vals_log[k, 0], eig_vals_log[k, 1], eig_angle[k])
        q_log = None if k >= len(q_logs) or not np.isfinite(q_logs[k]) else float(q_logs[k])
        q_log_var = None if k >= len(q_log_vars) or not np.isfinite(q_log_vars[k]) else float(q_log_vars[k])
        beliefs.append(
            Belief(
                mu=np.asarray(mus[k], dtype=float),
                sigma=np.asarray(sigma, dtype=float),
                q_log=q_log,
                q_log_var=q_log_var,
                q_log_bounds=q_log_bounds,
            )
        )

    return {
        "times": np.asarray(data["times"], dtype=float),
        "controls": [u for u in np.asarray(data["controls"], dtype=float)],
        "beliefs": beliefs,
        "residual_scores": list(np.asarray(data["residual_scores"], dtype=float)),
        "energy_steps": list(np.asarray(data["energy_steps"], dtype=float)),
        "energy_cycles": list(np.asarray(data["energy_cycles"], dtype=float)),
        "energy_cumulative_by_cycle": list(np.asarray(data["energy_cumulative_by_cycle"], dtype=float)),
        "energy_total": float(np.sum(np.asarray(data["energy_steps"], dtype=float))),
    }


def _downsample_path_inputs(traj: np.ndarray, log: dict, max_points: int = 1200) -> tuple[np.ndarray, dict]:
    """Downsample only the dense path plot, keeping measurement/belief plots exact."""
    n = traj.shape[-1]
    if n <= max_points:
        return traj, log
    idx = np.unique(np.linspace(0, n - 1, max_points, dtype=int))
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    log_path = dict(log)
    log_path["times"] = np.asarray(log["times"], dtype=float)[idx]
    return traj[:, :, idx], log_path


def _make_suite(
    *,
    seed: int,
    method: str,
    discount: float,
    variant: str,
    results_dir: Path,
    out_dir: Path,
    runtime: RuntimeOverrides,
) -> None:
    source_dir = results_dir / f"seed_{seed:03d}" / method
    npz_path = source_dir / "run_results.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Saved run results not found: {npz_path}")

    scenario = _sample_scenario(seed, variant)
    par, problem, state, _ = _build_case(
        scenario=scenario,
        discount=discount,
        out_dir=str(out_dir),
        variant=variant,
        runtime=runtime,
    )

    data = _load_results(npz_path)
    log = _reconstruct_log(data, state.prior.q_log_bounds)
    traj = np.asarray(data["traj"], dtype=float)

    num_updates = max(0, len(log["beliefs"]) - 1)
    par.meas_times = np.asarray(data["meas_times"], dtype=float)[:num_updates]
    par.output_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_path, log_path = _downsample_path_inputs(traj, log)
    plot_drone_paths(traj_path, log_path, par, problem, state, show_full_domain=True)
    plot_belief_convergence_with_intensity(log, par, problem, out_dir=str(out_dir))
    plot_belief_distribution_ellipses(log, par, problem, out_dir=str(out_dir))
    plot_confidence_ellipses_with_wind(log, par, problem, out_dir=str(out_dir))
    plot_running_cost_breakdown(traj, log, par, problem, drone_idx=None, out_dir=str(out_dir))
    print(f"Saved standard diagnostic suite for seed={seed}, method={method}: {out_dir}")


def _load_convergence_case(
    *,
    seed: int,
    method: str,
    discount: float,
    variant: str,
    results_dir: Path,
    out_dir: Path,
    runtime: RuntimeOverrides,
):
    source_dir = results_dir / f"seed_{seed:03d}" / method
    npz_path = source_dir / "run_results.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Saved run results not found: {npz_path}")

    scenario = _sample_scenario(seed, variant)
    par, problem, state, _ = _build_case(
        scenario=scenario,
        discount=discount,
        out_dir=str(out_dir),
        variant=variant,
        runtime=runtime,
    )
    data = _load_results(npz_path)
    log = _reconstruct_log(data, state.prior.q_log_bounds)
    return par, problem, data, log


def _belief_series(data: np.lib.npyio.NpzFile, log: dict, problem) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beliefs = log["beliefs"]
    n_updates = max(0, len(beliefs) - 1)
    t = np.concatenate(([0.0], np.asarray(data["meas_times"], dtype=float)[:n_updates]))
    mus = np.stack([b.mu for b in beliefs], axis=0)
    sigmas = np.stack([b.sigma for b in beliefs], axis=0)
    q_logs = np.asarray([np.nan if b.q_log is None else float(b.q_log) for b in beliefs], dtype=float)
    distance = np.linalg.norm(mus - np.asarray(problem.pollutant.source, dtype=float)[None, :], axis=1)
    trace = np.trace(sigmas, axis1=1, axis2=2)
    q_error = np.abs(np.exp(q_logs) - float(problem.pollutant.q))
    return t, np.maximum(distance, 1e-12), np.maximum(trace, 1e-12), np.maximum(q_error, 1e-12)


def _plot_belief_convergence_comparison(
    *,
    seed: int,
    discount: float,
    variant: str,
    results_dir: Path,
    out_dir: Path,
    runtime: RuntimeOverrides,
) -> None:
    cases = {}
    for method in ("order1", "order2"):
        cases[method] = _load_convergence_case(
            seed=seed,
            method=method,
            discount=discount,
            variant=variant,
            results_dir=results_dir,
            out_dir=out_dir,
            runtime=runtime,
        )

    fig, axes = plt.subplots(3, 2, figsize=(9.4, 7.4), sharex=True, sharey="row")
    row_specs = [
        ("Distance [m]", "C0"),
        (r"$\mathrm{tr}(\Sigma)$ [m$^2$]", "C2"),
        (r"$|\hat q-q|$ [g/s]", "C3"),
    ]
    for col, method in enumerate(("order1", "order2")):
        par, problem, data, log = cases[method]
        t, distance, trace, q_error = _belief_series(data, log, problem)
        series = (distance, trace, q_error)
        title = "One-body" if method == "order1" else "Pairwise"
        axes[0, col].set_title(title)
        for row, (ylabel, color) in enumerate(row_specs):
            ax = axes[row, col]
            ax.semilogy(t, series[row], "o-", color=color, linewidth=1.9, markersize=4.5)
            ax.grid(True, which="both", alpha=0.3)
            ax.set_xlim(0.0, float(runtime.t_end))
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.axhline(float(par.tol_sigma), color="k", linestyle="--", linewidth=1.0, alpha=0.75)
            if row == 2:
                ax.set_xlabel("Time [s]")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "belief_convergence_comparison.pdf"
    png_path = out_dir / "belief_convergence_comparison.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved paired belief convergence comparison: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the standard diagnostic plot suite for one saved seed")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--methods", nargs="+", choices=("order1", "order2"), default=["order1", "order2"])
    parser.add_argument("--discount", type=float, default=0.3)
    parser.add_argument("--variant", choices=("primary", "fallback", "collision_tuned", "prior_tuned"), default="prior_tuned")
    parser.add_argument("--results-dir", default=os.path.join(ROOT, "results", "paper_pairwise_showcase"))
    parser.add_argument("--out-dir", default=None, help="Default: <results-dir>/diagnostic_seed_<seed>")
    parser.add_argument("--figures-dir", default=os.path.join(ROOT, "paper", "figures"),
                        help="Directory where paper-named PDFs are written")
    parser.add_argument("--t-end", type=float, default=600.0)
    parser.add_argument("--dt-meas", type=float, default=30.0)
    parser.add_argument("--dx", type=float, default=60.0)
    parser.add_argument("--dx-secondary", type=float, default=120.0)
    parser.add_argument("--num-belief-discr", type=int, default=3)
    parser.add_argument("--fp-max-iter", type=int, default=12)
    args = parser.parse_args()

    import shutil

    runtime = RuntimeOverrides(
        t_end=float(args.t_end),
        dt_meas=float(args.dt_meas),
        dx=float(args.dx),
        dx_secondary=float(args.dx_secondary),
        num_belief_discr=int(args.num_belief_discr),
        fp_max_iter=int(args.fp_max_iter),
    )
    seed = int(args.seed)
    base_out = Path(args.out_dir) if args.out_dir is not None else Path(args.results_dir) / f"diagnostic_seed_{seed:03d}"
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        _make_suite(
            seed=seed,
            method=str(method),
            discount=float(args.discount),
            variant=str(args.variant),
            results_dir=Path(args.results_dir),
            out_dir=base_out / str(method),
            runtime=runtime,
        )
        # Copy path figure to figures/ with paper-canonical name
        src_path = base_out / str(method) / "drones_path.pdf"
        if src_path.exists():
            dst = figures_dir / f"pairwise_showcase_seed{seed:03d}_{method}_path.pdf"
            shutil.copy2(src_path, dst)
            print(f"Saved: {dst}")

    if {"order1", "order2"}.issubset(set(args.methods)):
        _plot_belief_convergence_comparison(
            seed=seed,
            discount=float(args.discount),
            variant=str(args.variant),
            results_dir=Path(args.results_dir),
            out_dir=base_out,
            runtime=runtime,
        )
        # Copy belief convergence figure to figures/ with paper-canonical name
        src_conv = base_out / "belief_convergence_comparison.pdf"
        if src_conv.exists():
            dst_conv = figures_dir / f"pairwise_showcase_seed{seed:03d}_belief_convergence_comparison.pdf"
            shutil.copy2(src_conv, dst_conv)
            print(f"Saved: {dst_conv}")


if __name__ == "__main__":
    main()
