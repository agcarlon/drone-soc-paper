from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any, List

from hjb import OCProblem, solve_hjb, n_linear_interpolate
from fixed_point import PicardFixedPointSolver, BaseFixedPointSolver
from drone_soc import (
    Params,
    ProblemSpec,
    DroneProblemState,
    ControlPlanner,
    collision_potential_from_state,
    compute_collision_gradients,
    no_fly_potential_from_state,
)
from _drone_soc_internal import (
    _apply_measurement_jump,
    construct_hjb_grid_axes,
    default_fixed_state_indices,
    eigen_from_covariance,
)
from plotting.plotting import animate_value_function

def _extract_norms(errors: List[Any]) -> Tuple[List[float], List[float]]:
    """
    Convert a list of residual dicts or floats into separate linf and l2 lists.
    """
    linf_list: List[float] = []
    l2_list: List[float] = []
    for e in errors:
        if isinstance(e, dict):
            linf_list.append(float(e.get("linf", 0.0)))
            l2_list.append(float(e.get("l2", np.nan)))
        else:
            linf_list.append(float(e))
            l2_list.append(float("nan"))
    return linf_list, l2_list


def _save_fp_residue_plot(errors: List[Any], tol: float, output_dir: Optional[str], ts: Tuple[float, float], tag: Optional[str] = None) -> None:
    """
    Save a semilogy plot of fixed-point residuals for debugging divergence.
    """
    linf_errs, l2_errs = _extract_norms(errors)
    save_dir = output_dir or "."
    os.makedirs(save_dir, exist_ok=True)
    tag_suffix = f"_{tag}" if tag else ""
    fname = os.path.join(save_dir, f"fp_residue_{ts[0]}-{ts[1]}{tag_suffix}.png")
    try:
        import matplotlib.pyplot as plt
        its = np.arange(1, len(linf_errs) + 1)
        fig, ax = plt.subplots()
        ax.semilogy(its, linf_errs, marker="o", label="linf")
        if not np.all(np.isnan(l2_errs)):
            ax.semilogy(its, l2_errs, marker="s", label="l2")
        ax.axhline(tol, color="r", linestyle="--", label=f"tol={tol:g}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual")
        ax.set_title("Fixed-point residual")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved fixed-point residual plot to {fname}")
    except Exception as exc:
        print(f"Failed to plot fixed-point residual: {exc}")


def _solve_infinite_hjb_nd_direction(
    ts: Tuple[float, float],
    par: Params,
    problem: ProblemSpec,
    s: DroneProblemState,
    order: int,
    wind_aligned: bool,
    fp_solver: Optional[BaseFixedPointSolver] = None,
) -> Tuple[Callable, np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Solve a discounted cycle surrogate on the fixed absolute-time interval ts.
    Virtual cycles replay this interval's patrol waveform; real replanning
    supplies the next interval. The interval need not be a patrol period.
    """

    # Spatial axes around current belief mean
    num_drones = order
    grid_axes = construct_hjb_grid_axes(par, problem, s, num_drones, wind_aligned)
    if problem.patrol is None:
        raise ValueError("ProblemSpec.patrol must be provided for infinite-horizon planning.")

    def running(x, u, t):
        total = 0.0
        for k in range(num_drones):
            zkx = x[2 * k]
            zky = x[2 * k + 1]
            ukx = u[2 * k]
            uky = u[2 * k + 1]
            work = 0.5 * problem.c1 * ((ukx - problem.wind[0]) ** 2 + (uky - problem.wind[1]) ** 2)
            x_abs = np.stack([zkx + s.belief.mu[0], zky + s.belief.mu[1]], axis=0)
            u_p = problem.patrol(x_abs, t)
            penalty = 0.5 * problem.c2 * ((ukx - u_p[0]) ** 2 + (uky - u_p[1]) ** 2)
            total = total + work + penalty
        total = total + problem.c3 * collision_potential_from_state(x, num_drones, problem.collision)
        total = total + problem.c4 * no_fly_potential_from_state(x, num_drones, problem.no_fly_zones, offset=s.belief.mu)
        return total

    def analytic_minimizer(mesh: np.ndarray, gradV: np.ndarray, t_now: float):
        spatial = gradV[: 2 * num_drones]
        alphas = []
        c_eff = problem.c1 + problem.c2
        for k in range(num_drones):
            z1 = mesh[2 * k]; z2 = mesh[2 * k + 1]
            x_abs = np.stack([z1 + s.belief.mu[0], z2 + s.belief.mu[1]], axis=0)
            u_p = problem.patrol(x_abs, t_now)
            # clip u_p to v_max per grid point
            up_norm = np.sqrt(u_p[0] ** 2 + u_p[1] ** 2) + 1e-12
            up_scale = np.minimum(1.0, par.v_max / up_norm)
            u_px = u_p[0] * up_scale
            u_py = u_p[1] * up_scale
            B0 = (problem.c1 * problem.wind[0] + problem.c2 * u_px) / c_eff
            B1 = (problem.c1 * problem.wind[1] + problem.c2 * u_py) / c_eff
            alphas.append(np.stack([B0 - spatial[2 * k] / c_eff, B1 - spatial[2 * k + 1] / c_eff], axis=0))
        alpha = np.stack(alphas, axis=0)
        grid_shape = alpha.shape[2:]
        alpha_view = alpha
        norms = np.sqrt(alpha_view[:, 0] ** 2 + alpha_view[:, 1] ** 2) + 1e-12
        scale = np.minimum(1.0, par.v_max / norms)
        alpha_view[:, 0] *= scale
        alpha_view[:, 1] *= scale
        return alpha_view.reshape((2 * num_drones,) + grid_shape), None

    # One jump at the end of the cycle
    meas_times_window = np.array([ts[1]], dtype=float)

    solver = fp_solver if fp_solver is not None else PicardFixedPointSolver()

    def _run_fixed_point(dt_hjb: float) -> Tuple[np.ndarray, List[Any], float]:
        def F_map(V0_prev: Optional[np.ndarray]) -> np.ndarray:
            def terminal_cost_from_prev(x):
                if V0_prev is None:
                    return 0.0 * x[0]
                return V0_prev
            oc = OCProblem(
                dynamics=lambda x, u, t: u,
                running_cost=running,
                terminal_cost=terminal_cost_from_prev,
                grid_axes=grid_axes,
                dt=dt_hjb,
                T0=ts[0],
                T=ts[1],
                discount=float(par.discount),
                diffusion_coeff=0.5 * par.dyn_noise_std ** 2,
                analytic_minimizer=analytic_minimizer,
                measurement_times=meas_times_window,
                measurement_jump=lambda axes, Vnext, t_now: _apply_measurement_jump(
                    axes, Vnext, t_now, s=s, par=par, problem=problem, num_drones=num_drones, wind_aligned=wind_aligned
                ),
                num_spatial_axes=2 * num_drones,
            )
            _, V_time, _ = solve_hjb(oc)
            return V_time[0]

        V0_star_local, errors_local = solver.solve(F_map, x0=None)
        return V0_star_local, errors_local, dt_hjb

    dt_used = float(par.dt_hjb)
    V0_star, errors, dt_used = _run_fixed_point(dt_used)
    attempts = [
        {
            "dt_hjb": float(dt_used),
            "iterations": int(len(errors)),
            "final_linf": float(_extract_norms(errors)[0][-1]) if errors else float("nan"),
            "final_residual_linf": float(errors[-1].get("residual_linf", np.nan)) if errors else float("nan"),
            "final_step_linf": float(errors[-1].get("step_linf", np.nan)) if errors else float("nan"),
            "fallback_count": int(sum(bool(item.get("fallback", False)) for item in errors)),
            "converged": bool(
                errors and float(_extract_norms(errors)[0][-1]) < solver.tol
            ),
        }
    ]
    linf_errs, l2_errs = _extract_norms(errors)
    if len(linf_errs) > 0 and linf_errs[-1] >= solver.tol:
        _save_fp_residue_plot(errors, solver.tol, par.output_dir, ts, tag=f"dt{dt_used}")
        last_l2 = l2_errs[-1] if len(l2_errs) > 0 else float("nan")
        print(
            f"Warning: periodic FP did not converge in {solver.max_iter} iters (dt_hjb={dt_used}). "
            f"Final residual linf={linf_errs[-1]:.3e}, l2={last_l2:.3e}"
        )
        dt_retry = dt_used * 0.5
        print(f"Retrying fixed point with smaller dt_hjb={dt_retry} ...")
        V0_star, errors, dt_used = _run_fixed_point(dt_retry)
        attempts.append(
            {
                "dt_hjb": float(dt_used),
                "iterations": int(len(errors)),
                "final_linf": float(_extract_norms(errors)[0][-1]) if errors else float("nan"),
                "final_residual_linf": float(errors[-1].get("residual_linf", np.nan)) if errors else float("nan"),
                "final_step_linf": float(errors[-1].get("step_linf", np.nan)) if errors else float("nan"),
                "fallback_count": int(sum(bool(item.get("fallback", False)) for item in errors)),
                "converged": bool(
                    errors and float(_extract_norms(errors)[0][-1]) < solver.tol
                ),
            }
        )
        linf_errs, l2_errs = _extract_norms(errors)
        if len(linf_errs) > 0 and linf_errs[-1] >= solver.tol:
            _save_fp_residue_plot(errors, solver.tol, par.output_dir, ts, tag=f"dt{dt_used}")
            last_l2 = l2_errs[-1] if len(l2_errs) > 0 else float("nan")
            print(
                f"Warning: retry with smaller dt_hjb did not converge in {solver.max_iter} iters "
                f"(dt_hjb={dt_used}). Final residual linf={linf_errs[-1]:.3e}, l2={last_l2:.3e}"
            )
    final_linf = float(linf_errs[-1]) if linf_errs else float("nan")
    final_absolute_linf = float(errors[-1].get("residual_linf", np.nan)) if errors else float("nan")
    final_step_linf = float(errors[-1].get("step_linf", np.nan)) if errors else float("nan")
    diagnostic = {
        "converged": bool(np.isfinite(final_linf) and final_linf < solver.tol),
        "final_linf": final_linf,
        "final_residual_linf": final_absolute_linf,
        "final_normalized_residual_linf": final_linf,
        "final_iterate_difference_linf": final_step_linf,
        "criterion": "normalized_bellman_residual_linf",
        "tolerance": float(solver.tol),
        "iterations": int(len(errors)),
        "total_iterations": int(sum(int(attempt["iterations"]) for attempt in attempts)),
        "attempts": int(len(attempts)),
        "attempt_details": attempts,
        "retry_used": bool(len(attempts) > 1),
        "anderson_fallback_count": int(sum(int(attempt["fallback_count"]) for attempt in attempts)),
        "dt_hjb": float(dt_used),
        "t_start": float(ts[0]),
        "t_end": float(ts[1]),
    }
    solver.last_diagnostic = diagnostic
    if not hasattr(solver, "diagnostics_history"):
        solver.diagnostics_history = []
    solver.diagnostics_history.append(diagnostic)
    # Optionally, one final pass with animation if requested
    # if par.save_videos and par.output_dir:
    oc_final = OCProblem(
        dynamics=lambda x, u, t: u,
        running_cost=running,
        terminal_cost=lambda x: V0_star if V0_star is not None else 0.0 * x[0],
        grid_axes=grid_axes,
        dt=dt_used,
        T0=ts[0],
        T=ts[1],
        discount=float(par.discount),
        diffusion_coeff=0.5 * par.dyn_noise_std ** 2,
        analytic_minimizer=analytic_minimizer,
        measurement_times=meas_times_window,
        measurement_jump=lambda axes, Vnext, t_now: _apply_measurement_jump(
            axes, Vnext, t_now, s=s, par=par, problem=problem, num_drones=num_drones, wind_aligned=wind_aligned
        ),
        num_spatial_axes=2 * num_drones,
    )
    feedback_last, V_time, grid_axes = solve_hjb(oc_final)

    if par.save_videos and par.output_dir:
        spatial_axes = (0, 1)
        fixed_indices = default_fixed_state_indices(grid_axes, spatial_axes)
        # For order-2 (pairwise) problems, visualize a slice over drone-0 axes (z1x,z1y)
        # while fixing the "other drone" axes (z2x,z2y) to a meaningful value.
        # Using the midpoint index can create a collision "lump" far from the actual
        # other drone position; instead, fix to the current state of drone 2 (if available)
        # and mark that point on the slice.
        mark_indices = None
        if num_drones >= 2:
            # Axes 2 and 3 correspond to z2x, z2y for the order-2 state.
            # Use the second real drone in the current state as a representative "other drone".
            if hasattr(s, "drones") and len(s.drones) >= 2:
                z_other = np.asarray(s.drones[1].x - s.belief.mu, dtype=float)
                # Clamp to nearest grid points for slicing
                try:
                    idx2 = int(np.argmin(np.abs(grid_axes[2] - z_other[0])))
                    idx3 = int(np.argmin(np.abs(grid_axes[3] - z_other[1])))
                    fixed_indices[2] = idx2
                    fixed_indices[3] = idx3
                    # IMPORTANT: mark the snapped coordinate actually used in the slice.
                    # With coarse dx_secondary this can differ visibly from the continuous z_other.
                    mark_indices = (float(grid_axes[2][idx2]), float(grid_axes[3][idx3]))
                except Exception:
                    mark_indices = None
        times = np.linspace(ts[0], ts[1], V_time.shape[0])
        base_name = "value_evolution_infinite"
        suffix = f"{ts[0]}-{ts[1]}"
        save_path = os.path.join(par.output_dir, f"{base_name}_{suffix}.mp4")
        animate_value_function(
            V_time=V_time,
            grid_axes=grid_axes,
            spatial_axes=spatial_axes,
            fixed_state_indices=fixed_indices,
            mark_drone_indices=mark_indices,
            times=times,
            save_path=save_path,
            quiver_alpha=0.5,
        )
    return feedback_last, V_time, grid_axes


def _pairwise_control_from_grad(
    zi: np.ndarray,
    grad_i: np.ndarray,
    t: float,
    par: Params,
    problem: ProblemSpec,
    belief_mu: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct a single drone control from the averaged gradient of the
    pairwise value function with respect to that drone's coordinates.
    """
    c_eff = problem.c1 + problem.c2
    x_abs = np.asarray(zi, dtype=float) + np.asarray(belief_mu, dtype=float)
    u_p = np.asarray(problem.patrol(x_abs, t), dtype=float)
    up_norm = np.linalg.norm(u_p) + 1e-12
    u_p = u_p * min(1.0, par.v_max / up_norm)
    target = (problem.c1 * np.asarray(problem.wind, dtype=float) + problem.c2 * u_p) / c_eff
    u = target - np.asarray(grad_i, dtype=float) / c_eff
    norm_u = np.linalg.norm(u) + 1e-12
    return u * min(1.0, par.v_max / norm_u)


def _apply_runtime_collision_safeguard(
    controls: np.ndarray,
    par: Params,
    problem: ProblemSpec,
    s: DroneProblemState,
) -> np.ndarray:
    """
    Apply the runtime collision safeguard.
    """
    a = np.asarray(controls, dtype=float).copy()
    if (problem.collision is not None) and (problem.c3 > 0.0):
        grads = compute_collision_gradients(s.drone_positions, problem.collision)
        c_eff = problem.c1 + problem.c2
        a = a - (problem.tau_col * problem.c3 / c_eff) * grads
    norms = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    return a * np.minimum(1.0, par.v_max / norms)


@dataclass
class InfiniteHorizonSingleDronePlanner(ControlPlanner):
    wind_aligned: bool = False
    fp_solver: Optional[BaseFixedPointSolver] = None
    def __post_init__(self):
        self._feedback: Optional[Callable] = None

    def planning_summary(self) -> str:
        return "Solving infinite-horizon HJB"

    def plan(self, t_now: float, par: Params, problem: ProblemSpec, s: DroneProblemState):
        meas = np.asarray(par.meas_times, dtype=float)
        idx0 = int(np.searchsorted(meas, float(t_now) - 1e-12, side="left"))
        idx0 = min(max(idx0, 0), len(meas) - 1)
        if idx0 >= len(meas) - 1:
            return
        # Keep at least one forward step when possible:
        # num_obs_in_hjb=1 should plan to the next measurement time.
        idx1 = min(idx0 + par.num_obs_in_hjb, len(meas) - 1)
        ts = (float(t_now), float(meas[idx1]))
        feedback, _, _ = _solve_infinite_hjb_nd_direction(
            ts=ts,
            par=par,
            problem=problem,
            s=s,
            order=1,
            wind_aligned=self.wind_aligned,
            fp_solver=self.fp_solver,
        )
        self._feedback = feedback

    def compute_alpha(self, t: float, par: Params, problem: ProblemSpec, s: DroneProblemState) -> np.ndarray:
        num_drones = len(s.drones)
        assert num_drones >= 1
        vals_log, ang = eigen_from_covariance(s.belief.sigma)
        a = np.zeros((num_drones, 2), dtype=float)
        intensity_coord = None
        if s.belief.intensity_enabled:
            intensity_coord = float(np.log(np.maximum(s.belief.q_log_var, 1e-12)))
        for i, d in enumerate(s.drones):
            zi = d.x - s.belief.mu
            if self.wind_aligned:
                coords = (*zi, *vals_log)
            else:
                coords = (*zi, *vals_log, ang)
            if intensity_coord is not None:
                coords = (*coords, intensity_coord)
            a[i] = np.asarray(self._feedback(t, coords))
        return _apply_runtime_collision_safeguard(a, par=par, problem=problem, s=s)


@dataclass
class InfiniteHorizonTwoDronesPlanner(ControlPlanner):
    wind_aligned: bool = False
    fp_solver: Optional[BaseFixedPointSolver] = None
    def __post_init__(self):
        self._feedback: Optional[Callable] = None
        self._time_axis: Optional[np.ndarray] = None
        self._grid_axes: Optional[Tuple[np.ndarray, ...]] = None
        self._grad_pair_0: Optional[np.ndarray] = None
        self._grad_pair_1: Optional[np.ndarray] = None
        self._belief_mu_plan: Optional[np.ndarray] = None

    def planning_summary(self) -> str:
        return "Solving infinite-horizon HJB"

    def plan(self, t_now: float, par: Params, problem: ProblemSpec, s: DroneProblemState):
        meas = np.asarray(par.meas_times, dtype=float)
        idx0 = int(np.searchsorted(meas, float(t_now) - 1e-12, side="left"))
        idx0 = min(max(idx0, 0), len(meas) - 1)
        if idx0 >= len(meas) - 1:
            return
        # Keep at least one forward step when possible:
        # num_obs_in_hjb=1 should plan to the next measurement time.
        idx1 = min(idx0 + par.num_obs_in_hjb, len(meas) - 1)
        ts = (float(t_now), float(meas[idx1]))
        feedback, V_time, grid_axes = _solve_infinite_hjb_nd_direction(
            ts=ts,
            par=par,
            problem=problem,
            s=s,
            order=2,
            wind_aligned=self.wind_aligned,
            fp_solver=self.fp_solver,
        )
        self._feedback = feedback
        self._time_axis = np.linspace(ts[0], ts[1], V_time.shape[0])
        self._grid_axes = grid_axes
        d0 = float(grid_axes[0][1] - grid_axes[0][0])
        d1 = float(grid_axes[1][1] - grid_axes[1][0])
        self._grad_pair_0 = np.gradient(V_time, d0, axis=1, edge_order=2)
        self._grad_pair_1 = np.gradient(V_time, d1, axis=2, edge_order=2)
        self._belief_mu_plan = np.asarray(s.belief.mu, dtype=float).copy()

    def compute_alpha(self, t: float, par: Params, problem: ProblemSpec, s: DroneProblemState) -> np.ndarray:
        num_drones = len(s.drones)
        assert num_drones >= 2
        if (
            self._time_axis is None
            or self._grid_axes is None
            or self._grad_pair_0 is None
            or self._grad_pair_1 is None
            or self._belief_mu_plan is None
        ):
            raise RuntimeError("Order-2 planner has no cached value-gradient data. Call plan() first.")
        vals_log, ang = eigen_from_covariance(s.belief.sigma)
        a = np.zeros((num_drones, 2), dtype=float)
        intensity_coord = None
        if s.belief.intensity_enabled:
            intensity_coord = float(np.log(np.maximum(s.belief.q_log_var, 1e-12)))
        for i in range(num_drones):
            zi = s.drones[i].x - s.belief.mu
            grad_i = np.zeros(2, dtype=float)
            for j in range(num_drones):
                if j == i:
                    continue
                zj = s.drones[j].x - s.belief.mu
                if self.wind_aligned:
                    coords = (*zi, *zj, *vals_log)
                else:
                    coords = (*zi, *zj, *vals_log, ang)
                if intensity_coord is not None:
                    coords = (*coords, intensity_coord)
                interp_coords = tuple(np.asarray([v], dtype=float) for v in (t, *coords))
                grad_i[0] += float(
                    n_linear_interpolate(
                        self._grad_pair_0,
                        (self._time_axis, *self._grid_axes),
                        interp_coords,
                        boundary_mode="clip",
                    )[0]
                )
                grad_i[1] += float(
                    n_linear_interpolate(
                        self._grad_pair_1,
                        (self._time_axis, *self._grid_axes),
                        interp_coords,
                        boundary_mode="clip",
                    )[0]
                )
            grad_i /= float(num_drones - 1)
            a[i] = _pairwise_control_from_grad(
                zi=zi,
                grad_i=grad_i,
                t=float(t),
                par=par,
                problem=problem,
                belief_mu=self._belief_mu_plan,
            )
        return _apply_runtime_collision_safeguard(a, par=par, problem=problem, s=s)
