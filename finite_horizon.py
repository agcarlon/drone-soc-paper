from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Tuple

import numpy as np

from hjb import OCProblem, solve_hjb
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


def _solve_hjb_nd_direction(
    ts: Tuple[float, float],
    par: Params,
    problem: ProblemSpec,
    s: DroneProblemState,
    meas_times_window: np.ndarray,
    order: int,
    wind_aligned: bool,
):
    """
    Solve a finite-horizon HJB for an ANOVA term of given order (1 or 2).
    """
    num_drones = order
    grid_axes = construct_hjb_grid_axes(par, problem, s, num_drones, wind_aligned)
    meas_times = meas_times_window
    if problem.patrol is None:
        raise ValueError("ProblemSpec.patrol must be provided for finite-horizon planning.")

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

    def terminal(x):
        total = 0.0 * x[0]
        total = total + problem.c3_terminal * collision_potential_from_state(x, num_drones, problem.collision)
        return total

    def analytic_minimizer(mesh: np.ndarray, gradV: np.ndarray, t_now: float):
        spatial = gradV[: 2 * num_drones]
        alphas = []
        c_eff = problem.c1 + problem.c2
        for k in range(num_drones):
            z1 = mesh[2 * k]
            z2 = mesh[2 * k + 1]
            x_abs = np.stack([z1 + s.belief.mu[0], z2 + s.belief.mu[1]], axis=0)
            u_p = problem.patrol(x_abs, t_now)
            up_norm = np.sqrt(u_p[0] ** 2 + u_p[1] ** 2) + 1e-12
            up_scale = np.minimum(1.0, par.v_max / up_norm)
            u_px = u_p[0] * up_scale
            u_py = u_p[1] * up_scale
            B0 = (problem.c1 * problem.wind[0] + problem.c2 * u_px) / c_eff
            B1 = (problem.c1 * problem.wind[1] + problem.c2 * u_py) / c_eff
            alphas.append(
                np.stack([B0 - spatial[2 * k] / c_eff, B1 - spatial[2 * k + 1] / c_eff], axis=0)
            )
        alpha = np.stack(alphas, axis=0)
        grid_shape = alpha.shape[2:]
        alpha_view = alpha
        norms = np.sqrt(alpha_view[:, 0] ** 2 + alpha_view[:, 1] ** 2) + 1e-12
        scale = np.minimum(1.0, par.v_max / norms)
        alpha_view[:, 0] *= scale
        alpha_view[:, 1] *= scale
        return alpha_view.reshape((2 * num_drones,) + grid_shape), None

    oc = OCProblem(
        dynamics=lambda x, u, t: u,
        running_cost=running,
        terminal_cost=terminal,
        grid_axes=grid_axes,
        dt=par.dt_hjb,
        T0=ts[0],
        T=ts[1],
        diffusion_coeff=0.5 * par.dyn_noise_std**2,
        analytic_minimizer=analytic_minimizer,
        measurement_times=meas_times,
        measurement_jump=partial(
            _apply_measurement_jump,
            s=s,
            par=par,
            problem=problem,
            num_drones=num_drones,
            wind_aligned=wind_aligned,
        ),
        num_spatial_axes=2 * num_drones,
    )
    feedback, V_time, grid_axes = solve_hjb(oc)

    if par.save_videos and par.output_dir:
        spatial_axes = (0, 1)
        fixed_indices = default_fixed_state_indices(grid_axes, spatial_axes)
        # For order-2 (pairwise) problems, visualize a slice over drone-0 axes (z1x,z1y)
        # while fixing the "other drone" axes (z2x,z2y) to a meaningful value.
        mark_indices = None
        if num_drones >= 2:
            if hasattr(s, "drones") and len(s.drones) >= 2:
                z_other = np.asarray(s.drones[1].x - s.belief.mu, dtype=float)
                try:
                    idx2 = int(np.argmin(np.abs(grid_axes[2] - z_other[0])))
                    idx3 = int(np.argmin(np.abs(grid_axes[3] - z_other[1])))
                    fixed_indices[2] = idx2
                    fixed_indices[3] = idx3
                    # Mark the snapped coordinate actually used in the slice.
                    mark_indices = (float(grid_axes[2][idx2]), float(grid_axes[3][idx3]))
                except Exception:
                    mark_indices = None
        times = np.linspace(ts[0], ts[1], V_time.shape[0])
        base_name = "value_evolution_finite"
        suffix = f"{num_drones}d_{'wind' if wind_aligned else 'anova'}"
        save_path = os.path.join(par.output_dir, f"{base_name}_{suffix}.mp4")
        animate_value_function(
            V_time=V_time,
            grid_axes=grid_axes,
            spatial_axes=spatial_axes,
            fixed_state_indices=fixed_indices,
            mark_drone_indices=mark_indices,
            times=times,
            save_path=save_path,
        )
    return feedback, grid_axes


def _planning_window(t_now: float, par: Params) -> Tuple[Tuple[float, float], np.ndarray]:
    meas = np.asarray(par.meas_times, dtype=float)
    idx0 = int(np.searchsorted(meas, float(t_now) + 1e-12, side="right"))
    idx0 = max(idx0, 0)
    if idx0 >= len(meas):
        return (float(t_now), float(t_now)), np.asarray([], dtype=float)

    idx1 = min(idx0 + int(par.num_obs_in_hjb) - 1, len(meas) - 1)
    meas_times_window = meas[idx0 : idx1 + 1]
    t_end = float(meas_times_window[-1])
    return (float(t_now), t_end), meas_times_window


@dataclass
class FiniteHorizonSingleDronePlanner(ControlPlanner):
    wind_aligned: bool = False
    _feedback: Optional[Callable] = field(init=False, default=None)

    def plan(
        self,
        t_now: float,
        par: Params,
        problem: ProblemSpec,
        s: DroneProblemState,
    ):
        ts, meas_times_window = _planning_window(t_now, par)
        feedback, _ = _solve_hjb_nd_direction(
            ts=ts,
            par=par,
            problem=problem,
            s=s,
            meas_times_window=meas_times_window,
            order=1,
            wind_aligned=self.wind_aligned,
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
        # Collision: for ANOVA-1 (single-drone HJB reused for many drones), we apply an
        # additional potential-field correction to enforce multi-drone safety.
        if (problem.collision is not None) and (problem.c3 > 0.0):
            grads = compute_collision_gradients(s.drone_positions, problem.collision)
            c_eff = problem.c1 + problem.c2
            a = a - (problem.tau_col * problem.c3 / c_eff) * grads
        norms = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
        a = a * np.minimum(1.0, par.v_max / norms)
        return a


@dataclass
class FiniteHorizonTwoDronesPlanner(ControlPlanner):
    wind_aligned: bool = False
    _feedback: Optional[Callable] = field(init=False, default=None)

    def plan(
        self,
        t_now: float,
        par: Params,
        problem: ProblemSpec,
        s: DroneProblemState,
    ):
        ts, meas_times_window = _planning_window(t_now, par)
        feedback, _ = _solve_hjb_nd_direction(
            ts=ts,
            par=par,
            problem=problem,
            s=s,
            meas_times_window=meas_times_window,
            order=2,
            wind_aligned=self.wind_aligned,
        )
        self._feedback = feedback

    def compute_alpha(self, t: float, par: Params, problem: ProblemSpec, s: DroneProblemState) -> np.ndarray:
        num_drones = len(s.drones)
        assert num_drones >= 2
        vals_log, ang = eigen_from_covariance(s.belief.sigma)
        a = np.zeros((num_drones, 2), dtype=float)
        intensity_coord = None
        if s.belief.intensity_enabled:
            intensity_coord = float(np.log(np.maximum(s.belief.q_log_var, 1e-12)))
        for i in range(num_drones):
            zi = s.drones[i].x - s.belief.mu
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
                a_flat = np.asarray(self._feedback(t, coords))
                a_pair = a_flat.reshape((2, 2))
                a[i] += a_pair[0] / (num_drones - 1)
        norms = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
        a = a * np.minimum(1.0, par.v_max / norms)
        return a
