"""Shared baseline scenario and controller definitions for paper experiments."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from drone_soc import (  # noqa: E402
    Params,
    ProblemSpec,
    Belief,
    DroneProblemState,
    Drone,
    ControlPlanner,
    SinusoidalPatrolling,
    SigmoidCollisionAvoidance,
    compute_collision_gradients,
)
from fixed_point import AndersonAccelerationFixedPointSolver  # noqa: E402
from measurement import SlicedScreenedAdvectionDiffusionPlume  # noqa: E402


@dataclass(frozen=True)
class BaselineScenarioSpec:
    output_dir: str = os.path.join(ROOT, "results", "greedy_eig_comparison")
    snr0_levels: Sequence[float] = (50, 100, 150, 200)
    seeds: Sequence[int] = tuple(range(50))
    success_radius: float = 50.0
    patrol_period: float = 200.0
    patrol_y_scale: float = 0.8
    passive_patrol_period: float = 40.0
    passive_patrol_y_scale: float = 0.99


@dataclass(frozen=True)
class BaselineScenario:
    seed: int
    snr0_target: float
    source_pos: np.ndarray
    q: float
    snr0_realized: float


class PatrolOnlyPlanner(ControlPlanner):
    """
    Passive baseline: follow the patrol field at all times.
    """

    def planning_summary(self) -> str:
        return "Refreshing passive patrol control"

    def plan(self, t_now: float, par: Params, problem: ProblemSpec, s: DroneProblemState):
        return None

    def compute_alpha(self, t: float, par: Params, problem: ProblemSpec, s: DroneProblemState) -> np.ndarray:
        if problem.patrol is None:
            raise ValueError("ProblemSpec.patrol must be provided for patrol-only baseline.")
        controls = np.zeros((len(s.drones), 2), dtype=float)
        for i, drone in enumerate(s.drones):
            controls[i] = np.asarray(problem.patrol(np.asarray(drone.x, dtype=float), float(t)), dtype=float).reshape(2)

        if (problem.collision is not None) and (problem.c3 > 0.0):
            grads = compute_collision_gradients(s.drone_positions, problem.collision)
            c_eff = problem.c1 + problem.c2
            controls = controls - (problem.tau_col * problem.c3 / c_eff) * grads

        norms = np.linalg.norm(controls, axis=1, keepdims=True) + 1e-12
        controls = controls * np.minimum(1.0, par.v_max / norms)
        return controls


def _base_geometry() -> Dict[str, Any]:
    domain = np.array([[-500.0, 500.0], [-500.0, 500.0]], dtype=float)
    drone_positions = np.array(
        [
            [-450.0, 400.0],
            [-400.0, -100.0],
            [-450.0, -400.0],
        ],
        dtype=float,
    )
    wind = np.array([-5.0, 1.5], dtype=float)
    return {
        "domain": domain,
        "drone_positions": drone_positions,
        "wind": wind,
    }


def _unit_plume_for_source(source_pos: np.ndarray, noise_std: float) -> SlicedScreenedAdvectionDiffusionPlume:
    geom = _base_geometry()
    return SlicedScreenedAdvectionDiffusionPlume(
        source=np.asarray(source_pos, dtype=float),
        q=1.0,
        ds=80.0,
        tau_s=3000.0,
        wind=geom["wind"],
        sigma_e=noise_std,
        eps_c=5.0,
    )


def _q_for_source_snr0(snr0_target: float, noise_std: float) -> float:
    """
    Return q such that C(θ)/σ_e = snr0_target at the source location.

    For the sliced screened plume C(θ) = q·exp(−κ·ε_c)/(4π·d_s·ε_c), which is
    linear in q, so the inversion is exact.
    """
    pol_unit = _unit_plume_for_source(np.array([0.0, 0.0], dtype=float), noise_std=noise_std)
    kappa = pol_unit._kappa()
    r_c = float(pol_unit.eps_c)
    c_at_source_per_unit_q = float(np.exp(-kappa * r_c) / (4.0 * np.pi * pol_unit.ds * r_c))
    return float(snr0_target * noise_std / c_at_source_per_unit_q)


def _sample_scenario(
    seed: int,
    snr0_target: float,
    spec: BaselineScenarioSpec,
) -> BaselineScenario:
    """Sample a source position uniformly and set q to match snr0_target."""
    rng = np.random.default_rng(seed)
    noise_std = 5e-5
    source_pos = np.array(
        [
            rng.uniform(150.0, 400.0),
            rng.uniform(-250.0, 250.0),
        ],
        dtype=float,
    )
    q = _q_for_source_snr0(float(snr0_target), noise_std=noise_std)
    snr0_realized = float(snr0_target)
    return BaselineScenario(
        seed=seed,
        snr0_target=float(snr0_target),
        source_pos=source_pos,
        q=q,
        snr0_realized=snr0_realized,
    )


def make_baseline_case(
    scenario: BaselineScenario,
    output_dir: str,
    *,
    patrol_period: float = 200.0,
    patrol_y_scale: float = 0.8,
) -> tuple[Params, ProblemSpec, DroneProblemState, AndersonAccelerationFixedPointSolver]:
    """Build the shared 3-drone problem for a given scenario."""
    geom = _base_geometry()
    domain = geom["domain"]
    drone_positions = geom["drone_positions"]
    wind = geom["wind"]
    v_max = 12.0

    t_end = 400.0
    dt_meas = 5.0
    par = Params(
        meas_times=np.linspace(0.0, t_end, int(t_end / dt_meas) + 1),
        t_end=t_end,
        dt=0.05,
        dt_hjb=0.1,
        dx=50.0,
        dx_secondary=100.0,
        v_max=v_max,
        noise_std=5e-5,
        dyn_noise_std=0.5,
        tol_sigma=10.0,
        tol_residual=3.0,
        tol_q_log_var=0.5,
        discount=0.3,
        save_videos=False,
        update_mode="full_batch",
        output_dir=output_dir,
        stop_when_confident=True,
        min_stop_updates=20,
        confident_streak=3,
        verbose_planning=False,
        num_obs_in_hjb=1,
        seed=scenario.seed,
    )

    pol = SlicedScreenedAdvectionDiffusionPlume(
        source=np.asarray(scenario.source_pos, dtype=float),
        q=float(scenario.q),
        ds=80.0,
        tau_s=3000.0,
        wind=wind,
        sigma_e=par.noise_std,
        eps_c=5.0,
    )
    patrol = SinusoidalPatrolling(
        wind=wind,
        v_max=v_max,
        period=float(patrol_period),
        y_scale=float(patrol_y_scale),
    )
    collision = SigmoidCollisionAvoidance(d_min=50.0, d_max=70.0, max_value=100.0, confidence=0.99)
    problem = ProblemSpec(
        pollutant=pol,
        domain=domain,
        wind=wind,
        no_fly_zones=None,
        c1=1e-4,
        c2=1e-4,
        c3=1e-2,
        tau_col=1.0,
        c4=0.0,
        patrol=patrol,
        collision=collision,
    )
    
    prior = Belief(
        mu=np.array([0.0, 0.0], dtype=float),
        sigma=(300.0**2) * np.eye(2),
        q_log=float(np.log(30.0)),
        q_log_var=5.0,
        q_log_bounds=(2, 10),
    )
    s = DroneProblemState(
        prior=prior.copy(),
        belief=prior.copy(),
        drones=[Drone(x0=pos, v_max=par.v_max, patrol_dir_fn=problem.patrol) for pos in drone_positions],
    )
    fp_solver = AndersonAccelerationFixedPointSolver(m=3, tol=1e-3, max_iter=40, verbose=False)
    return par, problem, s, fp_solver


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
    step_lengths = np.linalg.norm(diffs, axis=1)
    return float(np.sum(step_lengths))


def _snr_dir(output_dir: str, snr0_target: float) -> str:
    return os.path.join(output_dir, f"snr0_{float(snr0_target):.1f}")
