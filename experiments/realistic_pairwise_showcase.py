"""
Scenario definition for the pairwise-planning showcase (§5.2–5.7).

Provides make_realistic_pairwise_showcase(), which builds the shared
3-drone cooperative localization problem (plume, no-fly zones, collision
avoidance, patrol baseline, prior) used by all paper experiments that
involve pairwise HJB planning.

Variants:
  prior_tuned     — tighter prior centred on the source box; used for all paper results
  primary         — broad uniform prior, primary wind
  fallback        — broad prior, fallback wind
  collision_tuned — primary scenario with tighter collision penalty onset
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from drone_soc import (
    Belief,
    CircularNoFlyZone,
    Drone,
    DroneProblemState,
    Params,
    ProblemSpec,
    SigmoidCollisionAvoidance,
    SinusoidalPatrolling,
)
from fixed_point import AndersonAccelerationFixedPointSolver
from measurement import SlicedScreenedAdvectionDiffusionPlume


SOURCE_X_BOUNDS: tuple[float, float] = (200.0, 310.0)
SOURCE_Y_BOUNDS: tuple[float, float] = (-110.0, 110.0)

PRIMARY_TRUE_Q: float = 60.0
PRIMARY_WIND = np.array([0.0, 0.5], dtype=float)

FALLBACK_TRUE_Q: float = 75.0
FALLBACK_WIND = np.array([0.0, 0.8], dtype=float)


@dataclass(frozen=True)
class RealisticPairwiseShowcaseSpec:
    discount: float = 0.3
    output_dir: str = os.path.join("results", "pairwise_showcase")
    variant: str = "primary"
    t_end: float = 600.0
    dt_meas: float = 30.0
    dx: float = 60.0
    dx_secondary: float = 120.0
    num_belief_discr: int = 4
    fp_max_iter: int = 25


def _variant_values(variant: str) -> tuple[np.ndarray, float]:
    if variant == "primary":
        return PRIMARY_WIND.copy(), PRIMARY_TRUE_Q
    if variant == "fallback":
        return FALLBACK_WIND.copy(), FALLBACK_TRUE_Q
    if variant in {"collision_tuned", "prior_tuned"}:
        return PRIMARY_WIND.copy(), PRIMARY_TRUE_Q
    raise ValueError(
        "RealisticPairwiseShowcaseSpec.variant must be 'primary', 'fallback', "
        "'collision_tuned', or 'prior_tuned'."
    )


def _variant_drone_positions(variant: str) -> np.ndarray:
    if variant in {"collision_tuned", "prior_tuned"}:
        return np.array(
            [
                [-220.0, 90.0],
                [-220.0, 0.0],
                [-220.0, -90.0],
            ],
            dtype=float,
        )
    return np.array(
        [
            [-220.0, 90.0],
            [-240.0, 0.0],
            [-220.0, -90.0],
        ],
        dtype=float,
    )


def _variant_collision_d_max(variant: str) -> float:
    if variant in {"collision_tuned", "prior_tuned"}:
        return 90.0
    return 130.0


def _variant_prior(variant: str) -> tuple[np.ndarray, np.ndarray, float, float, tuple[float, float]]:
    if variant == "prior_tuned":
        return (
            np.array([255.0, 0.0], dtype=float),
            np.diag([90.0**2, 110.0**2]),
            float(np.log(45.0)),
            1.0,
            (float(np.log(10.0)), float(np.log(500.0))),
        )
    return (
        np.array([0.0, 0.0], dtype=float),
        (220.0**2) * np.eye(2),
        float(np.log(35.0)),
        2.0,
        (float(np.log(10.0)), float(np.log(1000.0))),
    )


def make_realistic_pairwise_showcase(
    spec: Optional[RealisticPairwiseShowcaseSpec] = None,
) -> Tuple[Params, ProblemSpec, DroneProblemState, AndersonAccelerationFixedPointSolver]:
    if spec is None:
        spec = RealisticPairwiseShowcaseSpec()

    wind, true_q = _variant_values(spec.variant)

    domain = np.array([[-360.0, 360.0], [-360.0, 360.0]], dtype=float)
    drone_positions = _variant_drone_positions(spec.variant)
    source_pos = np.array([255.0, 0.0], dtype=float)

    t_end = float(spec.t_end)
    dt_meas = float(spec.dt_meas)
    v_max = 10.0
    noise_std = 2e-5

    par = Params(
        meas_times=np.linspace(0.0, t_end, int(t_end / dt_meas) + 1),
        t_end=t_end,
        dt=0.1,
        dt_hjb=1.0,
        dx=float(spec.dx),
        dx_secondary=float(spec.dx_secondary),
        num_belief_discr=int(spec.num_belief_discr),
        v_max=v_max,
        noise_std=noise_std,
        dyn_noise_std=0.3,
        tol_sigma=40.0,
        tol_residual=8.0,
        tol_q_log_var=2.0,
        discount=float(spec.discount),
        save_videos=False,
        update_mode="full_batch",
        output_dir=spec.output_dir,
        stop_when_confident=True,
        min_stop_updates=6,
        confident_streak=2,
        seed=1,
        verbose_planning=False,
    )

    pollutant = SlicedScreenedAdvectionDiffusionPlume(
        source=source_pos,
        q=true_q,
        ds=90.0,
        tau_s=2600.0,
        wind=wind,
        sigma_e=par.noise_std,
        eps_c=22.0,
    )
    no_fly_zones = [
        CircularNoFlyZone(center=np.array([-95.0, 0.0]), radius=55.0, smooth_width=40.0, max_value=1.0),
        CircularNoFlyZone(center=np.array([35.0, 120.0]), radius=50.0, smooth_width=40.0, max_value=1.0),
        CircularNoFlyZone(center=np.array([35.0, -120.0]), radius=50.0, smooth_width=40.0, max_value=1.0),
    ]
    patrol = SinusoidalPatrolling(wind=wind, v_max=v_max, period=220.0, y_scale=0.82)
    collision = SigmoidCollisionAvoidance(
        d_min=45.0,
        d_max=_variant_collision_d_max(spec.variant),
        max_value=80.0,
        confidence=0.99,
    )

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

    prior_mu, prior_sigma, q_log, q_log_var, q_log_bounds = _variant_prior(spec.variant)
    prior = Belief(
        mu=prior_mu,
        sigma=prior_sigma,
        q_log=q_log,
        q_log_var=q_log_var,
        q_log_bounds=q_log_bounds,
    )
    state = DroneProblemState(
        prior=prior.copy(),
        belief=prior.copy(),
        drones=[Drone(x0=pos, v_max=par.v_max, patrol_dir_fn=problem.patrol) for pos in drone_positions],
    )
    fp_solver = AndersonAccelerationFixedPointSolver(m=3, tol=1e-3, max_iter=int(spec.fp_max_iter), verbose=False)
    return par, problem, state, fp_solver
