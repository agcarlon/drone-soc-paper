import numpy as np

from drone_soc import (
    Params,
    ProblemSpec,
    Belief,
    DroneProblemState,
    Drone,
)
from finite_horizon import (
    FiniteHorizonSingleDronePlanner,
    FiniteHorizonTwoDronesPlanner,
)


class _DummyDrone(Drone):
    """
    Minimal concrete Drone for finite-horizon planner tests.
    """

    def patrolling_trajectory(self, t: float):
        return self.x


class _DummyPollutant:
    """
    Simple pollutant with zero concentration everywhere.
    Used to keep HJB tests fast and deterministic.
    """

    def get_pollutant_concentration(self, x, t, theta, intensity=None):
        x = np.asarray(x)
        return np.zeros(x.shape[0], dtype=float)

    def get_concentration_gradient_wrt_theta(self, x, t, theta, intensity=None):
        x = np.atleast_2d(x)
        return np.zeros((x.shape[0], 2), dtype=float)

    def get_concentration_derivative_wrt_logq(self, x, t, theta, intensity=None):
        x = np.atleast_2d(x)
        return np.zeros(x.shape[0], dtype=float)


def _make_small_finite_problem(num_drones: int):
    par = Params(
        t_0=0.0,
        t_end=1.0,
        dt=0.5,
        dt_hjb=0.5,
        meas_times=np.array([0.0, 1.0]),
        dx=0.5,
        noise_std=1e-3,
        dyn_noise_std=0.0,
        num_belief_discr=3,
        belief_discr_factor=0.1,
        v_max=0.5,
        save_videos=False,
        output_dir=None,
    )
    drones = [
        _DummyDrone(x0=np.array([0.0, 0.0]), v_max=par.v_max) for _ in range(num_drones)
    ]
    s = DroneProblemState(
        prior=Belief(np.zeros(2, dtype=float), np.eye(2)),
        belief=Belief(np.zeros(2, dtype=float), np.eye(2)),
        drones=drones,
    )
    pol = _DummyPollutant()
    problem = ProblemSpec(
        pollutant=pol,
        domain=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        wind=np.zeros(2, dtype=float),
        c1=1e-3,
        c2=1e-3,
        c3=0.0,
        c3_terminal=0.0,
        patrol=lambda x, t: np.zeros_like(x),
        collision=None,
    )
    return par, problem, s


def test_finite_horizon_single_drone_planner_smoke():
    """
    FiniteHorizonSingleDronePlanner should produce finite controls with correct shape.
    """
    par, problem, s = _make_small_finite_problem(num_drones=3)

    planner = FiniteHorizonSingleDronePlanner(
        wind_aligned=False
    )
    planner.plan(0.0, par, problem, s)
    alpha = planner.compute_alpha(0.0, par, problem, s)

    assert alpha.shape == (len(s.drones), 2)
    assert np.all(np.isfinite(alpha))


def test_finite_horizon_two_drones_planner_smoke():
    """
    FiniteHorizonTwoDronesPlanner should produce finite controls with correct shape.
    """
    par, problem, s = _make_small_finite_problem(num_drones=3)

    planner = FiniteHorizonTwoDronesPlanner(
        wind_aligned=False
    )
    planner.plan(0.0, par, problem, s)
    alpha = planner.compute_alpha(0.0, par, problem, s)

    assert alpha.shape == (len(s.drones), 2)
    assert np.all(np.isfinite(alpha))
