import numpy as np
import pytest

from drone_soc import Params, ProblemSpec, Belief, Drone, DroneProblemState
from finite_horizon import FiniteHorizonSingleDronePlanner, FiniteHorizonTwoDronesPlanner
from infinite_horizon import InfiniteHorizonSingleDronePlanner, InfiniteHorizonTwoDronesPlanner


class _DummyPollutant:
    def __init__(self):
        self.source = np.zeros(2, dtype=float)
        self.q = 1.0
        self.sigma_e = 1e-3

    def get_pollutant_concentration(self, x, t, theta, intensity=None):
        x = np.asarray(x)
        return np.zeros(x.shape[0], dtype=float)


def _patrol_zero(x, t):
    return np.zeros_like(x)


def _make_problem(num_drones: int):
    par = Params(
        t_0=0.0,
        t_end=0.4,
        dt=0.1,
        dt_hjb=0.1,
        meas_times=np.array([0.0, 0.2, 0.4]),
        num_obs_in_hjb=1,
        dx=0.5,
        noise_std=1e-3,
        dyn_noise_std=0.0,
        num_belief_discr=3,
        belief_discr_factor=0.1,
        v_max=0.5,
    )
    drones = [Drone(x0=np.array([0.0, 0.0]), v_max=par.v_max, patrol_dir_fn=_patrol_zero) for _ in range(num_drones)]
    prior = Belief(np.zeros(2), np.eye(2))
    s = DroneProblemState(
        prior=prior.copy(),
        belief=prior.copy(),
        drones=drones,
    )
    pol = _DummyPollutant()
    problem = ProblemSpec(
        pollutant=pol,
        domain=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        wind=np.zeros(2, dtype=float),
        c1=1.0,
        c2=0.1,
        c3=0.0,
        c4=0.0,
        patrol=_patrol_zero,
        collision=None,
    )
    return par, problem, s


@pytest.mark.parametrize(
    "planner_cls,num_drones,solve_symbol,solve_module,order",
    [
        (FiniteHorizonSingleDronePlanner, 2, "_solve_hjb_nd_direction", "finite_horizon", 1),
        (FiniteHorizonTwoDronesPlanner, 2, "_solve_hjb_nd_direction", "finite_horizon", 2),
        (InfiniteHorizonSingleDronePlanner, 2, "_solve_infinite_hjb_nd_direction", "infinite_horizon", 1),
        (InfiniteHorizonTwoDronesPlanner, 2, "_solve_infinite_hjb_nd_direction", "infinite_horizon", 2),
    ],
)
def test_planner_window_and_feedback_interval(monkeypatch, planner_cls, num_drones, solve_symbol, solve_module, order):
    """
    Fast structural test for planner contracts.

    Ensures each planner:
    - plans over a non-degenerate interval when num_obs_in_hjb=1,
    - uses a horizon ending at the *next* measurement when available,
    - produces finite controls with norm bounded by v_max for times inside that window.
    """
    par, problem, s = _make_problem(num_drones)
    observed_ts = []

    def fake_solve(ts, *args, **kwargs):
        observed_ts.append((float(ts[0]), float(ts[1])))

        def feedback(t, coords):
            assert float(ts[0]) <= float(t) <= float(ts[1])
            return np.zeros(2 * order, dtype=float)

        if solve_module == "infinite_horizon":
            # The infinite-horizon solver also returns the value grid and its
            # axes. The order-2 planner uses them to reconstruct pairwise
            # controls from value gradients.
            num_state_dims = 2 * order + 3
            grid_axes = tuple(np.linspace(-1.0, 1.0, 3) for _ in range(num_state_dims))
            V_time = np.zeros((2,) + tuple(axis.size for axis in grid_axes), dtype=float)
            return feedback, V_time, grid_axes

        return feedback, None

    monkeypatch.setattr(f"{solve_module}.{solve_symbol}", fake_solve)

    planner = planner_cls(wind_aligned=False)
    planner.plan(0.0, par, problem, s)

    assert len(observed_ts) == 1
    t0, t1 = observed_ts[0]
    assert t0 == pytest.approx(0.0)
    assert t1 == pytest.approx(0.2)
    assert t1 > t0

    alpha = planner.compute_alpha(0.1, par, problem, s)
    assert alpha.shape == (num_drones, 2)
    assert np.all(np.isfinite(alpha))
    norms = np.linalg.norm(alpha, axis=1)
    assert np.all(norms <= par.v_max + 1e-12)
