import numpy as np
import pytest

from drone_soc import Params, ProblemSpec, Belief, Drone, DroneProblemState
from finite_horizon import FiniteHorizonSingleDronePlanner, FiniteHorizonTwoDronesPlanner
from infinite_horizon import InfiniteHorizonSingleDronePlanner, InfiniteHorizonTwoDronesPlanner
from fixed_point import PicardFixedPointSolver


class _ZeroPollutant:
    def __init__(self):
        self.source = np.zeros(2, dtype=float)
        self.q = 1.0
        self.sigma_e = 1e-3

    def get_pollutant_concentration(self, x, t, theta, intensity=None):
        x = np.asarray(x)
        return np.zeros(x.shape[0], dtype=float)

    def get_concentration_gradient_wrt_theta(self, x, t, theta, intensity=None):
        x = np.atleast_2d(x)
        return np.zeros((x.shape[0], 2), dtype=float)

    def get_concentration_derivative_wrt_logq(self, x, t, theta, intensity=None):
        x = np.atleast_2d(x)
        return np.zeros(x.shape[0], dtype=float)



def _patrol_zero(x, t):
    return np.zeros_like(x)



def _make_canonical_problem(num_drones: int):
    par = Params(
        t_0=0.0,
        t_end=0.5,
        dt=0.1,
        dt_hjb=0.1,
        meas_times=np.array([0.0, 0.5]),
        num_obs_in_hjb=1,
        dx=0.5,
        dx_secondary=0.5,
        noise_std=1e-2,
        dyn_noise_std=0.0,
        num_belief_discr=3,
        belief_discr_factor=0.1,
        v_max=1.0,
        discount=0.1,
    )
    drones = [Drone(x0=np.array([0.0, 0.0]), v_max=par.v_max, patrol_dir_fn=_patrol_zero) for _ in range(num_drones)]
    prior = Belief(np.zeros(2), np.eye(2))
    s = DroneProblemState(
        prior=prior.copy(),
        belief=prior.copy(),
        drones=drones,
    )
    pol = _ZeroPollutant()
    problem = ProblemSpec(
        pollutant=pol,
        domain=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        wind=np.zeros(2, dtype=float),
        c1=1.0,
        c2=0.0,
        c3=0.0,
        c4=0.0,
        patrol=_patrol_zero,
        collision=None,
    )
    return par, problem, s


@pytest.mark.parametrize(
    "planner_factory",
    [
        lambda: FiniteHorizonSingleDronePlanner(wind_aligned=False),
        lambda: FiniteHorizonTwoDronesPlanner(wind_aligned=False),
        lambda: InfiniteHorizonSingleDronePlanner(
            wind_aligned=False,
            fp_solver=PicardFixedPointSolver(tol=1e-6, max_iter=8, verbose=False),
        ),
        lambda: InfiniteHorizonTwoDronesPlanner(
            wind_aligned=False,
            fp_solver=PicardFixedPointSolver(tol=1e-6, max_iter=8, verbose=False),
        ),
    ],
)
def test_canonical_zero_problem_yields_small_controls(planner_factory):
    """
    Reference/regression test with a real tiny HJB solve.

    Canonical setup has zero wind, zero patrol target, zero collision/no-fly,
    and a pollutant that yields zero information gain. Expected controls should
    remain very small across planners.
    """
    num_drones = 2
    par, problem, s = _make_canonical_problem(num_drones=num_drones)
    planner = planner_factory()

    planner.plan(0.0, par, problem, s)
    alpha = planner.compute_alpha(0.25, par, problem, s)

    assert alpha.shape == (num_drones, 2)
    assert np.all(np.isfinite(alpha))

    norms = np.linalg.norm(alpha, axis=1)
    assert np.all(norms <= par.v_max + 1e-12)

    # Regression envelope for canonical near-zero-control setup.
    assert float(np.max(np.abs(alpha))) < 5e-2
