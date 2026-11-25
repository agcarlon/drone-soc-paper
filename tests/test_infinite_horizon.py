import numpy as np

from drone_soc import (
    Params,
    ProblemSpec,
    Belief,
    DroneProblemState,
    Drone,
)
from infinite_horizon import (
    InfiniteHorizonSingleDronePlanner,
    InfiniteHorizonTwoDronesPlanner,
    _solve_infinite_hjb_nd_direction,
)
from fixed_point import PicardFixedPointSolver


class _DummyDrone(Drone):
    def patrolling_trajectory(self, t: float):
        return self.x


class _DummyPollutant:
    def get_pollutant_concentration(self, x, t, theta, intensity=None):
        x = np.asarray(x)
        return np.zeros(x.shape[0], dtype=float)

    def get_concentration_gradient_wrt_theta(self, x, t, theta, intensity=None):
        x = np.atleast_2d(x)
        return np.zeros((x.shape[0], 2), dtype=float)

    def get_concentration_derivative_wrt_logq(self, x, t, theta, intensity=None):
        x = np.atleast_2d(x)
        return np.zeros(x.shape[0], dtype=float)


def _make_small_infinite_problem(num_drones: int):
    par = Params(
        t_0=0.0,
        t_end=1.0,
        dt=0.5,
        dt_hjb=0.5,
        meas_times=np.array([0.0, 0.5]),
        dx=0.5,
        noise_std=1e-3,
        dyn_noise_std=0.0,
        num_belief_discr=3,
        belief_discr_factor=0.1,
        discount=0.1,
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


def test_infinite_horizon_single_planner_smoke():
    """
    InfiniteHorizonSingleDronePlanner should solve a tiny HJB and produce finite controls.
    """
    par, problem, s = _make_small_infinite_problem(num_drones=2)

    planner = InfiniteHorizonSingleDronePlanner(
        wind_aligned=False, fp_solver=None
    )
    planner.plan(0.0, par, problem, s)
    alpha = planner.compute_alpha(0.0, par, problem, s)

    assert alpha.shape == (len(s.drones), 2)
    assert np.all(np.isfinite(alpha))


def test_infinite_horizon_two_drones_planner_smoke():
    """
    InfiniteHorizonTwoDronesPlanner should solve a tiny HJB and produce finite controls.
    """
    par, problem, s = _make_small_infinite_problem(num_drones=3)

    planner = InfiniteHorizonTwoDronesPlanner(
        wind_aligned=False, fp_solver=None
    )
    planner.plan(0.0, par, problem, s)
    alpha = planner.compute_alpha(0.0, par, problem, s)

    assert alpha.shape == (len(s.drones), 2)
    assert np.all(np.isfinite(alpha))


def test_infinite_horizon_records_fixed_point_diagnostics():
    par, problem, s = _make_small_infinite_problem(num_drones=1)
    solver = PicardFixedPointSolver(tol=1e-3, max_iter=5)
    planner = InfiniteHorizonSingleDronePlanner(
        wind_aligned=False,
        fp_solver=solver,
    )

    planner.plan(0.0, par, problem, s)

    assert len(solver.diagnostics_history) == 1
    diagnostic = solver.diagnostics_history[0]
    assert diagnostic["iterations"] >= 1
    assert diagnostic["total_iterations"] >= diagnostic["iterations"]
    assert np.isfinite(diagnostic["final_linf"])
    assert diagnostic["criterion"] == "normalized_bellman_residual_linf"
    assert diagnostic["tolerance"] == solver.tol
    assert diagnostic["final_normalized_residual_linf"] == diagnostic["final_linf"]
    assert np.isfinite(diagnostic["final_residual_linf"])
    assert np.isfinite(diagnostic["final_iterate_difference_linf"])
    assert diagnostic["attempts"] == len(diagnostic["attempt_details"])
    assert diagnostic["attempt_details"][0]["dt_hjb"] == par.dt_hjb


def test_pairwise_fixed_point_and_final_pass_evolve_all_spatial_axes(monkeypatch):
    par, problem, s = _make_small_infinite_problem(num_drones=2)
    observed = []

    def fake_solve(oc):
        observed.append(oc.num_spatial_axes)
        shape = tuple(len(axis) for axis in oc.grid_axes)
        terminal = np.asarray(oc.terminal_cost(np.stack(np.meshgrid(*oc.grid_axes, indexing="ij"))))
        values = np.stack([terminal, terminal])
        return lambda t, x: np.zeros(4), values, oc.grid_axes

    monkeypatch.setattr("infinite_horizon.solve_hjb", fake_solve)
    _solve_infinite_hjb_nd_direction(
        (0.0, 0.5), par, problem, s, order=2, wind_aligned=False,
        fp_solver=PicardFixedPointSolver(tol=1e-3, max_iter=1),
    )
    assert observed
    assert set(observed) == {4}
