"""Focused checks for the review's behavior-preserving A1 and A2 revisions."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from drone_soc import (
    Belief, Drone, DroneProblemState, Params, ProblemSpec,
    SigmoidCollisionAvoidance, compute_collision_gradients, _save_run_config,
)
from finite_horizon import FiniteHorizonSingleDronePlanner
from greedy_eig import GreedyEIGPlanner, _clip_to_vmax
from experiments.baseline_scenario import PatrolOnlyPlanner
import infinite_horizon as ih


def case(distance=1.5):
    problem = ProblemSpec(
        pollutant=SimpleNamespace(q=1.0), wind=np.zeros(2),
        c1=0.1, c2=0.2, c3=0.7,
        patrol=lambda x, t: np.array([0.3, -0.2]),
        collision=SigmoidCollisionAvoidance(d_min=1., d_max=2., max_value=10.),
    )
    belief = Belief(np.zeros(2), np.eye(2), q_log=0., q_log_var=1.)
    state = DroneProblemState(belief.copy(), belief, [
        Drone(np.array([0., 0.])), Drone(np.array([distance, 0.])),
        Drone(np.array([0., distance])),
    ])
    par = Params(t_end=60., meas_times=np.array([0., 30., 60.]), v_max=1.)
    return par, problem, state


@pytest.mark.parametrize('distance', [0., 1.5, 20.])
@pytest.mark.parametrize('kind', ['one', 'pair', 'finite_one', 'greedy', 'patrol'])
def test_collision_time_scale_preserves_historical_commands(monkeypatch, kind, distance):
    par, problem, state = case(distance)
    if kind == 'one':
        planner = ih.InfiniteHorizonSingleDronePlanner(wind_aligned=True)
        planner._feedback = lambda t, coords: np.array([0.4, 0.1])
    elif kind == 'pair':
        planner = ih.InfiniteHorizonTwoDronesPlanner(wind_aligned=True)
        planner._time_axis = np.array([0., 30.])
        planner._grid_axes = (np.array([-1., 1.]),) * 7
        planner._grad_pair_0 = planner._grad_pair_1 = np.zeros((2,) * 8)
        planner._belief_mu_plan = np.zeros(2)
        monkeypatch.setattr(ih, 'n_linear_interpolate', lambda *a, **kw: np.zeros(1))
    elif kind == 'finite_one':
        planner = FiniteHorizonSingleDronePlanner(wind_aligned=True)
        planner._feedback = lambda t, coords: np.array([0.4, 0.1])
    elif kind == 'greedy':
        planner = GreedyEIGPlanner()
        planner._target_positions = state.drone_positions + np.array([0.2, -0.4])
    else:
        planner = PatrolOnlyPlanner()

    c3 = problem.c3
    problem.c3 = 0.
    nominal = planner.compute_alpha(0., par, problem, state)
    problem.c3 = c3
    gradient = compute_collision_gradients(state.drone_positions, problem.collision)
    historical_raw = nominal - (c3 / (problem.c1 + problem.c2)) * gradient

    def project(a):
        if kind == 'greedy':
            return np.array([_clip_to_vmax(x, par.v_max) for x in a])
        norms = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
        return a * np.minimum(1., par.v_max / norms)

    original = project(historical_raw)
    actual = planner.compute_alpha(0., par, problem, state)
    np.testing.assert_allclose(actual, original, rtol=1e-12, atol=1e-12)
    assert np.max(np.linalg.norm(actual, axis=1)) <= par.v_max + 1e-12
    problem.tau_col = 0.25
    expected = project(nominal - 0.25 * c3 / (problem.c1 + problem.c2) * gradient)
    np.testing.assert_allclose(planner.compute_alpha(0., par, problem, state), expected,
                               rtol=1e-12, atol=1e-12)
    if distance == 0.:
        np.testing.assert_array_equal(gradient, 0.)
    elif distance == 1.5:
        assert historical_raw[0, 0] < nominal[0, 0]  # repulsion from right neighbor
        assert np.max(np.linalg.norm(historical_raw, axis=1)) > par.v_max


def test_tau_col_metadata_and_historical_default(tmp_path):
    par, problem, state = case()
    assert problem.tau_col == 1.0
    problem.tau_col = 2.5
    par.output_dir = str(tmp_path)
    _save_run_config(par, problem, state, PatrolOnlyPlanner())
    cfg = json.loads((tmp_path / 'run_config.json').read_text())
    assert cfg['problem']['tau_col'] == 2.5
    # Historic constructors/configurations omit this new optional field.
    assert ProblemSpec(pollutant=problem.pollutant).tau_col == 1.0


@pytest.mark.parametrize('value', [0., -1., np.inf, np.nan])
def test_tau_col_requires_a_positive_finite_time(value):
    with pytest.raises(ValueError, match='tau_col'):
        ProblemSpec(pollutant=SimpleNamespace(), tau_col=value)


@pytest.mark.parametrize('planner_type', [ih.InfiniteHorizonSingleDronePlanner,
                                        ih.InfiniteHorizonTwoDronesPlanner])
def test_virtual_cycles_replay_phase_and_real_replanning_advances(monkeypatch, planner_type):
    par, problem, state = case()
    seen = []
    phase_values = []
    problem.patrol = lambda x, t: np.stack([
        0.1 + np.zeros_like(x[0]),
        0.2 * np.sin(2*np.pi*t/220.) + np.zeros_like(x[0]),
    ])
    monkeypatch.setattr(ih, 'construct_hjb_grid_axes', lambda p, q, s, n, w:
                        (np.linspace(-1., 1., 3),) * (2*n+3))

    def fake_hjb(oc):
        seen.append((oc.T0, oc.T))
        mesh = np.stack(np.meshgrid(*oc.grid_axes, indexing='ij'))
        a, _ = oc.analytic_minimizer(mesh, np.zeros_like(mesh), oc.T0)
        # Exercise the actual running-cost closure at the current interval's start.
        phase_values.append(float(np.mean(oc.running_cost(mesh, 0.*a, oc.T0))))
        return lambda t, c: np.zeros(2), np.zeros((2,) + mesh.shape[1:]), oc.grid_axes

    class Solver:
        tol = 1e-3
        max_iter = 3

        def solve(self, function, x0=None):
            first = function(None)
            second = function(first)
            return second, [{'linf': 0., 'residual_linf': 0.}]

    monkeypatch.setattr(ih, 'solve_hjb', fake_hjb)
    planner = planner_type(wind_aligned=True, fp_solver=Solver())
    planner.plan(0., par, problem, state)
    planner.plan(30., par, problem, state)  # identical reduced state, new physical phase
    assert seen == [(0., 30.)] * 3 + [(30., 60.)] * 3
    np.testing.assert_allclose(phase_values[:3], phase_values[0])
    np.testing.assert_allclose(phase_values[3:], phase_values[3])
    assert not np.isclose(phase_values[0], phase_values[3])
