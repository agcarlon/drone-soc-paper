import numpy as np

from drone_soc import (
    Params,
    ProblemSpec,
    Belief,
    DroneProblemState,
    Drone,
)
from _drone_soc_internal import construct_hjb_grid_axes, default_fixed_state_indices
from drone_soc import _save_run_config


class _DummyDrone(Drone):
    def patrolling_trajectory(self, t: float):
        return self.x


def _make_state(use_intensity: bool):
    q_log_prior = 0.1
    q_log_var_prior = 0.3
    par = Params(dx=1.0, num_belief_discr=3, belief_discr_factor=0.2)
    s = DroneProblemState(
        prior=Belief(
            np.zeros(2, dtype=float),
            np.eye(2),
            q_log=q_log_prior if use_intensity else None,
            q_log_var=q_log_var_prior if use_intensity else None,
        ),
        belief=Belief(
            np.zeros(2, dtype=float),
            np.eye(2),
            q_log=q_log_prior if use_intensity else None,
            q_log_var=q_log_var_prior if use_intensity else None,
        ),
        drones=[_DummyDrone(x0=np.zeros(2), v_max=1.0)],
    )
    problem = ProblemSpec(
        pollutant=object(),
        domain=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        wind=np.zeros(2, dtype=float),
    )
    return par, problem, s


def test_construct_hjb_grid_axes_includes_intensity_axis():
    par, problem, s = _make_state(use_intensity=True)
    axes = construct_hjb_grid_axes(par, problem, s, num_drones=1, wind_aligned=False)

    # Expect: 2 position axes + 2 eigenvalue axes + angle axis + intensity axis
    assert len(axes) == 6
    intensity_axis = axes[-1]
    assert np.allclose(intensity_axis[-1], np.log(s.belief.q_log_var))


def test_default_fixed_state_indices_skips_spatial_axes():
    par, problem, s = _make_state(use_intensity=True)
    axes = construct_hjb_grid_axes(par, problem, s, num_drones=1, wind_aligned=False)
    spatial_axes = (0, 1)
    fixed = default_fixed_state_indices(axes, spatial_axes)

    assert 0 not in fixed and 1 not in fixed
    for idx, axis in enumerate(axes):
        if idx in spatial_axes:
            continue
        assert fixed[idx] == len(axis) // 2


def test_run_config_distinguishes_nominal_and_actual_spacing(tmp_path):
    par, problem, s = _make_state(use_intensity=False)
    par.output_dir = str(tmp_path)

    class Planner:
        pass

    _save_run_config(par, problem, s, Planner())
    import json
    payload = json.loads((tmp_path / "run_config.json").read_text())
    primary = payload["hjb_grid_resolution"][0]
    assert primary["nominal_dx"] == 1.0
    assert primary["node_counts"] == [2, 2]
    assert primary["actual_spacings"] == [2.0, 2.0]
