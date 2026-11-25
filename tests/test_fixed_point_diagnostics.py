import json
from types import SimpleNamespace

from experiments.fixed_point_diagnostics import fixed_point_metrics


def test_fixed_point_metrics_persist_attempt_details(tmp_path):
    solver = SimpleNamespace(
        diagnostics_history=[
            {
                "converged": True,
                "final_linf": 8e-4,
                "final_iterate_difference_linf": 8e-4,
                "criterion": "successive_iterate_linf",
                "tolerance": 1e-3,
                "iterations": 4,
                "total_iterations": 16,
                "attempts": 2,
                "attempt_details": [
                    {"dt_hjb": 1.0, "iterations": 12, "final_linf": 2e-3, "converged": False},
                    {"dt_hjb": 0.5, "iterations": 4, "final_linf": 8e-4, "converged": True},
                ],
                "retry_used": True,
                "dt_hjb": 0.5,
                "t_start": 0.0,
                "t_end": 30.0,
            }
        ]
    )

    metrics = fixed_point_metrics(solver, tmp_path)

    assert metrics["fp_all_converged"] == 1
    assert metrics["fp_retry_solve_count"] == 1
    assert metrics["fp_failed_solve_count"] == 0
    assert metrics["fp_max_final_iterate_difference_linf"] == 8e-4
    payload = json.loads((tmp_path / "fixed_point_diagnostics.json").read_text())
    assert payload["solves"][0]["attempt_details"][1]["dt_hjb"] == 0.5
