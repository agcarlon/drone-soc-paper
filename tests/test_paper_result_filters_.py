import pytest

from experiments.compute_paper_interval_tables import _fixed_point_final_difference
from experiments.paper_result_filters import (
    accepted_fixed_point_rows,
    fixed_point_policy_accepted,
)


@pytest.mark.parametrize("value", [1, "1", 1.0, "1.0", "", None, float("nan")])
def test_fixed_point_policy_accepted_includes_accepted_and_legacy_rows(value):
    assert fixed_point_policy_accepted({"fp_all_converged": value})


@pytest.mark.parametrize("value", [0, "0", 0.0, "0.0"])
def test_fixed_point_policy_accepted_excludes_explicit_failures(value):
    assert not fixed_point_policy_accepted({"fp_all_converged": value})


def test_accepted_fixed_point_rows_preserves_raw_rows_and_order():
    rows = [
        {"seed": 1, "fp_all_converged": "1"},
        {"seed": 2, "fp_all_converged": "0"},
        {"seed": 3},
    ]

    assert accepted_fixed_point_rows(rows) == [rows[0], rows[2]]
    assert len(rows) == 3


def test_fixed_point_policy_accepted_rejects_malformed_values():
    with pytest.raises(ValueError, match="fp_all_converged"):
        fixed_point_policy_accepted({"fp_all_converged": "unknown"})


def test_fixed_point_final_difference_prefers_precise_name():
    row = {
        "fp_max_final_iterate_difference_linf": "0.002",
        "fp_max_final_residual": "0.003",
    }

    assert _fixed_point_final_difference(row) == pytest.approx(0.002)


def test_fixed_point_final_difference_supports_archived_name():
    assert _fixed_point_final_difference({"fp_max_final_residual": "0.003"}) == pytest.approx(
        0.003
    )
