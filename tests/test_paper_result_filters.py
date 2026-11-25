from experiments.paper_result_filters import (
    accepted_fixed_point_rows,
    fixed_point_policy_accepted,
    fixed_point_policy_status,
)


def test_result_filter_rejects_explicit_failure():
    assert not fixed_point_policy_accepted({"fp_all_converged": 0})
    assert fixed_point_policy_status({"fp_failed_solve_count": "2"}) == "rejected"


def test_result_filter_keeps_but_labels_legacy_rows_unverified():
    legacy = {"method": "order1", "final_error": 1.0}
    assert fixed_point_policy_status(legacy) == "unverified"
    assert accepted_fixed_point_rows([legacy]) == [legacy]
