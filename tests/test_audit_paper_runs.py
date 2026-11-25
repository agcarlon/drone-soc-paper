import json

from experiments.audit_paper_runs import audit_results


def test_audit_results_reports_fp_failures_and_boundary_contact(tmp_path):
    case = tmp_path / "paper_pairwise_discount_sweep" / "discount_0p100" / "seed_008" / "order2"
    case.mkdir(parents=True)
    (case / "metrics.json").write_text(json.dumps({
        "method": "order2",
        "fp_all_converged": 0,
        "fp_failed_solve_count": 1,
        "fp_retry_solve_count": 1,
        "fp_max_final_iterate_difference_linf": 0.002,
        "boundary_dwell_fraction": 0.01,
    }))

    report = audit_results(tmp_path)

    assert report["failed_fixed_point_case_count"] == 1
    assert report["retry_case_count"] == 1
    assert report["failed_fixed_point_cases"][0]["failed_solve_count"] == 1
    assert report["boundary_summary_by_method"]["order2"]["max_dwell_fraction"] == 0.01
