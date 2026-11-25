import math

import experiments.compute_paper_interval_tables as interval_tables


def test_fixed_point_comparison_handles_no_rejected_runs(monkeypatch):
    accepted = {
        "method": "order2",
        "discount": "0.1",
        "fp_all_converged": "1",
        "stopped": "1",
        "success_5m": "1",
        "final_error": "1.0",
        "final_trace_sigma": "2.0",
        "energy": "3.0",
        "travel_distance": "4.0",
        "runtime_sec": "5.0",
        "fp_max_final_iterate_difference_linf": "0.0005",
    }
    monkeypatch.setattr(interval_tables, "_read_csv", lambda _path: [accepted])

    rows = interval_tables._discount_fixed_point_comparison_rows()

    assert rows[0]["status"] == "accepted"
    assert rows[0]["n"] == 1
    assert rows[1]["status"] == "rejected"
    assert rows[1]["n"] == 0
    assert math.isnan(rows[1]["max_final_iterate_difference_linf"])
