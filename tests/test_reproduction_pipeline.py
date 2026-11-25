import csv
import json
import subprocess
import sys

from experiments.audit_paper_runs import audit_results
from experiments.exp_wind_aligned_comparison import _write_rows
from experiments.result_io import write_csv_atomic, write_json_atomic
from experiments.validate_paper_results import CsvContract, validate_paper_results


def test_atomic_writers_replace_complete_files(tmp_path):
    csv_path = tmp_path / "rows.csv"
    json_path = tmp_path / "row.json"

    write_csv_atomic(csv_path, [{"a": 1}, {"a": 2, "b": 3}])
    write_json_atomic(json_path, {"ok": True})

    with csv_path.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [
            {"a": "1", "b": ""},
            {"a": "2", "b": "3"},
        ]
    assert json.loads(json_path.read_text()) == {"ok": True}
    assert not list(tmp_path.glob(".*.tmp"))


def test_wind_rows_are_keyed_and_written_without_duplicates(tmp_path):
    path = tmp_path / "wind.csv"
    rows = {
        (0, True): {"seed": 0, "wind_aligned": True, "value": 1},
        (0, False): {"seed": 0, "wind_aligned": False, "value": 2},
    }

    _write_rows(str(path), rows)

    with path.open(newline="") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 2
    assert {(row["seed"], row["wind_aligned"]) for row in written} == {
        ("0", "True"),
        ("0", "False"),
    }


def test_result_validator_reports_missing_and_duplicate_cases(tmp_path, monkeypatch):
    result_dir = tmp_path / "sample"
    result_dir.mkdir()
    path = result_dir / "rows.csv"
    path.write_text("id,value\n0,1\n0,2\n")
    contract = CsvContract(
        "sample/rows.csv",
        frozenset({"id", "value"}),
        lambda row: int(row["id"]),
        frozenset({0, 1}),
    )
    monkeypatch.setattr(
        "experiments.validate_paper_results._contracts", lambda: (contract,)
    )

    errors = validate_paper_results(tmp_path)

    assert any("duplicate" in error for error in errors)
    assert any("missing" in error for error in errors)


def test_audit_treats_string_zero_as_failed(tmp_path):
    case = tmp_path / "scaling_M" / "M2" / "seed_000"
    case.mkdir(parents=True)
    (case / "metrics.json").write_text(
        json.dumps({
            "fp_all_converged": "0",
            "fp_failed_solve_count": 1,
            "fp_retry_solve_count": 1,
        })
    )

    report = audit_results(tmp_path)

    assert report["failed_fixed_point_case_count"] == 1


def test_audit_script_runs_by_file_path(tmp_path):
    output = tmp_path / "audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/audit_paper_runs.py",
            "--results-dir",
            str(tmp_path),
            "--out",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
