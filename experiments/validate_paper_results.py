"""Validate completeness and uniqueness of the data consumed by paper plots."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Hashable, Iterable


@dataclass(frozen=True)
class CsvContract:
    relative_path: str
    required_columns: frozenset[str]
    key: Callable[[dict[str, str]], Hashable]
    expected_keys: frozenset[Hashable]
    numeric_columns: frozenset[str] = frozenset()


def _read_rows(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader), set(reader.fieldnames)


def _bool(value: str) -> bool:
    cleaned = value.strip().lower()
    if cleaned in {"1", "true", "yes"}:
        return True
    if cleaned in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid Boolean value {value!r}")


def _contracts() -> tuple[CsvContract, ...]:
    seeds_100 = range(100)
    seeds_50 = range(50)
    methods_baseline = ("order1", "greedy_eig", "patrol")
    methods_cooperative = ("order1", "order2")
    snr_levels = (50.0, 100.0, 150.0, 200.0)
    discounts = (0.7, 0.5, 0.3, 0.2, 0.1)
    return (
        CsvContract(
            "wind_aligned_comparison/wind_aligned_comparison.csv",
            frozenset({"seed", "wind_aligned", "dist_final", "energy", "num_plan_calls"}),
            lambda row: (int(row["seed"]), _bool(row["wind_aligned"])),
            frozenset((seed, aligned) for seed in seeds_100 for aligned in (True, False)),
            frozenset({"seed", "dist_final", "energy", "num_plan_calls"}),
        ),
        CsvContract(
            "greedy_eig_comparison/per_trial_results.csv",
            frozenset({"seed", "snr0_target", "method", "final_error", "energy", "success"}),
            lambda row: (float(row["snr0_target"]), int(row["seed"]), row["method"]),
            frozenset(
                (snr, seed, method)
                for snr in snr_levels
                for seed in seeds_50
                for method in methods_baseline
            ),
            frozenset({"seed", "snr0_target", "final_error", "energy", "success"}),
        ),
        CsvContract(
            "paper_pairwise_showcase/per_trial_results.csv",
            frozenset({"seed", "method", "final_error", "energy", "fp_all_converged"}),
            lambda row: (int(row["seed"]), row["method"]),
            frozenset((seed, method) for seed in seeds_100 for method in methods_cooperative),
            frozenset({"seed", "final_error", "energy", "fp_all_converged"}),
        ),
        CsvContract(
            "paper_pairwise_discount_sweep/per_trial_results.csv",
            frozenset({"discount", "seed", "method", "final_error", "energy", "fp_all_converged"}),
            lambda row: (float(row["discount"]), int(row["seed"]), row["method"]),
            frozenset(
                (discount, seed, method)
                for discount in discounts
                for seed in seeds_100
                for method in methods_cooperative
            ),
            frozenset({"discount", "seed", "final_error", "energy", "fp_all_converged"}),
        ),
        CsvContract(
            "scaling_M/per_trial_results.csv",
            frozenset({"num_drones", "seed", "final_error", "energy_per_drone", "fp_all_converged"}),
            lambda row: (int(row["num_drones"]), int(row["seed"])),
            frozenset((num_drones, seed) for num_drones in (2, 3, 4) for seed in seeds_50),
            frozenset({"num_drones", "seed", "final_error", "energy_per_drone", "fp_all_converged"}),
        ),
        CsvContract(
            "greedy_eig_comparison/summary_results.csv",
            frozenset({"method", "snr0", "num_trials", "success_rate", "median_energy"}),
            lambda row: (float(row["snr0"]), row["method"]),
            frozenset((snr, method) for snr in snr_levels for method in methods_baseline),
            frozenset({"snr0", "num_trials", "success_rate", "median_energy"}),
        ),
        CsvContract(
            "paper_pairwise_showcase/summary_results.csv",
            frozenset({"method", "num_trials", "median_final_error", "median_energy"}),
            lambda row: row["method"],
            frozenset(methods_cooperative),
            frozenset({"num_trials", "median_final_error", "median_energy"}),
        ),
        CsvContract(
            "paper_pairwise_discount_sweep/summary_results.csv",
            frozenset({"discount", "method", "num_trials", "median_final_error", "median_energy"}),
            lambda row: (float(row["discount"]), row["method"]),
            frozenset((discount, method) for discount in discounts for method in methods_cooperative),
            frozenset({"discount", "num_trials", "median_final_error", "median_energy"}),
        ),
        CsvContract(
            "scaling_M/summary_results.csv",
            frozenset({"num_drones", "num_trials", "median_final_error", "median_energy_per_drone"}),
            lambda row: int(row["num_drones"]),
            frozenset({2, 3, 4}),
            frozenset({"num_drones", "num_trials", "median_final_error", "median_energy_per_drone"}),
        ),
    )


def _duplicates(keys: Iterable[Hashable]) -> set[Hashable]:
    seen: set[Hashable] = set()
    duplicates: set[Hashable] = set()
    for key in keys:
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return duplicates


def validate_paper_results(results_dir: str | Path) -> list[str]:
    root = Path(results_dir)
    errors: list[str] = []
    for contract in _contracts():
        path = root / contract.relative_path
        try:
            rows, columns = _read_rows(path)
        except (OSError, ValueError) as error:
            errors.append(str(error))
            continue

        missing_columns = contract.required_columns - columns
        if missing_columns:
            errors.append(f"{path}: missing columns {sorted(missing_columns)}")
            continue
        blank_fields = [
            (index + 2, column)
            for index, row in enumerate(rows)
            for column in contract.required_columns
            if str(row.get(column, "")).strip() == ""
        ]
        if blank_fields:
            errors.append(f"{path}: {len(blank_fields)} required values are blank")
            continue
        try:
            nonfinite = [
                (index + 2, column)
                for index, row in enumerate(rows)
                for column in contract.numeric_columns
                if not math.isfinite(float(row[column]))
            ]
        except (TypeError, ValueError) as error:
            errors.append(f"{path}: invalid numeric field: {error}")
            continue
        if nonfinite:
            errors.append(f"{path}: {len(nonfinite)} required numeric values are non-finite")
            continue
        try:
            keys = [contract.key(row) for row in rows]
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{path}: invalid key field: {error}")
            continue

        duplicate_keys = _duplicates(keys)
        actual_keys = set(keys)
        missing_keys = contract.expected_keys - actual_keys
        unexpected_keys = actual_keys - contract.expected_keys
        if duplicate_keys:
            errors.append(f"{path}: {len(duplicate_keys)} duplicate case keys")
        if missing_keys:
            errors.append(f"{path}: {len(missing_keys)} expected cases are missing")
        if unexpected_keys:
            errors.append(f"{path}: {len(unexpected_keys)} unexpected cases are present")

    required_artifacts = []
    for snr in (50.0, 100.0, 150.0, 200.0):
        for seed in range(50):
            for method in ("order1", "greedy_eig", "patrol"):
                case = root / "greedy_eig_comparison" / f"snr0_{snr:.1f}" / f"seed_{seed:03d}" / method
                required_artifacts.extend((case / "run_config.json", case / "run_results.npz"))
    for seed in range(100):
        for method in ("order1", "order2"):
            case = root / "paper_pairwise_showcase" / f"seed_{seed:03d}" / method
            required_artifacts.extend((case / "run_config.json", case / "run_results.npz"))
    missing_artifacts = [path for path in required_artifacts if not path.is_file()]
    if missing_artifacts:
        errors.append(
            f"{root}: {len(missing_artifacts)} run artifacts required by paper figures are missing"
        )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    errors = validate_paper_results(args.results_dir)
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise SystemExit(f"Paper result validation failed:\n{details}")
    print("Validated complete, unique paper result tables.")


if __name__ == "__main__":
    main()
