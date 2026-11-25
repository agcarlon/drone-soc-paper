"""Compatibility-aware acceptance policy for paper experiment rows."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


_FALSE_STRINGS = {"0", "false", "no", "n", "failed", "failure"}
_TRUE_STRINGS = {"1", "true", "yes", "y", "passed", "success"}


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned:
            return None
        if cleaned == "nan":
            return None
        if cleaned in _FALSE_STRINGS:
            return False
        if cleaned in _TRUE_STRINGS:
            return True
        try:
            numeric = float(cleaned)
        except ValueError as error:
            raise ValueError(f"invalid Boolean value {value!r}") from error
        if math.isnan(numeric):
            return None
        if numeric == 0.0:
            return False
        if numeric == 1.0:
            return True
        raise ValueError(f"invalid Boolean value {value!r}")
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (bool, int, float)):
        if value == 0:
            return False
        if value == 1:
            return True
    raise ValueError(f"invalid Boolean value {value!r}")


def fixed_point_policy_status(row: Mapping[str, Any]) -> str:
    """Return ``accepted``, ``rejected``, or ``unverified`` for one result row.

    Corrected rows carry ``fp_all_converged`` and are accepted only when every
    solve met its recorded convergence criterion. Older rows without fixed-point
    diagnostics remain readable but must not be represented as verified.
    """

    try:
        converged = _optional_bool(row.get("fp_all_converged"))
    except ValueError as error:
        raise ValueError(
            f"invalid fp_all_converged value {row.get('fp_all_converged')!r}"
        ) from error
    if converged is not None:
        return "accepted" if converged else "rejected"

    failed = row.get("fp_failed_solve_count")
    if failed not in (None, ""):
        try:
            return "accepted" if int(float(failed)) == 0 else "rejected"
        except (TypeError, ValueError):
            pass
    return "unverified"


def fixed_point_policy_accepted(row: Mapping[str, Any]) -> bool:
    """Accept verified successes and backward-compatible unverified legacy rows."""

    return fixed_point_policy_status(row) != "rejected"


def accepted_fixed_point_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if fixed_point_policy_accepted(row)]
