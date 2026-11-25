"""Shared inferential statistics used by the paper plotting scripts.

The experiment drivers write one row per seed.  Plotting scripts import the
deterministic BCa interval helper from this module so that every paper figure
uses the same estimator, confidence level, and number of resamples.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import bootstrap


BOOTSTRAP_SEED = 100
BOOTSTRAP_RESAMPLES = 10_000


def _bca_interval(
    values: Iterable[float] | np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return the sample median and its two-sided 95% BCa interval."""
    sample = np.asarray(values, dtype=float).reshape(-1)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return float("nan"), float("nan"), float("nan")

    estimate = float(np.median(sample))
    if sample.size == 1 or np.all(sample == sample[0]):
        return estimate, estimate, estimate

    result = bootstrap(
        (sample,),
        np.median,
        confidence_level=0.95,
        n_resamples=BOOTSTRAP_RESAMPLES,
        method="BCa",
        rng=rng,
    )
    return (
        estimate,
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )
