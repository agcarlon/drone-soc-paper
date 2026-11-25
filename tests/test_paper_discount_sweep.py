import math

import pytest

from experiments.exp_paper_discount_sweep import DEFAULT_DISCOUNTS, _runtime


def test_default_discount_grid_increases_per_cycle_factor():
    assert DEFAULT_DISCOUNTS == (0.7, 0.5, 0.3, 0.2, 0.1)

    delta = _runtime().dt_meas
    betas = [math.exp(-rate * delta) for rate in DEFAULT_DISCOUNTS]
    assert betas == sorted(betas)
    assert betas == pytest.approx(
        [
            7.5825604279e-10,
            3.0590232050e-7,
            1.2340980409e-4,
            2.4787521767e-3,
            4.9787068368e-2,
        ]
    )
