"""The day-block bootstrap: honest about luck, deterministic, and safe on thin data."""

import numpy as np

from parkwaits.uncertainty import day_bootstrap_gap


def _data(n_days=10, per_day=400, edge=1.0, seed=0):
    rng = np.random.default_rng(seed)
    days = np.repeat(np.arange(n_days), per_day).astype(str)
    y = (rng.random(len(days)) < 0.2).astype(int)
    noise = rng.normal(size=len(days))
    better = y * edge + noise  # carries signal
    other = rng.normal(size=len(days))  # pure noise
    return y, better, other, days


def test_a_real_lead_has_an_interval_above_zero():
    iv = day_bootstrap_gap(*_data(edge=2.0), draws=100)
    assert iv["low"] > 0 and iv["gap"] > 0 and iv["share_ahead"] == 1.0


def test_no_lead_has_an_interval_that_includes_zero():
    y, _, other, days = _data()
    rng = np.random.default_rng(5)
    iv = day_bootstrap_gap(y, other, rng.normal(size=len(y)), days, draws=100)
    assert iv["low"] < 0 < iv["high"]


def test_same_seed_gives_the_same_interval_and_whole_days_are_resampled():
    a = day_bootstrap_gap(*_data(), draws=50, seed=3)
    b = day_bootstrap_gap(*_data(), draws=50, seed=3)
    assert a == b and a["days"] == 10


def test_too_few_days_or_one_class_gives_none():
    y, a, b, days = _data(n_days=2)
    assert day_bootstrap_gap(y, a, b, days) is None
    y, a, b, days = _data()
    assert day_bootstrap_gap(np.zeros_like(y), a, b, days) is None
