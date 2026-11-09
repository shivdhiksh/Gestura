"""
Unit tests for One Euro Filter and adaptive smoothing.
"""

import pytest
import time
from core.smoothing import LowPassFilter, OneEuroFilter, OneEuroFilter2D


def test_low_pass_filter_initialization():
    lpf = LowPassFilter(alpha=0.5)
    assert lpf.last_value is None
    # First value should pass through
    assert lpf.filter(10.0) == 10.0
    # Second value with alpha=0.5 should be average
    assert lpf.filter(20.0) == 15.0


def test_low_pass_filter_reset():
    lpf = LowPassFilter(alpha=0.5)
    lpf.filter(10.0)
    assert lpf.last_value == 10.0
    lpf.reset()
    assert lpf.last_value is None
    assert lpf.filter(50.0) == 50.0


def test_one_euro_filter_stability():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.01)
    t = 1.0

    # Constant signal should remain constant
    assert f.filter(100.0, timestamp=t) == 100.0

    for i in range(1, 20):
        t += 0.033  # ~30 FPS
        val = f.filter(100.0, timestamp=t)
        assert abs(val - 100.0) < 1e-4


def test_one_euro_filter_jitter_reduction():
    # Low speed with alternating jitter (-1, +1)
    f = OneEuroFilter(min_cutoff=0.8, beta=0.005)
    t = 1.0
    f.filter(500.0, timestamp=t)

    outputs = []
    for i in range(1, 30):
        t += 0.033
        jitter = 5.0 if i % 2 == 0 else -5.0
        outputs.append(f.filter(500.0 + jitter, timestamp=t))

    # Variance of filtered signal should be substantially lower than input jitter (5.0)
    deviations = [abs(v - 500.0) for v in outputs[5:]]
    assert max(deviations) < 3.5


def test_one_euro_filter_high_speed_tracking():
    # High speed step movement
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    t = 1.0
    f.filter(0.0, timestamp=t)

    # Fast jump to 1000
    t += 0.033
    val1 = f.filter(1000.0, timestamp=t)
    # Because beta expands cutoff during high speed, it should adapt quickly
    assert val1 > 400.0


def test_one_euro_filter_zero_dt_handling():
    f = OneEuroFilter()
    t = 1.0
    assert f.filter(10.0, timestamp=t) == 10.0
    # Zero or negative dt must not crash or divide by zero
    val = f.filter(20.0, timestamp=t)
    assert val == 10.0


def test_one_euro_filter_2d():
    f2d = OneEuroFilter2D(min_cutoff=1.2, beta=0.02)
    x, y = f2d.filter(100.0, 200.0, timestamp=1.0)
    assert x == 100.0 and y == 200.0

    f2d.reset()
    assert f2d.filter_x.last_value is None
    assert f2d.filter_y.last_value is None

