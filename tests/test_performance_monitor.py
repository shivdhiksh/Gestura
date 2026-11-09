"""
Unit tests for Performance Monitor utility.
"""

import pytest
import time
from utils.performance_monitor import PerformanceMonitor


def test_perf_monitor_initial_state():
    pm = PerformanceMonitor(window_size=10)
    assert pm.fps == 0.0
    assert pm.latency_ms == 0.0
    assert pm.p95_latency_ms == 0.0
    assert pm.inference_time_ms == 0.0
    assert pm.cursor_rate == 0.0


def test_perf_monitor_latency_and_fps():
    pm = PerformanceMonitor(window_size=15)

    base_time = 100.0
    # Simulate 15 frames at 30 FPS (dt = 0.0333s) with varying latency
    for i in range(15):
        t_cap = base_time + (i * 0.0333)
        t_fin = t_cap + (0.015 + (i * 0.001)) # 15 to 29 ms latency
        pm.record_frame(capture_timestamp=t_cap, finish_timestamp=t_fin, inference_time_ms=12.0)
        pm.record_cursor_update(t_fin)

    # FPS should be approximately 30
    assert 25.0 <= pm.fps <= 35.0
    assert pm.latency_ms > 0
    assert pm.p95_latency_ms >= pm.latency_ms
    assert pm.cursor_rate > 0
    assert pm.cpu_percent >= 0.0
    assert pm.memory_mb > 0.0

    d = pm.get_telemetry_dict()
    assert "p95_latency_ms" in d
    assert "cpu_percent" in d
    assert "memory_mb" in d
