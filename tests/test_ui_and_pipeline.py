"""
Integration tests for UI and real-time processing pipeline.
"""

import time
import pytest
import numpy as np
from PySide6.QtWidgets import QApplication

from config import AppConfig
from core.coordinate_mapper import CoordinateMapper
from core.cursor_controller import CursorController
from core.hand_tracker import HandTracker
from utils.performance_monitor import PerformanceMonitor


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_pipeline_processing_benchmark():
    config = AppConfig()
    tracker = HandTracker(config.model_path)
    cursor = CursorController(config.smoothing)
    mapper = CoordinateMapper(config.active_zone, cursor.screen_width, cursor.screen_height)
    perf_monitor = PerformanceMonitor()

    # Generate synthetic frames
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    latencies = []
    for i in range(20):
        t_cap = time.perf_counter()
        res = tracker.process_frame(frame)

        is_inside = False
        coords = None
        if res.hand_detected and res.index_tip_norm:
            is_inside = mapper.is_inside_zone(*res.index_tip_norm)
            if is_inside:
                coords = mapper.map_to_screen(*res.index_tip_norm)

        cursor.update_position(coords, t_cap)
        t_fin = time.perf_counter()

        perf_monitor.record_frame(t_cap, t_fin, res.inference_time_ms)
        latencies.append((t_fin - t_cap) * 1000.0)

    avg_lat = np.mean(latencies)
    print(f"\n[Benchmark] Synthetic pipeline average software latency: {avg_lat:.2f} ms")
    assert avg_lat < 50.0  # Must be strictly under 50 ms!
    tracker.close()
    del tracker
    import gc
    gc.collect()


def test_main_window_lifecycle(qapp):
    from ui.main_window import MainWindow
    config = AppConfig()
    window = MainWindow(config)
    window.show()
    qapp.processEvents()

    # Verify initial safe state: Gesture control must be OFF
    assert window.cursor.is_active is False
    assert window.btn_start.isEnabled() is True
    assert window.btn_stop.isEnabled() is False

    # Test toggling
    window.start_gesture_control()
    assert window.cursor.is_active is True
    assert window.btn_start.isEnabled() is False
    assert window.btn_stop.isEnabled() is True

    window.stop_gesture_control()
    assert window.cursor.is_active is False
    assert window.btn_start.isEnabled() is True
    assert window.btn_stop.isEnabled() is False

    # Allow worker thread loop to start and run a moment, then close cleanly
    time.sleep(0.15)
    qapp.processEvents()
    window.close()
    qapp.processEvents()
    window.deleteLater()
    qapp.processEvents()
    del window
