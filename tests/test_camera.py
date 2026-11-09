"""
Unit tests for CameraManager and hardware video acquisition.
"""

import pytest
import time
from config import CameraConfig
from core.camera import CameraManager


def test_camera_manager_initialization():
    cfg = CameraConfig(device_index=0, frame_width=640, frame_height=480, fps=60)
    cam = CameraManager(cfg, mirror=True)
    assert cam.config.frame_width == 640
    assert cam.config.frame_height == 480
    assert cam.is_connected is False


def test_camera_manager_start_and_frame_acquisition():
    cfg = CameraConfig(device_index=0, frame_width=640, frame_height=480, fps=60, backend="AUTO")
    cam = CameraManager(cfg, mirror=True)

    success = cam.start()
    assert success is True
    assert cam.is_connected is True
    assert cam.active_backend in ("MSMF", "DirectShow", "Default")

    # Wait for first frame (up to 2.0s)
    frame_data = None
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) < 2.0:
        frame_data = cam.get_latest_frame()
        if frame_data is not None:
            break
        time.sleep(0.05)

    assert frame_data is not None
    frame, ts = frame_data
    assert frame.shape[0] > 0 and frame.shape[1] > 0
    assert ts > 0

    # Test dynamic resolution and fps adjustment
    res_ok = cam.change_resolution(320, 240)
    assert res_ok is True

    fps_ok = cam.change_fps(30)
    assert fps_ok is True

    # Clean shutdown
    cam.stop()
    assert cam.is_connected is False
    assert cam._cap is None
