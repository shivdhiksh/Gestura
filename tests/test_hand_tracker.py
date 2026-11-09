"""
Unit tests for MediaPipe Hand Tracker and Landmark Extraction.
"""

import numpy as np
import pytest
from core.hand_tracker import HandTracker, HandTrackingResult
from config import AppConfig


def test_hand_tracker_init():
    config = AppConfig()
    tracker = HandTracker(config.model_path)
    assert tracker is not None
    tracker.close()


def test_hand_tracker_empty_frame():
    config = AppConfig()
    tracker = HandTracker(config.model_path)

    # Blank black frame
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = tracker.process_frame(black_frame)

    assert isinstance(result, HandTrackingResult)
    assert result.hand_detected is False
    assert result.index_tip_norm is None
    assert result.index_tip_px is None
    assert len(result.landmarks_norm) == 0
    assert result.inference_time_ms > 0

    tracker.close()


def test_hand_tracker_draw_landmarks_on_empty():
    config = AppConfig()
    tracker = HandTracker(config.model_path)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = HandTrackingResult(hand_detected=False)

    annotated = tracker.draw_landmarks(frame, result, is_inside_active_zone=False)
    assert annotated.shape == frame.shape
    tracker.close()

