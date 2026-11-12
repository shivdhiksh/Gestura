"""
Unit tests for Gesture Classification, State Machine, Debouncing, and Action Triggers.
"""

import pytest
import time
from typing import List, Tuple
from config import GestureConfig
from core.action_controller import ActionController
from core.gesture_recognizer import GestureRecognizer, HandFeatures


class MockActionController(ActionController):
    """Mock ActionController for testing gesture triggers without moving physical desktop cursor."""
    def __init__(self):
        self.left_clicks = 0
        self.double_clicks = 0
        self.right_clicks = 0
        self.dragging = False
        self.scroll_events = []
        self.volume_ups = 0
        self.volume_downs = 0
        self.brightness_events = []
        self.media_plays = 0
        self.media_nexts = 0
        self.media_prevs = 0
        self.app_switches_next = 0
        self.app_switches_prev = 0
        self.window_maximizes = 0
        self.zoom_events = []
        self._is_active = True

    def left_click(self):
        self.left_clicks += 1

    def double_click(self):
        self.double_clicks += 1

    def right_click(self):
        self.right_clicks += 1

    def start_drag(self):
        self.dragging = True

    def stop_drag(self):
        self.dragging = False

    def scroll(self, dx, dy):
        self.scroll_events.append((dx, dy))

    def volume_up(self):
        self.volume_ups += 1

    def volume_down(self):
        self.volume_downs += 1

    def change_brightness(self, delta):
        self.brightness_events.append(delta)
        return 50

    def media_play_pause(self):
        self.media_plays += 1

    def media_next_track(self):
        self.media_nexts += 1

    def media_prev_track(self):
        self.media_prevs += 1

    def switch_app_next(self):
        self.app_switches_next += 1

    def switch_app_prev(self):
        self.app_switches_prev += 1

    def toggle_window_maximize(self):
        self.window_maximizes += 1

    def zoom(self, delta):
        self.zoom_events.append(delta)

    def release_all(self):
        self.dragging = False


def create_synthetic_hand(
    thumb_pos=(0.35, 0.45),
    index_pos=(0.45, 0.25),
    middle_pos=(0.53, 0.22),
    ring_pos=(0.61, 0.25),
    pinky_pos=(0.68, 0.30),
    wrist=(0.50, 0.70),
    fingers_extended=(True, True, True, True, True), # (thumb, index, middle, ring, pinky)
) -> List[Tuple[float, float, float]]:
    """Creates a 21-landmark array reflecting specified finger extension states."""
    lms = [(0.0, 0.0, 0.0)] * 21
    lms[0] = (wrist[0], wrist[1], 0.0) # Wrist
    lms[9] = (wrist[0], wrist[1] - 0.20, 0.0) # Middle MCP
    lms[17] = (wrist[0] + 0.10, wrist[1] - 0.18, 0.0) # Pinky MCP
    lms[5] = (wrist[0] - 0.08, wrist[1] - 0.18, 0.0) # Index MCP
    lms[2] = (wrist[0] - 0.08, wrist[1] - 0.08, 0.0) # Thumb MCP

    # Thumb
    if fingers_extended[0]:
        lms[4] = (thumb_pos[0], thumb_pos[1], 0.0)
    else:
        # Folded across palm
        lms[4] = (wrist[0] - 0.02, wrist[1] - 0.15, 0.0)

    # Index
    if fingers_extended[1]:
        lms[6] = (index_pos[0], index_pos[1] + 0.12, 0.0) # PIP
        lms[8] = (index_pos[0], index_pos[1], 0.0) # TIP
    else:
        lms[6] = (index_pos[0], wrist[1] - 0.10, 0.0)
        lms[8] = (index_pos[0], wrist[1] - 0.05, 0.0)

    # Middle
    if fingers_extended[2]:
        lms[10] = (middle_pos[0], middle_pos[1] + 0.12, 0.0)
        lms[12] = (middle_pos[0], middle_pos[1], 0.0)
    else:
        lms[10] = (middle_pos[0], wrist[1] - 0.10, 0.0)
        lms[12] = (middle_pos[0], wrist[1] - 0.05, 0.0)

    # Ring
    if fingers_extended[3]:
        lms[14] = (ring_pos[0], ring_pos[1] + 0.12, 0.0)
        lms[16] = (ring_pos[0], ring_pos[1], 0.0)
    else:
        lms[14] = (ring_pos[0], wrist[1] - 0.10, 0.0)
        lms[16] = (ring_pos[0], wrist[1] - 0.05, 0.0)

    # Pinky
    if fingers_extended[4]:
        lms[18] = (pinky_pos[0], pinky_pos[1] + 0.10, 0.0)
        lms[20] = (pinky_pos[0], pinky_pos[1], 0.0)
    else:
        lms[18] = (pinky_pos[0], wrist[1] - 0.10, 0.0)
        lms[20] = (pinky_pos[0], wrist[1] - 0.05, 0.0)

    return lms


def test_feature_extraction():
    cfg = GestureConfig()
    actions = MockActionController()
    gr = GestureRecognizer(cfg, actions)

    # Synthetic open palm
    lms = create_synthetic_hand(fingers_extended=(True, True, True, True, True))
    features = gr.extract_features(lms)
    assert features is not None
    assert features.is_open_palm is True
    assert features.is_fist is False


def test_fist_pause_and_open_palm_resume():
    cfg = GestureConfig(hold_duration_pause=0.30, hold_duration_resume=0.30)
    actions = MockActionController()
    gr = GestureRecognizer(cfg, actions)

    # Fist landmarks
    fist_lms = create_synthetic_hand(fingers_extended=(False, False, False, False, False))
    features = gr.extract_features(fist_lms)
    assert features.is_fist is True

    # Hold fist for 0.35s
    gr.update_gestures(features, timestamp=1.0)
    assert gr.is_paused is False
    gr.update_gestures(features, timestamp=1.35)
    assert gr.is_paused is True

    # Open palm landmarks
    palm_lms = create_synthetic_hand(fingers_extended=(True, True, True, True, True))
    features_palm = gr.extract_features(palm_lms)
    assert features_palm.is_open_palm is True

    # Hold open palm for 0.35s to resume
    gr.update_gestures(features_palm, timestamp=2.0)
    assert gr.is_paused is True
    gr.update_gestures(features_palm, timestamp=2.35)
    assert gr.is_paused is False


def test_pinch_left_click_and_double_click():
    cfg = GestureConfig(hold_duration_drag=0.30, double_pinch_window=0.35)
    actions = MockActionController()
    gr = GestureRecognizer(cfg, actions)

    # 1. Normal index tracking
    idx_lms = create_synthetic_hand(fingers_extended=(False, True, False, False, False))
    f_idx = gr.extract_features(idx_lms)
    g1 = gr.update_gestures(f_idx, timestamp=1.0)
    assert g1 == "CURSOR TRACKING"

    # 2. Pinch thumb and index together (index at 0.45, middle separated at 0.55)
    pinch_lms = create_synthetic_hand(thumb_pos=(0.45, 0.25), index_pos=(0.45, 0.25), middle_pos=(0.55, 0.22))
    f_pinch = gr.extract_features(pinch_lms)
    assert f_pinch.thumb_index_dist < 0.055

    gr.update_gestures(f_pinch, timestamp=1.05)
    # 3. Release pinch quickly (after 0.1s < hold_duration_drag)
    gr.update_gestures(f_idx, timestamp=1.15)
    assert actions.left_clicks == 1
    assert actions.double_clicks == 0

    # 4. Second quick pinch within double_pinch_window (e.g. at 1.30s)
    gr.update_gestures(f_pinch, timestamp=1.25)
    gr.update_gestures(f_idx, timestamp=1.32)
    assert actions.double_clicks == 1


def test_pinch_drag_and_drop():
    cfg = GestureConfig(hold_duration_drag=0.30)
    actions = MockActionController()
    gr = GestureRecognizer(cfg, actions)

    pinch_lms = create_synthetic_hand(thumb_pos=(0.45, 0.25), index_pos=(0.45, 0.25), middle_pos=(0.55, 0.22))
    f_pinch = gr.extract_features(pinch_lms)

    # Start pinch at t=1.0
    gr.update_gestures(f_pinch, timestamp=1.0)
    assert actions.dragging is False

    # Continue holding pinch at t=1.35 (> hold_duration_drag of 0.30)
    gr.update_gestures(f_pinch, timestamp=1.35)
    assert actions.dragging is True

    # Release pinch -> Drop
    idx_lms = create_synthetic_hand(fingers_extended=(False, True, False, False, False))
    f_idx = gr.extract_features(idx_lms)
    gr.update_gestures(f_idx, timestamp=1.40)
    assert actions.dragging is False


def test_right_click_middle_pinch():
    cfg = GestureConfig()
    actions = MockActionController()
    gr = GestureRecognizer(cfg, actions)

    # Pinch thumb and middle
    lms = create_synthetic_hand(thumb_pos=(0.53, 0.22), middle_pos=(0.53, 0.22), index_pos=(0.43, 0.25))
    f = gr.extract_features(lms)
    assert f.thumb_middle_dist < 0.055

    gr.update_gestures(f, timestamp=1.0)
    assert actions.right_clicks == 1


def test_two_finger_scrolling():
    cfg = GestureConfig(scroll_sensitivity=1.0, scroll_deadzone=0.01)
    actions = MockActionController()
    gr = GestureRecognizer(cfg, actions)

    # 1. Two fingers extended close together (index + middle) at y=0.40
    lms1 = create_synthetic_hand(
        index_pos=(0.50, 0.40), middle_pos=(0.52, 0.40),
        fingers_extended=(False, True, True, False, False)
    )
    f1 = gr.extract_features(lms1)
    assert f1.index_middle_together is True
    g1 = gr.update_gestures(f1, timestamp=1.0)
    assert "SCROLL" in g1

    # 2. Move fingers up to y=0.35 (delta = 0.05 > deadzone 0.01)
    lms2 = create_synthetic_hand(
        index_pos=(0.50, 0.35), middle_pos=(0.52, 0.35),
        fingers_extended=(False, True, True, False, False)
    )
    f2 = gr.extract_features(lms2)
    g2 = gr.update_gestures(f2, timestamp=1.033)
    assert g2 == "SCROLL: UP (FINGERS UP)"
    assert len(actions.scroll_events) == 1
    assert actions.scroll_events[-1][1] > 0

    # 3. KEEP fingers up at y=0.35 -> continuous scrolling up!
    g3 = gr.update_gestures(f2, timestamp=1.066)
    assert g3 == "SCROLL: UP (FINGERS UP)"
    assert len(actions.scroll_events) == 2
    assert actions.scroll_events[-1][1] > 0

    # 4. Return fingers to neutral resting position (y=0.40) -> stops scrolling
    g_neutral = gr.update_gestures(f1, timestamp=1.100)
    assert g_neutral == "TWO-FINGER SCROLL (NEUTRAL)"
    assert len(actions.scroll_events) == 2  # No new scroll dispatched

    # 5. Move/KEEP fingers DOWN (y=0.46 -> dy = -0.06 < -0.01) -> scrolls down!
    lms_down = create_synthetic_hand(
        index_pos=(0.50, 0.46), middle_pos=(0.52, 0.46),
        fingers_extended=(False, True, True, False, False)
    )
    f_down = gr.extract_features(lms_down)
    g_down = gr.update_gestures(f_down, timestamp=1.133)
    assert g_down == "SCROLL: DOWN (FINGERS DOWN)"
    assert len(actions.scroll_events) == 3
    assert actions.scroll_events[-1][1] < 0

    # 6. Fingers spread wide apart (peace/V sign) -> NOT recognized as two fingers together
    lms_spread = create_synthetic_hand(
        index_pos=(0.40, 0.35), middle_pos=(0.60, 0.35),
        fingers_extended=(False, True, True, False, False)
    )
    f_spread = gr.extract_features(lms_spread)
    assert f_spread.index_middle_together is False
    g_spread = gr.update_gestures(f_spread, timestamp=1.166)
    assert "SCROLL" not in g_spread


def test_volume_controls():
    cfg = GestureConfig(volume_repeat_interval=0.10)
    actions = MockActionController()
    gr = GestureRecognizer(cfg, actions)

    # Thumb Up
    lms_up = create_synthetic_hand(
        thumb_pos=(0.50, 0.40), wrist=(0.50, 0.70),
        fingers_extended=(True, False, False, False, False)
    )
    f_up = gr.extract_features(lms_up)
    assert f_up.thumb_up is True

    gr.update_gestures(f_up, timestamp=1.0)
    assert actions.volume_ups == 1

    # Immediate repeat blocked by interval
    gr.update_gestures(f_up, timestamp=1.05)
    assert actions.volume_ups == 1

    # After interval -> second step
    gr.update_gestures(f_up, timestamp=1.12)
    assert actions.volume_ups == 2


def test_four_fingers_maximize_window():
    cfg = GestureConfig()
    actions = MockActionController()
    gr = GestureRecognizer(cfg, actions)

    # 4 fingers up, thumb folded
    lms = create_synthetic_hand(fingers_extended=(False, True, True, True, True))
    f = gr.extract_features(lms)
    assert f.is_four_fingers_up is True

    gr.update_gestures(f, timestamp=1.0)
    assert actions.window_maximizes == 0

    # Held for 0.42s
    gr.update_gestures(f, timestamp=1.42)
    assert actions.window_maximizes == 1
