"""
GESTURA - Gesture Recognizer & State Machine
Extracts hand geometric features from MediaPipe 21 landmarks, classifies gestures,
and runs a state machine with debouncing, hysteresis, hold durations, and cooldowns.
"""

from collections import deque
from dataclasses import dataclass, field
import math
import time
from typing import Deque, Dict, List, Optional, Tuple

from config import GestureConfig
from core.action_controller import ActionController


def euclidean_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


@dataclass
class HandFeatures:
    """Extracted geometric features for a single hand."""
    palm_size: float = 0.2
    wrist: Tuple[float, float] = (0.5, 0.5)
    index_tip: Tuple[float, float] = (0.5, 0.5)
    thumb_tip: Tuple[float, float] = (0.5, 0.5)
    middle_tip: Tuple[float, float] = (0.5, 0.5)
    ring_tip: Tuple[float, float] = (0.5, 0.5)
    pinky_tip: Tuple[float, float] = (0.5, 0.5)

    thumb_extended: bool = False
    thumb_folded: bool = False
    index_extended: bool = False
    middle_extended: bool = False
    ring_extended: bool = False
    pinky_extended: bool = False

    thumb_index_dist: float = 1.0
    thumb_middle_dist: float = 1.0
    thumb_ring_dist: float = 1.0
    thumb_pinky_dist: float = 1.0

    thumb_up: bool = False
    thumb_down: bool = False
    is_fist: bool = False
    is_open_palm: bool = False
    is_four_fingers_up: bool = False


class GestureRecognizer:
    """
    Classifies 16 distinct hand gestures and executes corresponding system actions
    via ActionController with state debouncing, hysteresis, and cooldowns.
    """

    def __init__(self, config: GestureConfig, action_controller: ActionController):
        self.config = config
        self.actions = action_controller

        # Gesture state machine timers and trackers
        self.current_gesture_name: str = "IDLE"
        self._last_gesture_name: str = "IDLE"

        # Pinch states
        self._thumb_index_pinched: bool = False
        self._pinch_start_time: float = 0.0
        self._is_dragging: bool = False
        self._last_left_click_time: float = 0.0

        # Thumb + Pinky double-pinch tracker
        self._thumb_pinky_pinched: bool = False
        self._last_pinky_pinch_release_time: float = 0.0
        self._pinky_pinch_count: int = 0

        # Hold duration timers
        self._fist_start_time: float = 0.0
        self._open_palm_start_time: float = 0.0
        self._four_fingers_start_time: float = 0.0

        # Cooldown timestamps
        self._last_action_times: Dict[str, float] = {}

        # Tracking history for swipe and scroll deltas
        self._pos_history: Deque[Tuple[float, float, float]] = deque(maxlen=10)
        self._scroll_anchor: Optional[Tuple[float, float]] = None
        self._brightness_anchor_y: Optional[float] = None
        self._last_two_hand_dist: Optional[float] = None

        # Pause status
        self.is_paused: bool = False

    def extract_features(self, landmarks_norm: List[Tuple[float, float, float]]) -> Optional[HandFeatures]:
        """Calculates scale-invariant geometric features from 21 MediaPipe landmarks."""
        if not landmarks_norm or len(landmarks_norm) < 21:
            return None

        # Key landmark 2D coordinates (x, y)
        w_pt = (landmarks_norm[0][0], landmarks_norm[0][1])      # Wrist (0)
        m_mcp = (landmarks_norm[9][0], landmarks_norm[9][1])     # Middle MCP (9)
        p_mcp = (landmarks_norm[17][0], landmarks_norm[17][1])   # Pinky MCP (17)
        i_mcp = (landmarks_norm[5][0], landmarks_norm[5][1])     # Index MCP (5)
        t_mcp = (landmarks_norm[2][0], landmarks_norm[2][1])     # Thumb MCP (2)
        t_tip = (landmarks_norm[4][0], landmarks_norm[4][1])     # Thumb Tip (4)

        i_pip = (landmarks_norm[6][0], landmarks_norm[6][1])     # Index PIP (6)
        i_tip = (landmarks_norm[8][0], landmarks_norm[8][1])     # Index Tip (8)

        m_pip = (landmarks_norm[10][0], landmarks_norm[10][1])   # Middle PIP (10)
        m_tip = (landmarks_norm[12][0], landmarks_norm[12][1])   # Middle Tip (12)

        r_pip = (landmarks_norm[14][0], landmarks_norm[14][1])   # Ring PIP (14)
        r_tip = (landmarks_norm[16][0], landmarks_norm[16][1])   # Ring Tip (16)

        p_pip = (landmarks_norm[18][0], landmarks_norm[18][1])   # Pinky PIP (18)
        p_tip = (landmarks_norm[20][0], landmarks_norm[20][1])   # Pinky Tip (20)

        # Scale reference: Palm Length
        palm_size = max(0.05, euclidean_dist(w_pt, m_mcp))

        # Finger extension tests (distance from wrist compared to PIP joint)
        index_ext = euclidean_dist(i_tip, w_pt) > (euclidean_dist(i_pip, w_pt) + 0.10 * palm_size)
        middle_ext = euclidean_dist(m_tip, w_pt) > (euclidean_dist(m_pip, w_pt) + 0.10 * palm_size)
        ring_ext = euclidean_dist(r_tip, w_pt) > (euclidean_dist(r_pip, w_pt) + 0.10 * palm_size)
        pinky_ext = euclidean_dist(p_tip, w_pt) > (euclidean_dist(p_pip, w_pt) + 0.10 * palm_size)

        # Pinch distances
        d_ti = euclidean_dist(t_tip, i_tip)
        d_tm = euclidean_dist(t_tip, m_tip)
        d_tr = euclidean_dist(t_tip, r_tip)
        d_tp = euclidean_dist(t_tip, p_tip)

        # Thumb extension / fold
        dist_thumb_mcp = euclidean_dist(t_tip, t_mcp)
        dist_thumb_index_mcp = euclidean_dist(t_tip, i_mcp)
        thumb_ext = dist_thumb_mcp > 0.40 * palm_size and dist_thumb_index_mcp > 0.35 * palm_size
        thumb_folded = dist_thumb_index_mcp <= 0.45 * palm_size or dist_thumb_mcp <= 0.40 * palm_size

        pinch_thr = self.config.pinch_threshold * self._hand_scale(palm_size)
        has_pinch = d_ti < pinch_thr or d_tm < pinch_thr or d_tr < pinch_thr or d_tp < pinch_thr

        # Fist: all 4 fingers curled down + thumb curled
        is_fist = (not index_ext) and (not middle_ext) and (not ring_ext) and (not pinky_ext) and thumb_folded and not has_pinch

        # Open palm: all 5 fingers extended and no active pinch
        is_open_palm = index_ext and middle_ext and ring_ext and pinky_ext and thumb_ext and not has_pinch

        # Four fingers up: 4 fingers straight, thumb folded, no active pinch
        is_four_fingers = index_ext and middle_ext and ring_ext and pinky_ext and thumb_folded and not has_pinch

        # Thumb up / down: 4 fingers curled into fist, thumb straight
        thumb_up = False
        thumb_down = False
        if (not index_ext) and (not middle_ext) and (not ring_ext) and (not pinky_ext) and thumb_ext and not has_pinch:
            if t_tip[1] < (t_mcp[1] - 0.15 * palm_size):
                thumb_up = True
            elif t_tip[1] > (t_mcp[1] + 0.10 * palm_size):
                thumb_down = True

        return HandFeatures(
            palm_size=palm_size,
            wrist=w_pt,
            index_tip=i_tip,
            thumb_tip=t_tip,
            middle_tip=m_tip,
            ring_tip=r_tip,
            pinky_tip=p_tip,
            thumb_extended=thumb_ext,
            thumb_folded=thumb_folded,
            index_extended=index_ext,
            middle_extended=middle_ext,
            ring_extended=ring_ext,
            pinky_extended=pinky_ext,
            thumb_index_dist=d_ti,
            thumb_middle_dist=d_tm,
            thumb_ring_dist=d_tr,
            thumb_pinky_dist=d_tp,
            thumb_up=thumb_up,
            thumb_down=thumb_down,
            is_fist=is_fist,
            is_open_palm=is_open_palm,
            is_four_fingers_up=is_four_fingers,
        )

    # Palm length (wrist -> middle MCP) at which the configured pinch thresholds were tuned.
    REFERENCE_PALM_SIZE = 0.18

    def _hand_scale(self, palm_size: float) -> float:
        """Scale factor so pinch thresholds work whether the hand is near or far from the camera."""
        return max(0.5, min(2.0, palm_size / self.REFERENCE_PALM_SIZE))

    def _is_gesture_enabled(self, gesture_key: str) -> bool:
        return self.config.enabled_gestures.get(gesture_key, True)

    def _cooldown_ready(self, action_key: str, cooldown_sec: float, timestamp: Optional[float] = None) -> bool:
        now = timestamp if timestamp is not None else time.perf_counter()
        last = self._last_action_times.get(action_key, 0.0)
        if (now - last) >= cooldown_sec:
            self._last_action_times[action_key] = now
            return True
        return False

    def update_gestures(
        self,
        features: Optional[HandFeatures],
        second_features: Optional[HandFeatures] = None,
        is_inside_active_zone: bool = True,
        timestamp: Optional[float] = None,
    ) -> str:
        """
        Main gesture state machine evaluation loop.
        Classifies current hand posture, handles state transitions,
        and triggers Windows actions. Returns current recognized gesture name.
        """
        if timestamp is None:
            timestamp = time.perf_counter()

        scale = self._hand_scale(features.palm_size) if features is not None else 1.0
        pinch_thr = self.config.pinch_threshold * scale
        pinch_release_thr = self.config.pinch_release_threshold * scale

        # Handle pending double-pinch timeout for Pinky pinch (Next vs Previous Track)
        if self._pinky_pinch_count == 1:
            if (timestamp - self._last_pinky_pinch_release_time) > self.config.double_pinch_window:
                if self._is_gesture_enabled("media_controls") and not self.is_paused:
                    self.actions.media_next_track()
                self._pinky_pinch_count = 0

        # When no hand is detected
        if features is None:
            if self._is_dragging:
                self.actions.stop_drag()
                self._is_dragging = False
            self._thumb_index_pinched = False
            self._scroll_anchor = None
            self._brightness_anchor_y = None
            self._last_two_hand_dist = None
            self._pos_history.clear()
            self.current_gesture_name = "NO HAND"
            return self.current_gesture_name

        # Update position history for swipe velocity
        self._pos_history.append((features.wrist[0], features.wrist[1], timestamp))

        # =====================================================================
        # 1. TWO-HAND ZOOM MODE (Optional)
        # =====================================================================
        if (
            self.config.two_hand_zoom_enabled
            and self._is_gesture_enabled("two_hand_zoom")
            and second_features is not None
        ):
            p1 = features.thumb_index_dist < pinch_thr
            p2 = second_features.thumb_index_dist < pinch_thr
            if p1 and p2:
                curr_dist = euclidean_dist(features.index_tip, second_features.index_tip)
                if self._last_two_hand_dist is not None:
                    delta_d = curr_dist - self._last_two_hand_dist
                    if abs(delta_d) > 0.015:
                        self.actions.zoom(delta_d)
                self._last_two_hand_dist = curr_dist
                self.current_gesture_name = "TWO-HAND ZOOM"
                return self.current_gesture_name
            else:
                self._last_two_hand_dist = None

        # =====================================================================
        # 2. CLOSED FIST (PAUSE) & OPEN PALM (RESUME)
        # =====================================================================
        if self._is_gesture_enabled("pause_resume"):
            # A. Closed Fist held -> Pause
            if features.is_fist:
                if self._fist_start_time == 0.0:
                    self._fist_start_time = timestamp
                elif (timestamp - self._fist_start_time) >= self.config.hold_duration_pause:
                    if not self.is_paused:
                        self.is_paused = True
                        self.actions.release_all()
                    self.current_gesture_name = "PAUSED (CLOSED FIST)"
                    return self.current_gesture_name
            else:
                self._fist_start_time = 0.0

            # B. Open Palm held -> Resume (if paused)
            if features.is_open_palm:
                if self._open_palm_start_time == 0.0:
                    self._open_palm_start_time = timestamp
                elif (timestamp - self._open_palm_start_time) >= self.config.hold_duration_resume:
                    if self.is_paused:
                        self.is_paused = False
                    self.current_gesture_name = "RESUMED (OPEN PALM)"
                    return self.current_gesture_name
            else:
                self._open_palm_start_time = 0.0

        if self.is_paused:
            self.current_gesture_name = "PAUSED"
            return self.current_gesture_name

        # If finger is outside active rectangle, freeze actions
        if not is_inside_active_zone:
            if self._is_dragging:
                self.actions.stop_drag()
                self._is_dragging = False
            self.current_gesture_name = "OUTSIDE ACTIVE ZONE"
            return self.current_gesture_name

        # =====================================================================
        # 3. PINCH GESTURES (HIGHEST PRIORITY FOR ACTIONS)
        # =====================================================================
        # Disambiguate pinches by determining which finger is closest to thumb
        d_ti = features.thumb_index_dist
        d_tm = features.thumb_middle_dist
        d_tr = features.thumb_ring_dist
        d_tp = features.thumb_pinky_dist

        # A. Thumb + Middle Pinch (Right Click)
        if (
            self._is_gesture_enabled("right_click")
            and d_tm < pinch_thr
            and d_tm < d_ti
        ):
            if self._cooldown_ready("right_click", 0.45, timestamp):
                self.actions.right_click()
            self.current_gesture_name = "RIGHT CLICK"
            return self.current_gesture_name

        # B. Thumb + Ring Pinch (Play / Pause Media)
        if (
            self._is_gesture_enabled("media_controls")
            and d_tr < pinch_thr
            and d_tr < d_tm
            and d_tr < d_tp
        ):
            if self._cooldown_ready("media_play_pause", self.config.action_cooldown, timestamp):
                self.actions.media_play_pause()
            self.current_gesture_name = "PLAY / PAUSE MEDIA"
            return self.current_gesture_name

        # C. Thumb + Little Finger Pinch (Next / Previous Track)
        if self._is_gesture_enabled("media_controls"):
            is_pinky_pinched = (
                d_tp < pinch_thr
                and d_tp < d_tr
            )
            if is_pinky_pinched and not self._thumb_pinky_pinched:
                self._thumb_pinky_pinched = True
            elif not is_pinky_pinched and self._thumb_pinky_pinched:
                self._thumb_pinky_pinched = False
                if (timestamp - self._last_pinky_pinch_release_time) <= self.config.double_pinch_window:
                    self.actions.media_prev_track()
                    self._pinky_pinch_count = 0
                    self.current_gesture_name = "PREVIOUS TRACK"
                    return self.current_gesture_name
                else:
                    self._pinky_pinch_count = 1
                    self._last_pinky_pinch_release_time = timestamp

            if self._thumb_pinky_pinched:
                self.current_gesture_name = "MEDIA TRACK PINCH"
                return self.current_gesture_name

        # D. Thumb + Index Pinch (Left Click & Drag / Drop)
        is_ti_pinched = False
        if self._thumb_index_pinched:
            is_ti_pinched = d_ti < pinch_release_thr
        else:
            is_ti_pinched = d_ti < pinch_thr and d_ti <= d_tm

        if is_ti_pinched:
            if not self._thumb_index_pinched:
                self._thumb_index_pinched = True
                self._pinch_start_time = timestamp

            pinch_duration = timestamp - self._pinch_start_time

            if self._is_gesture_enabled("drag_and_drop") and pinch_duration >= self.config.hold_duration_drag:
                if not self._is_dragging:
                    self.actions.start_drag()
                    self._is_dragging = True
                self.current_gesture_name = "DRAGGING (HOLD PINCH)"
                return self.current_gesture_name
            else:
                self.current_gesture_name = "PINCH (LEFT CLICK PENDING)"
                return self.current_gesture_name
        else:
            if self._thumb_index_pinched:
                pinch_duration = timestamp - self._pinch_start_time
                self._thumb_index_pinched = False

                if self._is_dragging:
                    self.actions.stop_drag()
                    self._is_dragging = False
                    self.current_gesture_name = "DROPPED"
                    return self.current_gesture_name
                elif self._is_gesture_enabled("left_click"):
                    if (timestamp - self._last_left_click_time) <= self.config.double_pinch_window:
                        self.actions.double_click()
                        self._last_left_click_time = 0.0
                        self.current_gesture_name = "DOUBLE CLICK"
                        return self.current_gesture_name
                    else:
                        self.actions.left_click()
                        self._last_left_click_time = timestamp
                        self.current_gesture_name = "LEFT CLICK"
                        return self.current_gesture_name

        # =====================================================================
        # 4. TWO FINGERS (VERTICAL & HORIZONTAL SCROLLING)
        # =====================================================================
        if (
            self._is_gesture_enabled("scrolling")
            and features.index_extended
            and features.middle_extended
            and not features.ring_extended
            and not features.pinky_extended
            and features.thumb_index_dist > pinch_release_thr
        ):
            center_x = (features.index_tip[0] + features.middle_tip[0]) / 2.0
            center_y = (features.index_tip[1] + features.middle_tip[1]) / 2.0

            if self._scroll_anchor is not None:
                dx = center_x - self._scroll_anchor[0]
                dy = self._scroll_anchor[1] - center_y  # Moving up scrolls up (positive)

                deadzone = self.config.scroll_deadzone
                send_dx = dx if abs(dx) > deadzone else 0.0
                send_dy = dy if abs(dy) > deadzone else 0.0

                if send_dx != 0.0 or send_dy != 0.0:
                    multiplier = self.config.scroll_sensitivity * 18.0
                    self.actions.scroll(send_dx * multiplier, send_dy * multiplier)
                    self._scroll_anchor = (center_x, center_y)
            else:
                self._scroll_anchor = (center_x, center_y)

            self.current_gesture_name = "TWO-FINGER SCROLL"
            return self.current_gesture_name
        else:
            self._scroll_anchor = None

        # =====================================================================
        # 5. THREE FINGERS (BRIGHTNESS ADJUSTMENT)
        # =====================================================================
        if (
            self._is_gesture_enabled("brightness_control")
            and features.index_extended
            and features.middle_extended
            and features.ring_extended
            and not features.pinky_extended
        ):
            curr_y = (features.index_tip[1] + features.middle_tip[1] + features.ring_tip[1]) / 3.0
            if self._brightness_anchor_y is not None:
                dy = self._brightness_anchor_y - curr_y
                if abs(dy) > 0.035 and self._cooldown_ready("brightness", self.config.brightness_repeat_interval, timestamp):
                    step = 5 if dy > 0 else -5
                    self.actions.change_brightness(step)
                    self._brightness_anchor_y = curr_y
            else:
                self._brightness_anchor_y = curr_y
            self.current_gesture_name = "BRIGHTNESS ADJUST"
            return self.current_gesture_name
        else:
            self._brightness_anchor_y = None

        # =====================================================================
        # 6. THUMB UP / THUMB DOWN (VOLUME CONTROL)
        # =====================================================================
        if self._is_gesture_enabled("volume_control"):
            if features.thumb_up:
                if self._cooldown_ready("vol_up", self.config.volume_repeat_interval, timestamp):
                    self.actions.volume_up()
                self.current_gesture_name = "VOLUME UP"
                return self.current_gesture_name

            if features.thumb_down:
                if self._cooldown_ready("vol_down", self.config.volume_repeat_interval, timestamp):
                    self.actions.volume_down()
                self.current_gesture_name = "VOLUME DOWN"
                return self.current_gesture_name

        # =====================================================================
        # 7. OPEN-PALM HORIZONTAL SWIPE (APPLICATION SWITCHING)
        # =====================================================================
        if self._is_gesture_enabled("app_switch") and features.is_open_palm and len(self._pos_history) >= 4:
            dt = self._pos_history[-1][2] - self._pos_history[0][2]
            if 0.05 < dt < 0.4:
                dx = self._pos_history[-1][0] - self._pos_history[0][0]
                vx = dx / dt
                if vx > 0.40 and self._cooldown_ready("swipe_right", 0.60, timestamp):
                    self.actions.switch_app_next()
                    self._pos_history.clear()
                    self.current_gesture_name = "SWIPE RIGHT (APP SWITCH)"
                    return self.current_gesture_name
                elif vx < -0.40 and self._cooldown_ready("swipe_left", 0.60, timestamp):
                    self.actions.switch_app_prev()
                    self._pos_history.clear()
                    self.current_gesture_name = "SWIPE LEFT (PREV APP)"
                    return self.current_gesture_name

        # =====================================================================
        # 8. FOUR FINGERS HELD UP, THUMB FOLDED (MAXIMIZE / RESTORE)
        # =====================================================================
        if self._is_gesture_enabled("window_maximize") and features.is_four_fingers_up:
            if self._four_fingers_start_time == 0.0:
                self._four_fingers_start_time = timestamp
            elif (timestamp - self._four_fingers_start_time) >= 0.40:
                if self._cooldown_ready("window_maximize", 1.0, timestamp):
                    self.actions.toggle_window_maximize()
                self.current_gesture_name = "WINDOW MAXIMIZE/RESTORE"
                return self.current_gesture_name
        else:
            self._four_fingers_start_time = 0.0

        # =====================================================================
        # 9. INDEX FINGER ONLY (DEFAULT CURSOR MOVEMENT)
        # =====================================================================
        if self._is_gesture_enabled("cursor_move") and features.index_extended:
            self.current_gesture_name = "CURSOR TRACKING"
            return self.current_gesture_name

        self.current_gesture_name = "HAND DETECTED"
        return self.current_gesture_name
