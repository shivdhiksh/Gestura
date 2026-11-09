"""
GESTURA - Hand Tracker Module
Leverages Google MediaPipe HandLandmarker for real-time 21-point hand tracking,
extracting landmark #8 (Index Fingertip) and optional two-hand landmarks for zoom mode.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import List, Optional, Tuple
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
import numpy as np


# Standard hand landmark connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (9, 10), (10, 11), (11, 12),
    # Ring
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17)
]

INDEX_FINGER_TIP_IDX = 8
THUMB_TIP_IDX = 4


@dataclass
class HandTrackingResult:
    hand_detected: bool = False
    hands_count: int = 0
    index_tip_norm: Optional[Tuple[float, float]] = None
    index_tip_px: Optional[Tuple[int, int]] = None
    thumb_tip_norm: Optional[Tuple[float, float]] = None
    landmarks_norm: List[Tuple[float, float, float]] = field(default_factory=list)
    landmarks_px: List[Tuple[int, int]] = field(default_factory=list)
    handedness: Optional[str] = None

    # Secondary hand (for two-hand zoom mode)
    second_landmarks_norm: List[Tuple[float, float, float]] = field(default_factory=list)
    second_landmarks_px: List[Tuple[int, int]] = field(default_factory=list)
    second_index_tip_norm: Optional[Tuple[float, float]] = None

    inference_time_ms: float = 0.0


class HandTracker:
    """
    High-performance MediaPipe HandLandmarker wrapper optimized for low latency.
    Runs in VIDEO mode with strictly monotonic millisecond timestamps.
    """

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

    def __init__(self, model_path: str, max_num_hands: int = 1):
        self.model_path = self._ensure_model(model_path)
        self.max_num_hands = max_num_hands
        self._landmarker = self._init_landmarker(max_num_hands)
        self._last_timestamp_ms: int = -1

    def _ensure_model(self, model_path: str) -> str:
        target = Path(model_path)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"[HandTracker] Downloading model asset from {self.MODEL_URL}...")
            urllib.request.urlretrieve(self.MODEL_URL, str(target))
            print(f"[HandTracker] Download completed ({target.stat().st_size} bytes).")
        return str(target.resolve())

    def _init_landmarker(self, num_hands: int) -> vision.HandLandmarker:
        base_options = BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )
        return vision.HandLandmarker.create_from_options(options)

    def set_max_hands(self, num_hands: int) -> None:
        """Dynamically reconfigures 1-hand vs 2-hand tracking."""
        if num_hands != self.max_num_hands:
            self.max_num_hands = num_hands
            if hasattr(self, "_landmarker") and self._landmarker:
                self._landmarker.close()
            self._landmarker = self._init_landmarker(num_hands)

    def process_frame(self, frame_bgr: np.ndarray) -> HandTrackingResult:
        """Processes a single BGR video frame and extracts 21 landmarks."""
        h, w = frame_bgr.shape[:2]
        t_start = time.perf_counter()

        curr_timestamp_ms = int(time.perf_counter() * 1000)
        if curr_timestamp_ms <= self._last_timestamp_ms:
            curr_timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = curr_timestamp_ms

        # Convert OpenCV BGR to RGB
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self._landmarker.detect_for_video(mp_image, curr_timestamp_ms)
        t_end = time.perf_counter()
        inference_time_ms = (t_end - t_start) * 1000.0

        if not detection_result.hand_landmarks or len(detection_result.hand_landmarks) == 0:
            return HandTrackingResult(
                hand_detected=False,
                hands_count=0,
                inference_time_ms=inference_time_ms,
            )

        hands_count = len(detection_result.hand_landmarks)
        primary_hand = detection_result.hand_landmarks[0]
        handedness = None
        if detection_result.handedness and len(detection_result.handedness) > 0:
            handedness = detection_result.handedness[0][0].category_name

        landmarks_norm: List[Tuple[float, float, float]] = []
        landmarks_px: List[Tuple[int, int]] = []
        for lm in primary_hand:
            landmarks_norm.append((lm.x, lm.y, lm.z))
            landmarks_px.append((int(round(lm.x * w)), int(round(lm.y * h))))

        index_tip_norm = (primary_hand[INDEX_FINGER_TIP_IDX].x, primary_hand[INDEX_FINGER_TIP_IDX].y)
        index_tip_px = landmarks_px[INDEX_FINGER_TIP_IDX]
        thumb_tip_norm = (primary_hand[THUMB_TIP_IDX].x, primary_hand[THUMB_TIP_IDX].y)

        # Secondary hand if present (zoom mode)
        second_norm: List[Tuple[float, float, float]] = []
        second_px: List[Tuple[int, int]] = []
        second_index_tip: Optional[Tuple[float, float]] = None

        if hands_count > 1:
            second_hand = detection_result.hand_landmarks[1]
            for lm in second_hand:
                second_norm.append((lm.x, lm.y, lm.z))
                second_px.append((int(round(lm.x * w)), int(round(lm.y * h))))
            second_index_tip = (second_hand[INDEX_FINGER_TIP_IDX].x, second_hand[INDEX_FINGER_TIP_IDX].y)

        return HandTrackingResult(
            hand_detected=True,
            hands_count=hands_count,
            index_tip_norm=index_tip_norm,
            index_tip_px=index_tip_px,
            thumb_tip_norm=thumb_tip_norm,
            landmarks_norm=landmarks_norm,
            landmarks_px=landmarks_px,
            handedness=handedness,
            second_landmarks_norm=second_norm,
            second_landmarks_px=second_px,
            second_index_tip_norm=second_index_tip,
            inference_time_ms=inference_time_ms,
        )

    def draw_landmarks(
        self,
        frame: np.ndarray,
        tracking_result: HandTrackingResult,
        is_inside_active_zone: bool = False,
        gesture_name: str = "",
    ) -> np.ndarray:
        """
        Renders hand skeletal lines, landmark nodes, fingertip reticle,
        and current recognized gesture badge on the frame.
        """
        if not tracking_result.hand_detected or not tracking_result.landmarks_px:
            return frame

        # Render primary hand
        self._draw_single_hand(
            frame,
            tracking_result.landmarks_px,
            is_primary=True,
            is_inside_active_zone=is_inside_active_zone,
            gesture_name=gesture_name,
        )

        # Render secondary hand if present
        if tracking_result.second_landmarks_px:
            self._draw_single_hand(
                frame,
                tracking_result.second_landmarks_px,
                is_primary=False,
                is_inside_active_zone=is_inside_active_zone,
                gesture_name="HAND 2 (ZOOM)",
            )

        return frame

    def _draw_single_hand(
        self,
        frame: np.ndarray,
        landmarks: List[Tuple[int, int]],
        is_primary: bool,
        is_inside_active_zone: bool,
        gesture_name: str,
    ) -> None:
        bone_color = (200, 180, 70) if is_primary else (180, 120, 200)

        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = landmarks[start_idx]
            pt2 = landmarks[end_idx]
            cv2.line(frame, pt1, pt2, bone_color, 2, cv2.LINE_AA)

        for idx, (px, py) in enumerate(landmarks):
            if idx == INDEX_FINGER_TIP_IDX and is_primary:
                continue
            cv2.circle(frame, (px, py), 3, (240, 240, 240), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 4, (100, 80, 20), 1, cv2.LINE_AA)

        if is_primary:
            tip_x, tip_y = landmarks[INDEX_FINGER_TIP_IDX]
            if is_inside_active_zone:
                ring_color = (80, 240, 120)     # Emerald Green
                center_color = (255, 255, 255)
            else:
                ring_color = (0, 165, 255)      # Orange / Amber
                center_color = (220, 220, 220)

            # Target reticle on index fingertip
            cv2.circle(frame, (tip_x, tip_y), 12, ring_color, 2, cv2.LINE_AA)
            cv2.circle(frame, (tip_x, tip_y), 5, ring_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (tip_x, tip_y), 2, center_color, -1, cv2.LINE_AA)

            # Display recognized gesture badge near fingertip
            if gesture_name:
                text = f"[{gesture_name}]"
                label_pos = (tip_x + 16, max(24, tip_y - 8))
                cv2.putText(
                    frame,
                    text,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    text,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    ring_color,
                    1,
                    cv2.LINE_AA,
                )

    def close(self) -> None:
        if hasattr(self, "_landmarker") and self._landmarker:
            self._landmarker.close()
