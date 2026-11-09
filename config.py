"""
GESTURA - Configuration Module
Centralized configuration management for real-time hand gesture mouse control,
hardware camera settings, adaptive smoothing, and gesture state machine parameters.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict


@dataclass
class CameraConfig:
    device_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 60          # Request 60 FPS; negotiated down to sensor hardware max (e.g. 30 FPS)
    buffer_size: int = 1
    backend: str = "AUTO"  # "AUTO" uses MSMF with automatic fallback to DirectShow


@dataclass
class ActiveZoneConfig:
    """
    Active rectangle tracking zone parameters in normalized coordinates [0.0, 1.0].
    Configured via width ratio, height ratio, and center offsets to ensure valid bounds.
    """
    width_ratio: float = 0.65       # 65% of camera frame width
    height_ratio: float = 0.70      # 70% of camera frame height
    center_x_offset: float = 0.0     # Horizontal offset from center (0.5)
    center_y_offset: float = 0.0     # Vertical offset from center (0.5)

    @property
    def x_min(self) -> float:
        center = 0.5 + self.center_x_offset
        return max(0.0, min(1.0, center - (self.width_ratio / 2.0)))

    @property
    def x_max(self) -> float:
        center = 0.5 + self.center_x_offset
        return max(0.0, min(1.0, center + (self.width_ratio / 2.0)))

    @property
    def y_min(self) -> float:
        center = 0.5 + self.center_y_offset
        return max(0.0, min(1.0, center - (self.height_ratio / 2.0)))

    @property
    def y_max(self) -> float:
        center = 0.5 + self.center_y_offset
        return max(0.0, min(1.0, center + (self.height_ratio / 2.0)))


@dataclass
class SmoothingConfig:
    """
    One Euro Filter adaptive smoothing parameters.
    - min_cutoff: Minimum cutoff frequency in Hz. Lower values reduce jitter during slow movements.
    - beta: Speed coefficient. Higher values reduce lag during fast movements.
    - d_cutoff: Cutoff frequency for the derivative filter in Hz.
    """
    min_cutoff: float = 1.2
    beta: float = 0.025
    d_cutoff: float = 1.0


def default_enabled_gestures() -> Dict[str, bool]:
    return {
        "cursor_move": True,
        "left_click": True,
        "drag_and_drop": True,
        "right_click": True,
        "scrolling": True,
        "volume_control": True,
        "brightness_control": True,
        "media_controls": True,
        "app_switch": True,
        "window_maximize": True,
        "pause_resume": True,
        "two_hand_zoom": False,  # Off by default for maximum single-hand inference speed
    }


@dataclass
class GestureConfig:
    sensitivity: float = 1.0              # 0.5x to 2.5x cursor speed multiplier
    mirror_preview: bool = True           # Mirror webcam horizontally for natural user interaction
    freeze_outside_zone: bool = True      # Freeze cursor immediately when finger exits active rectangle
    cursor_control_enabled: bool = False  # Gesture control must be OFF on startup for safety
    two_hand_zoom_enabled: bool = False   # Enable optional 2-hand detection for zoom

    # Interaction thresholds & durations
    pinch_threshold: float = 0.055        # Normalized distance between thumb and finger tips
    pinch_release_threshold: float = 0.080 # Hysteresis threshold to confirm pinch release
    hold_duration_drag: float = 0.32      # Seconds of pinch hold to initiate drag & drop
    hold_duration_pause: float = 0.70     # Seconds of closed fist to pause control
    hold_duration_resume: float = 0.70    # Seconds of open palm to resume control
    double_pinch_window: float = 0.38     # Max seconds between pinches to trigger double-click/prev-track

    # Scrolling & Adjustments
    scroll_sensitivity: float = 1.0       # Scroll speed multiplier
    scroll_deadzone: float = 0.012        # Movement delta deadzone before triggering scroll
    volume_repeat_interval: float = 0.15  # Cooldown between volume steps (seconds)
    brightness_repeat_interval: float = 0.18 # Cooldown between brightness steps (seconds)
    action_cooldown: float = 0.50         # Cooldown for discrete media/window triggers

    # Per-gesture enable/disable toggles
    enabled_gestures: Dict[str, bool] = field(default_factory=default_enabled_gestures)


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    active_zone: ActiveZoneConfig = field(default_factory=ActiveZoneConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)

    # Path to MediaPipe Hand Landmarker model asset
    model_path: str = str(Path(__file__).parent / "assets" / "models" / "hand_landmarker.task")
    emergency_pause_key: str = "F8"
