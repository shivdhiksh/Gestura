"""
GESTURA - High-Performance Camera Manager
Provides threaded, non-blocking webcam capture with latest-frame-only strategy,
backend auto-selection (MSMF / DirectShow), and resolution/FPS tuning.
"""

import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from config import CameraConfig


class CameraManager:
    """
    Manages dedicated background camera capture thread.
    Always provides the newest available frame, dropping any stale frames
    to prevent frame queue latency buildup.
    """

    def __init__(self, config: CameraConfig, mirror: bool = True):
        self.config = config
        self.mirror = mirror

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._lifecycle_lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._frame_count: int = 0
        self._fps_measured: float = 0.0
        self._last_fps_time: float = time.perf_counter()
        self._last_fps_frame_count: int = 0

        self.actual_width: int = config.frame_width
        self.actual_height: int = config.frame_height
        self.actual_fps: float = config.fps
        self.active_backend: str = "Unknown"
        self.is_connected: bool = False

    def _open_capture_device(self) -> Optional[cv2.VideoCapture]:
        """
        Attempts to open camera using the optimal Windows backend.
        Prefers MSMF (Microsoft Media Foundation) which delivers 30-60 FPS,
        falling back to DirectShow or default backend.
        """
        preferred = self.config.backend.upper()
        backends_to_try = []

        if preferred == "MSMF":
            backends_to_try = [("MSMF", cv2.CAP_MSMF)]
        elif preferred == "DSHOW":
            backends_to_try = [("DirectShow", cv2.CAP_DSHOW)]
        else:
            # AUTO mode: Try MSMF first (DirectShow is often capped at ~10 FPS on laptop webcams), then DirectShow, then auto
            backends_to_try = [
                ("MSMF", cv2.CAP_MSMF),
                ("DirectShow", cv2.CAP_DSHOW),
                ("Default", 0),
            ]

        for name, backend_flag in backends_to_try:
            try:
                if backend_flag != 0:
                    cap = cv2.VideoCapture(self.config.device_index, backend_flag)
                else:
                    cap = cv2.VideoCapture(self.config.device_index)

                if cap is not None and cap.isOpened():
                    self.active_backend = name
                    return cap
            except Exception:
                continue

        return None

    def start(self) -> bool:
        """Initializes the webcam and starts background capture thread."""
        with self._lifecycle_lock:
            if self._running:
                return True

            self._cap = self._open_capture_device()
            if self._cap is None or not self._cap.isOpened():
                self.is_connected = False
                return False

            # Configure hardware capture properties.
            # MJPG lets most USB/laptop webcams deliver 30-60 FPS; raw YUY2 is often capped at 10-15 FPS.
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            # Minimize internal driver buffer to 1 frame to eliminate latency
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

            self.actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.config.frame_width)
            self.actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.config.frame_height)
            self.actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or self.config.fps)
            self.is_connected = True

            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, name="GesturaCameraCapture", daemon=True)
            self._thread.start()
            return True

    def _capture_loop(self) -> None:
        """Dedicated thread continuously reading the camera at hardware frame rate."""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                break

            ret, frame = self._cap.read()
            capture_time = time.perf_counter()

            if not ret or frame is None:
                self.is_connected = False
                time.sleep(0.008)
                continue

            self.is_connected = True

            # Mirror horizontally for natural, intuitive user interaction
            if self.mirror:
                frame = cv2.flip(frame, 1)

            with self._lock:
                self._latest_frame = frame
                self._latest_timestamp = capture_time
                self._frame_count += 1

                # Update measured camera capture FPS periodically
                elapsed = capture_time - self._last_fps_time
                if elapsed >= 0.8:
                    delta_frames = self._frame_count - self._last_fps_frame_count
                    self._fps_measured = delta_frames / elapsed
                    self._last_fps_time = capture_time
                    self._last_fps_frame_count = self._frame_count
                self._new_frame.notify_all()

    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Retrieves the newest captured frame and its acquisition timestamp.
        Does not wait or block; returns None if no frame is available.
        """
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame, self._latest_timestamp

    def wait_for_new_frame(self, last_timestamp: float, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, float]]:
        """Blocks until a frame newer than last_timestamp arrives (no busy-polling)."""
        with self._new_frame:
            self._new_frame.wait_for(
                lambda: self._latest_frame is not None and self._latest_timestamp != last_timestamp,
                timeout=timeout,
            )
            if self._latest_frame is None or self._latest_timestamp == last_timestamp:
                return None
            return self._latest_frame, self._latest_timestamp

    @property
    def capture_fps(self) -> float:
        """Measured hardware acquisition frame rate."""
        return self._fps_measured

    def change_resolution(self, width: int, height: int) -> bool:
        """Hot-reconfigures capture resolution."""
        with self._lifecycle_lock:
            self.config.frame_width = width
            self.config.frame_height = height
            if self._cap is not None and self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
                self.actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
                return True
        return False

    def change_fps(self, fps: int) -> bool:
        """Requests new hardware framerate (30 or 60)."""
        with self._lifecycle_lock:
            self.config.fps = fps
            if self._cap is not None and self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FPS, fps)
                self.actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or fps)
                return True
        return False
 
    def stop(self) -> None:
        """Stops background capture and safely releases webcam hardware."""
        with self._lifecycle_lock:
            self._running = False
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=1.5)
                self._thread = None

            if self._cap is not None:
                self._cap.release()
                self._cap = None

            self.is_connected = False
            with self._lock:
                self._latest_frame = None
