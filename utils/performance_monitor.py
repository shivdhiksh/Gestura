"""
GESTURA - Performance Monitor Utility
Measures true pipeline latency (ms, average & P95), processing frame rate (FPS),
cursor update frequency, CPU percentage, and RAM memory utilization using high-resolution timers.
"""

from collections import deque
import os
import time
from typing import Deque, Dict, Optional
import numpy as np
import psutil


class PerformanceMonitor:
    """
    Monitors end-to-end processing pipeline metrics:
    - Real-time rolling frame rate (FPS)
    - Average and 95th-percentile software processing latency (ms)
    - MediaPipe neural network inference latency & FPS capability
    - Cursor dispatch rate (Hz)
    - System process CPU% and RAM usage (MB)
    """

    def __init__(self, window_size: int = 30):
        self.window_size = max(10, int(window_size))

        self._frame_timestamps: Deque[float] = deque(maxlen=self.window_size)
        self._latencies_ms: Deque[float] = deque(maxlen=self.window_size)
        self._inference_times_ms: Deque[float] = deque(maxlen=self.window_size)
        self._cursor_timestamps: Deque[float] = deque(maxlen=self.window_size)

        self._latest_fps: float = 0.0
        self._latest_latency_ms: float = 0.0
        self._avg_latency_ms: float = 0.0
        self._p95_latency_ms: float = 0.0
        self._latest_inference_ms: float = 0.0
        self._cursor_rate_hz: float = 0.0

        # Process telemetry
        self._process = psutil.Process(os.getpid())
        self._cpu_percent: float = 0.0
        self._memory_mb: float = 0.0
        self._last_process_check: float = 0.0

    def record_cursor_update(self, now: Optional[float] = None) -> None:
        """Records a successful native cursor displacement event."""
        t = now if now is not None else time.perf_counter()
        self._cursor_timestamps.append(t)
        if len(self._cursor_timestamps) >= 2:
            elapsed = self._cursor_timestamps[-1] - self._cursor_timestamps[0]
            if elapsed > 1e-4:
                self._cursor_rate_hz = (len(self._cursor_timestamps) - 1) / elapsed

    def record_frame(
        self,
        capture_timestamp: float,
        finish_timestamp: Optional[float] = None,
        inference_time_ms: float = 0.0,
    ) -> None:
        """Records the completion of a full frame pipeline iteration."""
        now = finish_timestamp if finish_timestamp is not None else time.perf_counter()

        # Update FPS timestamps
        self._frame_timestamps.append(now)
        if len(self._frame_timestamps) >= 2:
            elapsed = self._frame_timestamps[-1] - self._frame_timestamps[0]
            if elapsed > 1e-4:
                self._latest_fps = (len(self._frame_timestamps) - 1) / elapsed

        # Calculate true end-to-end software latency (ms)
        latency = max(0.1, (now - capture_timestamp) * 1000.0)
        self._latencies_ms.append(latency)
        self._latest_latency_ms = latency
        self._avg_latency_ms = float(np.mean(self._latencies_ms))
        self._p95_latency_ms = float(np.percentile(self._latencies_ms, 95))

        # Record inference time
        if inference_time_ms > 0:
            self._inference_times_ms.append(inference_time_ms)
            self._latest_inference_ms = inference_time_ms

        # Periodically refresh CPU and RAM telemetry (every 0.5s to minimize overhead)
        if (now - self._last_process_check) >= 0.5:
            try:
                self._cpu_percent = self._process.cpu_percent(interval=None)
                self._memory_mb = self._process.memory_info().rss / (1024 * 1024)
            except Exception:
                pass
            self._last_process_check = now

    @property
    def fps(self) -> float:
        return self._latest_fps

    @property
    def latency_ms(self) -> float:
        return self._avg_latency_ms

    @property
    def p95_latency_ms(self) -> float:
        return self._p95_latency_ms

    @property
    def latest_latency_ms(self) -> float:
        return self._latest_latency_ms

    @property
    def inference_time_ms(self) -> float:
        return self._latest_inference_ms

    @property
    def inference_fps(self) -> float:
        return (1000.0 / self._latest_inference_ms) if self._latest_inference_ms > 0 else 0.0

    @property
    def cursor_rate(self) -> float:
        return self._cursor_rate_hz

    @property
    def cpu_percent(self) -> float:
        return self._cpu_percent

    @property
    def memory_mb(self) -> float:
        return self._memory_mb

    def get_telemetry_dict(self) -> Dict[str, float]:
        return {
            "fps": self.fps,
            "latency_ms": self.latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "inference_ms": self.inference_time_ms,
            "inference_fps": self.inference_fps,
            "cursor_rate": self.cursor_rate,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
        }

    def reset(self) -> None:
        self._frame_timestamps.clear()
        self._latencies_ms.clear()
        self._inference_times_ms.clear()
        self._cursor_timestamps.clear()
        self._latest_fps = 0.0
        self._latest_latency_ms = 0.0
        self._avg_latency_ms = 0.0
        self._p95_latency_ms = 0.0
        self._latest_inference_ms = 0.0
        self._cursor_rate_hz = 0.0
