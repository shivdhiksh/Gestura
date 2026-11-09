"""
GESTURA - Adaptive Smoothing Module
Implements the 1 € (One Euro) Filter for low-latency, jitter-free cursor tracking.
Reference: Géry Casiez, Nicolas Roussel, Daniel Vogel. 
"1 € Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems." CHI 2012.
"""

import math
import time
from typing import Optional, Tuple


class LowPassFilter:
    """Standard first-order exponential low-pass filter."""

    def __init__(self, alpha: float = 1.0):
        self._alpha: float = alpha
        self._last_val: Optional[float] = None

    def reset(self) -> None:
        self._last_val = None

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self._alpha = alpha

        if self._last_val is None:
            s = value
        else:
            s = self._alpha * value + (1.0 - self._alpha) * self._last_val
        self._last_val = s
        return s

    @property
    def last_value(self) -> Optional[float]:
        return self._last_val


class OneEuroFilter:
    """
    Adaptive 1 € Filter for 1D scalar signal.
    Dynamically adjusts cutoff frequency based on instantaneous speed:
    - At low speeds (slow movements/holding still): cutoff is low (min_cutoff), eliminating jitter.
    - At high speeds: cutoff expands with speed (beta * |speed|), eliminating lag.
    """

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.02,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._last_time: Optional[float] = None

    def set_parameters(
        self,
        min_cutoff: Optional[float] = None,
        beta: Optional[float] = None,
        d_cutoff: Optional[float] = None,
    ) -> None:
        if min_cutoff is not None:
            self.min_cutoff = float(min_cutoff)
        if beta is not None:
            self.beta = float(beta)
        if d_cutoff is not None:
            self.d_cutoff = float(d_cutoff)

    def reset(self) -> None:
        """Reset filter history (e.g. on hand exit/entry or tracking loss)."""
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_time = None

    @property
    def last_value(self) -> Optional[float]:
        """Returns the most recent filtered value."""
        return self._x_filter.last_value

    @staticmethod
    def _compute_alpha(cutoff: float, dt: float) -> float:
        """Calculate smoothing factor alpha = 1 / (1 + 1 / (2 * pi * cutoff * dt))."""
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        """
        Filter a new data point x observed at timestamp (seconds).
        If timestamp is None, current monotonic time is used.
        """
        if timestamp is None:
            timestamp = time.perf_counter()

        if self._last_time is None or self._x_filter.last_value is None:
            self._last_time = timestamp
            self._dx_filter.reset()
            return self._x_filter.filter(x, alpha=1.0)

        dt = timestamp - self._last_time
        if dt <= 1e-6:
            # Timestamp too close to previous, return last filtered value
            return self._x_filter.last_value or x

        self._last_time = timestamp

        # Estimate derivative (speed) and smooth it
        prev_x = self._x_filter.last_value
        dx = (x - prev_x) / dt
        d_alpha = self._compute_alpha(self.d_cutoff, dt)
        edx = self._dx_filter.filter(dx, alpha=d_alpha)

        # Dynamic cutoff frequency based on smoothed velocity
        cutoff = self.min_cutoff + self.beta * abs(edx)

        # Filter signal with dynamic cutoff
        alpha = self._compute_alpha(cutoff, dt)
        return self._x_filter.filter(x, alpha=alpha)


class OneEuroFilter2D:
    """2D wrapper for coordinates (X, Y) using independent 1 € Filters."""

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.02,
        d_cutoff: float = 1.0,
    ):
        self.filter_x = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.filter_y = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def set_parameters(
        self,
        min_cutoff: Optional[float] = None,
        beta: Optional[float] = None,
        d_cutoff: Optional[float] = None,
    ) -> None:
        self.filter_x.set_parameters(min_cutoff, beta, d_cutoff)
        self.filter_y.set_parameters(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self.filter_x.reset()
        self.filter_y.reset()

    def filter(
        self, x: float, y: float, timestamp: Optional[float] = None
    ) -> Tuple[float, float]:
        if timestamp is None:
            timestamp = time.perf_counter()
        fx = self.filter_x.filter(x, timestamp)
        fy = self.filter_y.filter(y, timestamp)
        return fx, fy
