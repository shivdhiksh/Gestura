"""
GESTURA - Windows Native Action Controller
Executes system actions via Windows Win32 APIs (mouse, keyboard, audio, display, windows)
with thread safety and automatic button/key release on tracking loss.
"""

import ctypes
import threading
from ctypes import wintypes
import time
from typing import Optional, Tuple
import screen_brightness_control as sbc

from core.smoothing import OneEuroFilter2D
from config import SmoothingConfig


# Win32 Mouse Event Flags
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x00010
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x01000

# Win32 Virtual Key Codes
VK_TAB = 0x09
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12  # Alt key
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3
KEYEVENTF_KEYUP = 0x0002

# Window Commands
SW_MAXIMIZE = 3
SW_RESTORE = 9


class ActionController:
    """
    Directly controls Windows desktop cursor, mouse clicks, scrolling,
    system volume, screen brightness, media controls, and window states.
    """

    def __init__(self, smoothing_config: SmoothingConfig):
        self._set_dpi_awareness()
        self._user32 = ctypes.windll.user32
        self.screen_width = self._user32.GetSystemMetrics(0)
        self.screen_height = self._user32.GetSystemMetrics(1)

        self.smoothing = OneEuroFilter2D(
            min_cutoff=smoothing_config.min_cutoff,
            beta=smoothing_config.beta,
            d_cutoff=smoothing_config.d_cutoff,
        )

        self._is_active: bool = False
        self._is_dragging: bool = False
        self._last_cursor_pos: Optional[Tuple[int, int]] = None
        self._needs_reentry_reset: bool = True
        self._cached_brightness: Optional[int] = None
        self._last_brightness_query: float = 0.0
        self._last_update_time: float = 0.0

        # Glide thread: the camera only delivers ~30 targets/s, so a 30 Hz SetCursorPos looks choppy.
        # This thread eases the real cursor toward the latest target at ~200 Hz.
        self._glide_target: Optional[Tuple[float, float]] = None
        self._glide_pos: Optional[Tuple[float, float]] = None
        self._glide_lock = threading.Lock()
        self._glide_thread: Optional[threading.Thread] = None

    # Seconds without a target before the smoothing filter is reset (a brief misclassified
    # frame should not reset the filter and cause a jump).
    REENTRY_RESET_GAP = 0.25
    GLIDE_HZ = 200.0
    GLIDE_FACTOR = 0.35
    def _set_dpi_awareness(self) -> None:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def is_dragging(self) -> bool:
        return self._is_dragging

    def enable(self) -> None:
        self._is_active = True
        self._needs_reentry_reset = True
        self.smoothing.reset()
        with self._glide_lock:
            self._glide_target = None
            self._glide_pos = None
        if self._glide_thread is None or not self._glide_thread.is_alive():
            self._glide_thread = threading.Thread(target=self._glide_loop, name="GesturaCursorGlide", daemon=True)
            self._glide_thread.start()

    def _glide_loop(self) -> None:
        period = 1.0 / self.GLIDE_HZ
        while self._is_active:
            with self._glide_lock:
                target = self._glide_target
                pos = self._glide_pos
                if target is not None:
                    if pos is None:
                        pos = target
                    else:
                        pos = (
                            pos[0] + (target[0] - pos[0]) * self.GLIDE_FACTOR,
                            pos[1] + (target[1] - pos[1]) * self.GLIDE_FACTOR,
                        )
                    self._glide_pos = pos
            if pos is not None and target is not None:
                ix, iy = int(round(pos[0])), int(round(pos[1]))
                if (ix, iy) != self._last_cursor_pos:
                    self._user32.SetCursorPos(ix, iy)
                    self._last_cursor_pos = (ix, iy)
            time.sleep(period)

    def disable(self) -> None:
        self._is_active = False
        with self._glide_lock:
            self._glide_target = None
            self._glide_pos = None
        self.release_all()
        self._needs_reentry_reset = True
        self.smoothing.reset()

    def toggle(self) -> bool:
        if self._is_active:
            self.disable()
        else:
            self.enable()
        return self._is_active

    def update_smoothing_parameters(
        self,
        min_cutoff: Optional[float] = None,
        beta: Optional[float] = None,
        d_cutoff: Optional[float] = None,
    ) -> None:
        self.smoothing.set_parameters(min_cutoff, beta, d_cutoff)

    # =========================================================================
    # 1. CURSOR POSITIONING
    # =========================================================================
    def update_position(
        self,
        target_coords: Optional[Tuple[float, float]],
        timestamp: Optional[float] = None,
    ) -> Optional[Tuple[int, int]]:
        """Moves hardware cursor smoothly to target_coords if active."""
        if not self._is_active:
            self._needs_reentry_reset = True
            return None

        if target_coords is None:
            # Freeze the cursor where it is; only reset the filter after a real gap
            with self._glide_lock:
                self._glide_target = None
                self._glide_pos = None
            return None

        raw_x, raw_y = target_coords
        if timestamp is None:
            timestamp = time.perf_counter()

        if self._last_update_time and (timestamp - self._last_update_time) > self.REENTRY_RESET_GAP:
            self._needs_reentry_reset = True
        self._last_update_time = timestamp

        if self._needs_reentry_reset:
            self.smoothing.reset()
            self._needs_reentry_reset = False

        smooth_x, smooth_y = self.smoothing.filter(raw_x, raw_y, timestamp)
        target_x = max(0, min(self.screen_width - 1, int(round(smooth_x))))
        target_y = max(0, min(self.screen_height - 1, int(round(smooth_y))))

        if self._glide_thread is not None and self._glide_thread.is_alive():
            with self._glide_lock:
                if self._glide_pos is None:
                    # Start gliding from the real cursor position to avoid a snap
                    pt = wintypes.POINT()
                    self._user32.GetCursorPos(ctypes.byref(pt))
                    self._glide_pos = (float(pt.x), float(pt.y))
                self._glide_target = (float(target_x), float(target_y))
        else:
            self._user32.SetCursorPos(target_x, target_y)
            self._last_cursor_pos = (target_x, target_y)
        return target_x, target_y

    # =========================================================================
    # 2. MOUSE CLICKS & DRAGGING
    # =========================================================================
    def left_click(self) -> None:
        """Dispatches single left click."""
        if not self._is_active:
            return
        self._user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self._user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def double_click(self) -> None:
        """Dispatches double left click."""
        if not self._is_active:
            return
        self.left_click()
        time.sleep(0.04)
        self.left_click()

    def right_click(self) -> None:
        """Dispatches single right click."""
        if not self._is_active:
            return
        self._user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        self._user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

    def start_drag(self) -> None:
        """Initiates drag & drop (holds left mouse button down)."""
        if not self._is_active or self._is_dragging:
            return
        self._user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self._is_dragging = True

    def stop_drag(self) -> None:
        """Releases held left mouse button."""
        if self._is_dragging:
            self._user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self._is_dragging = False

    # =========================================================================
    # 3. SCROLLING
    # =========================================================================
    def scroll(self, delta_x: float, delta_y: float) -> None:
        """Dispatches vertical and horizontal scroll wheel events."""
        if not self._is_active:
            return
        # Vertical scroll (WHEEL_DELTA = 120)
        if abs(delta_y) > 0.001:
            wheel_amount = int(round(delta_y * 120))
            if wheel_amount != 0:
                self._user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, wheel_amount, 0)

        # Horizontal scroll (WHEEL_DELTA = 120)
        if abs(delta_x) > 0.001:
            hwheel_amount = int(round(delta_x * 120))
            if hwheel_amount != 0:
                self._user32.mouse_event(MOUSEEVENTF_HWHEEL, 0, 0, hwheel_amount, 0)

    # =========================================================================
    # 4. SYSTEM VOLUME CONTROL
    # =========================================================================
    def volume_up(self) -> None:
        """Steps Windows master volume upward."""
        self._user32.keybd_event(VK_VOLUME_UP, 0, 0, 0)
        self._user32.keybd_event(VK_VOLUME_UP, 0, KEYEVENTF_KEYUP, 0)

    def volume_down(self) -> None:
        """Steps Windows master volume downward."""
        self._user32.keybd_event(VK_VOLUME_DOWN, 0, 0, 0)
        self._user32.keybd_event(VK_VOLUME_DOWN, 0, KEYEVENTF_KEYUP, 0)

    # =========================================================================
    # 5. SCREEN BRIGHTNESS CONTROL
    # =========================================================================
    def change_brightness(self, delta_percent: int) -> Optional[int]:
        """Adjusts display brightness upward or downward."""
        try:
            now = time.perf_counter()
            if self._cached_brightness is None or (now - self._last_brightness_query) > 5.0:
                vals = sbc.get_brightness()
                self._cached_brightness = vals[0] if isinstance(vals, list) and vals else 50
                self._last_brightness_query = now

            new_val = max(0, min(100, self._cached_brightness + delta_percent))
            sbc.set_brightness(new_val)
            self._cached_brightness = new_val
            return new_val
        except Exception:
            return None

    # =========================================================================
    # 6. MEDIA PLAYBACK CONTROLS
    # =========================================================================
    def media_play_pause(self) -> None:
        """Toggles play / pause across media players and browsers."""
        self._user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 0, 0)
        self._user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, KEYEVENTF_KEYUP, 0)

    def media_next_track(self) -> None:
        """Skips to next track."""
        self._user32.keybd_event(VK_MEDIA_NEXT_TRACK, 0, 0, 0)
        self._user32.keybd_event(VK_MEDIA_NEXT_TRACK, 0, KEYEVENTF_KEYUP, 0)

    def media_prev_track(self) -> None:
        """Replays or skips to previous track."""
        self._user32.keybd_event(VK_MEDIA_PREV_TRACK, 0, 0, 0)
        self._user32.keybd_event(VK_MEDIA_PREV_TRACK, 0, KEYEVENTF_KEYUP, 0)

    # =========================================================================
    # 7. WINDOW MANAGEMENT & APPLICATION SWITCHING
    # =========================================================================
    def switch_app_next(self) -> None:
        """Dispatches Alt+Tab to switch to next application."""
        self._user32.keybd_event(VK_MENU, 0, 0, 0)
        self._user32.keybd_event(VK_TAB, 0, 0, 0)
        self._user32.keybd_event(VK_TAB, 0, KEYEVENTF_KEYUP, 0)
        self._user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)

    def switch_app_prev(self) -> None:
        """Dispatches Alt+Shift+Tab to switch to previous application."""
        self._user32.keybd_event(VK_MENU, 0, 0, 0)
        self._user32.keybd_event(VK_SHIFT, 0, 0, 0)
        self._user32.keybd_event(VK_TAB, 0, 0, 0)
        self._user32.keybd_event(VK_TAB, 0, KEYEVENTF_KEYUP, 0)
        self._user32.keybd_event(VK_SHIFT, 0, KEYEVENTF_KEYUP, 0)
        self._user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)

    def toggle_window_maximize(self) -> None:
        """Toggles active foreground window between Maximized and Restored states."""
        try:
            hwnd = self._user32.GetForegroundWindow()
            if hwnd:
                is_zoomed = self._user32.IsZoomed(hwnd)
                cmd = SW_RESTORE if is_zoomed else SW_MAXIMIZE
                self._user32.ShowWindow(hwnd, cmd)
        except Exception:
            pass

    # =========================================================================
    # 8. TWO-HAND ZOOM MODE
    # =========================================================================
    def zoom(self, delta: float) -> None:
        """Dispatches Ctrl + Mouse Wheel for zoom in (delta > 0) or zoom out (delta < 0)."""
        if not self._is_active:
            return
        direction = 120 if delta > 0 else -120
        self._user32.keybd_event(VK_CONTROL, 0, 0, 0)
        self._user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, direction, 0)
        self._user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)

    # =========================================================================
    # 9. SAFETY RELEASE
    # =========================================================================
    def release_all(self) -> None:
        """
        Immediately releases all held mouse buttons and modifier keys.
        Ensures zero stuck state upon tracking loss, pause, or application stop.
        """
        if self._is_dragging:
            self._user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self._is_dragging = False

        self._user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        self._user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
        self._user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)
        self._user32.keybd_event(VK_SHIFT, 0, KEYEVENTF_KEYUP, 0)
        self._user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)

