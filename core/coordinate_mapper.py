"""
GESTURA - Coordinate Mapper Module
Maps normalized index fingertip coordinates within the Active Tracking Zone
to proportional desktop screen coordinates with anti-jump protection and visual HUD overlays.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from config import ActiveZoneConfig


class CoordinateMapper:
    """
    Manages the active tracking zone boundary and maps normalized hand coordinates
    to native screen pixel coordinates.
    """

    def __init__(
        self,
        zone_config: ActiveZoneConfig,
        screen_width: int,
        screen_height: int,
        sensitivity: float = 1.0,
        edge_padding: float = 0.0,
    ):
        self.zone_config = zone_config
        self.screen_width = max(1, int(screen_width))
        self.screen_height = max(1, int(screen_height))
        self.sensitivity = float(sensitivity)
        # Inner fraction of the zone that already maps to the screen edge, so corners are
        # reachable without the fingertip leaving the zone (which freezes the cursor).
        self.edge_padding = max(0.0, min(0.3, float(edge_padding)))

        # Hysteresis margin to prevent rapid jitter toggling at boundaries
        self._hysteresis: float = 0.012
        self._was_inside: bool = False

    def update_screen_size(self, width: int, height: int) -> None:
        self.screen_width = max(1, int(width))
        self.screen_height = max(1, int(height))

    def update_sensitivity(self, sensitivity: float) -> None:
        self.sensitivity = max(0.2, min(3.0, float(sensitivity)))

    def is_inside_zone(self, norm_x: float, norm_y: float) -> bool:
        """
        Check if normalized coordinates (norm_x, norm_y) fall inside the active rectangle.
        Uses slight hysteresis margin to avoid border flickering when hovering on the boundary.
        """
        x_min = self.zone_config.x_min
        x_max = self.zone_config.x_max
        y_min = self.zone_config.y_min
        y_max = self.zone_config.y_max

        if self._was_inside:
            # Slightly expand exit boundary to prevent rapid on/off jitter
            inside = (
                (x_min - self._hysteresis) <= norm_x <= (x_max + self._hysteresis)
                and (y_min - self._hysteresis) <= norm_y <= (y_max + self._hysteresis)
            )
        else:
            # Standard strict boundary for entry
            inside = (x_min <= norm_x <= x_max) and (y_min <= norm_y <= y_max)

        self._was_inside = inside
        return inside

    def map_to_screen(
        self, norm_x: float, norm_y: float
    ) -> Optional[Tuple[float, float]]:
        """
        Maps normalized fingertip position inside the active rectangle proportionally
        to the full screen resolution.

        Returns:
            (screen_x, screen_y) as floating point pixel coordinates if inside zone,
            or None if the fingertip is outside the active rectangle (cursor should freeze).
        """
        if not self.is_inside_zone(norm_x, norm_y):
            return None

        x_min = self.zone_config.x_min
        x_max = self.zone_config.x_max
        y_min = self.zone_config.y_min
        y_max = self.zone_config.y_max

        span_x = max(1e-5, x_max - x_min)
        span_y = max(1e-5, y_max - y_min)

        # Proportional position within the rectangle in range [0.0, 1.0]
        u = (norm_x - x_min) / span_x
        v = (norm_y - y_min) / span_y

        if self.edge_padding > 0.0:
            usable = 1.0 - 2.0 * self.edge_padding
            u = (u - self.edge_padding) / usable
            v = (v - self.edge_padding) / usable

        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))

        # Apply sensitivity scaling relative to center (0.5, 0.5)
        if abs(self.sensitivity - 1.0) > 1e-4:
            u = 0.5 + (u - 0.5) * self.sensitivity
            v = 0.5 + (v - 0.5) * self.sensitivity
            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))

        # Map to physical screen pixels
        screen_x = u * (self.screen_width - 1)
        screen_y = v * (self.screen_height - 1)

        return screen_x, screen_y

    def get_pixel_rect(
        self, frame_width: int, frame_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Computes the pixel bounding box (x, y, w, h) of the active zone for drawing on frame.
        """
        x1 = int(round(self.zone_config.x_min * frame_width))
        y1 = int(round(self.zone_config.y_min * frame_height))
        x2 = int(round(self.zone_config.x_max * frame_width))
        y2 = int(round(self.zone_config.y_max * frame_height))

        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        return x1, y1, w, h

    def draw_active_zone(
        self,
        frame: np.ndarray,
        cursor_control_enabled: bool,
        is_fingertip_inside: bool,
        has_hand: bool,
    ) -> np.ndarray:
        """
        Renders a prominent rectangular active tracking boundary on the camera frame.
        Colors and status badges dynamically reflect tracking state:
        - Emerald/Green: Control Active & Fingertip Inside (Cursor moving)
        - Amber/Orange: Control Active & Fingertip Outside / No Hand (Cursor frozen)
        - Muted Slate: Control Disabled (Standby)
        """
        fh, fw = frame.shape[:2]
        x1, y1, w, h = self.get_pixel_rect(fw, fh)
        x2, y2 = x1 + w, y1 + h

        # 1. Subtle semi-transparent vignette outside active rectangle
        for region in (
            frame[: max(0, y1), :],
            frame[min(fh, y2):, :],
            frame[max(0, y1):min(fh, y2), : max(0, x1)],
            frame[max(0, y1):min(fh, y2), min(fw, x2):],
        ):
            if region.size:
                region[:] = cv2.convertScaleAbs(region, alpha=0.65, beta=4)

        # 2. Determine border styling and status message
        if not cursor_control_enabled:
            border_color = (130, 120, 100)       # Slate Gray (BGR)
            badge_bg = (50, 45, 38)
            status_text = "STANDBY - GESTURE CONTROL OFF"
        elif has_hand and is_fingertip_inside:
            border_color = (80, 240, 120)        # Emerald / Neon Green (BGR)
            badge_bg = (25, 90, 40)
            status_text = "TRACKING ACTIVE [CURSOR ENGAGED]"
        elif has_hand and not is_fingertip_inside:
            border_color = (0, 165, 255)         # Warm Amber (BGR)
            badge_bg = (10, 60, 120)
            status_text = "CURSOR FROZEN [FINGER OUTSIDE ACTIVE ZONE]"
        else:
            border_color = (80, 130, 200)        # Muted Blue-Orange (BGR)
            badge_bg = (40, 50, 70)
            status_text = "SEARCHING FOR HAND [MOVE HAND INTO VIEW]"

        # 3. Draw main active zone rectangle border
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2, cv2.LINE_AA)

        # 4. Draw high-visibility futuristic corner brackets (HUD aesthetic)
        corner_len = min(24, int(min(w, h) * 0.15))
        corner_thick = 3
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), border_color, corner_thick, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), border_color, corner_thick, cv2.LINE_AA)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), border_color, corner_thick, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), border_color, corner_thick, cv2.LINE_AA)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), border_color, corner_thick, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), border_color, corner_thick, cv2.LINE_AA)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), border_color, corner_thick, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), border_color, corner_thick, cv2.LINE_AA)

        # 5. Draw status banner above active rectangle
        text_size, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tw, th = text_size
        badge_y1 = max(4, y1 - th - 10)
        badge_y2 = y1 - 2
        badge_x1 = x1
        badge_x2 = min(fw - 4, x1 + tw + 16)

        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), badge_bg, -1)
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), border_color, 1)
        cv2.putText(
            frame,
            status_text,
            (badge_x1 + 8, badge_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return frame
