"""
GESTURA - Camera Preview Widget
PySide6 viewport widget rendering live webcam preview with high-DPI aspect-ratio preservation.
"""

from typing import Optional
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPainter, QImage, QColor, QFont
from PySide6.QtWidgets import QWidget


class CameraView(QWidget):
    """
    Renders the live video feed centered within the widget,
    maintaining the source aspect ratio and rendering smooth scaled frames.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self._current_image: Optional[QImage] = None
        self._placeholder_text: str = "Awaiting Camera Feed..."

        # Enable double buffering and optimize updates
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

    def update_frame(self, image: QImage) -> None:
        """Receives new video frame from background worker and requests repaint."""
        self._current_image = image
        self.update()

    def set_placeholder_text(self, text: str) -> None:
        self._placeholder_text = text
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        width = self.width()
        height = self.height()

        # Background fill
        painter.fillRect(0, 0, width, height, QColor("#090B10"))

        if self._current_image is not None and not self._current_image.isNull():
            # Calculate aspect-ratio preserving bounding box
            img_w = self._current_image.width()
            img_h = self._current_image.height()

            scale = min(width / img_w, height / img_h)
            target_w = int(round(img_w * scale))
            target_h = int(round(img_h * scale))

            offset_x = (width - target_w) // 2
            offset_y = (height - target_h) // 2

            target_rect = QRect(offset_x, offset_y, target_w, target_h)
            painter.drawImage(target_rect, self._current_image)
        else:
            # Standby Placeholder
            painter.setPen(QColor("#64748B"))
            font = QFont("Segoe UI", 13, QFont.Weight.Medium)
            painter.setFont(font)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                self._placeholder_text,
            )

