"""
GESTURA — Real-Time Hand Gesture Laptop Controller
Application Entry Point
"""

import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from config import AppConfig
from ui.main_window import MainWindow


def main():
    # Set Windows execution policy and high DPI support
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("GESTURA")
    app.setOrganizationName("DeepMind")

    # Load configuration
    config = AppConfig()

    # Launch desktop application
    window = MainWindow(config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

