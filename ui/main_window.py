"""
GESTURA - Main Window Application
Integrates camera acquisition, real-time hand landmark detection, gesture classification,
native Windows action dispatch, and a telemetry dashboard.
"""

import ctypes
from ctypes import wintypes
import time
from typing import Dict, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QMessageBox,
)

from config import AppConfig
from core.camera import CameraManager
from core.coordinate_mapper import CoordinateMapper
from core.action_controller import ActionController
from core.gesture_recognizer import GestureRecognizer
from core.hand_tracker import HandTracker
from ui.camera_view import CameraView
from ui.settings_panel import SettingsPanel
from ui.styles import DARK_THEME_QSS
from utils.performance_monitor import PerformanceMonitor


# Win32 Constants for Global Hotkey
WM_HOTKEY = 0x0312
VK_F8 = 0x77
HOTKEY_ID = 9001


class TrackingWorker(QThread):
    """
    Dedicated background worker thread running the real-time computer vision and action pipeline:
    Camera acquisition -> Hand tracking -> Gesture recognition -> Windows actions -> HUD rendering.
    """

    frame_processed = Signal(QImage, dict)
    camera_status_changed = Signal(bool, str)

    def __init__(
        self,
        config: AppConfig,
        camera: CameraManager,
        tracker: HandTracker,
        mapper: CoordinateMapper,
        actions: ActionController,
        gesture_recognizer: GestureRecognizer,
        perf_monitor: PerformanceMonitor,
    ):
        super().__init__()
        self.config = config
        self.camera = camera
        self.tracker = tracker
        self.mapper = mapper
        self.actions = actions
        self.gestures = gesture_recognizer
        self.perf_monitor = perf_monitor

        self._is_running: bool = False
        # Set while the GUI thread still has an unpainted frame; avoids a growing signal queue (lag + low FPS)
        self.gui_busy: bool = False

    def run(self) -> None:
        self._is_running = True
        last_capture_time: float = -1.0

        # Start hardware camera acquisition
        if not self.camera.start():
            self.camera_status_changed.emit(False, "Failed to initialize webcam hardware.")
            return

        self.camera_status_changed.emit(
            True,
            f"{self.camera.active_backend}: {self.camera.actual_width}x{self.camera.actual_height} @ {int(self.camera.actual_fps)} FPS"
        )

        while self._is_running:
            try:
                # Block until the camera thread publishes a new frame (no 1 ms busy-polling)
                frame_data = self.camera.wait_for_new_frame(last_capture_time, timeout=0.1)
                if frame_data is None:
                    continue

                raw_frame, capture_time = frame_data
                last_capture_time = capture_time

                # 1. MediaPipe 21-point Hand Landmark Detection
                tracking_result = self.tracker.process_frame(raw_frame)

                # 2. Extract Geometric Features for Primary & Secondary Hands
                features = None
                second_features = None
                is_inside = False
                screen_coords = None

                if tracking_result.hand_detected:
                    features = self.gestures.extract_features(tracking_result.landmarks_norm)
                    if tracking_result.second_landmarks_norm:
                        second_features = self.gestures.extract_features(tracking_result.second_landmarks_norm)

                    if tracking_result.index_tip_norm is not None:
                        norm_x, norm_y = tracking_result.index_tip_norm
                        is_inside = self.mapper.is_inside_zone(norm_x, norm_y)
                        if is_inside:
                            screen_coords = self.mapper.map_to_screen(norm_x, norm_y)

                # 3. Gesture Classification & Action Execution
                active_gesture_name = self.gestures.update_gestures(
                    features=features,
                    second_features=second_features,
                    is_inside_active_zone=is_inside,
                    timestamp=capture_time,
                )

                # 4. Native Windows Cursor Movement
                # Cursor only follows index finger during normal tracking or drag-and-drop
                cursor_pos = None
                if active_gesture_name in ("CURSOR TRACKING", "DRAGGING (HOLD PINCH)", "HAND DETECTED"):
                    cursor_pos = self.actions.update_position(screen_coords, capture_time)
                    if cursor_pos is not None:
                        self.perf_monitor.record_cursor_update(capture_time)
                else:
                    self.actions.update_position(None)

                # 5. Measure Full Pipeline Timing
                finish_time = time.perf_counter()
                self.perf_monitor.record_frame(
                    capture_timestamp=capture_time,
                    finish_timestamp=finish_time,
                    inference_time_ms=tracking_result.inference_time_ms,
                )

                # Skip preview rendering while the GUI is still painting the previous frame;
                # cursor control above already ran, so tracking FPS is not tied to GUI speed.
                if self.gui_busy:
                    continue

                # 6. Visual HUD Rendering (Active Zone, Hand Skeleton, Gesture Badge)
                annotated = self.mapper.draw_active_zone(
                    frame=raw_frame,
                    cursor_control_enabled=self.actions.is_active and not self.gestures.is_paused,
                    is_fingertip_inside=is_inside,
                    has_hand=tracking_result.hand_detected,
                )
                annotated = self.tracker.draw_landmarks(
                    frame=annotated,
                    tracking_result=tracking_result,
                    is_inside_active_zone=is_inside,
                    gesture_name=active_gesture_name,
                )

                # 7. Convert Frame for PySide6 GUI Thread
                h, w, ch = annotated.shape
                bytes_per_line = ch * w
                q_img = QImage(
                    annotated.data, w, h, bytes_per_line, QImage.Format.Format_BGR888
                ).copy()

                telemetry = {
                    "cam_fps": self.camera.capture_fps,
                    "fps": self.perf_monitor.fps,
                    "latency_ms": self.perf_monitor.latency_ms,
                    "p95_latency_ms": self.perf_monitor.p95_latency_ms,
                    "inference_ms": self.perf_monitor.inference_time_ms,
                    "inference_fps": self.perf_monitor.inference_fps,
                    "cursor_rate": self.perf_monitor.cursor_rate,
                    "cpu_percent": self.perf_monitor.cpu_percent,
                    "memory_mb": self.perf_monitor.memory_mb,
                    "hand_detected": tracking_result.hand_detected,
                    "handedness": tracking_result.handedness,
                    "is_inside": is_inside,
                    "cursor_active": self.actions.is_active,
                    "is_paused": self.gestures.is_paused,
                    "gesture_name": active_gesture_name,
                    "cursor_pos": cursor_pos,
                }

                self.gui_busy = True
                self.frame_processed.emit(q_img, telemetry)
            except Exception as e:
                import traceback
                traceback.print_exc()
                time.sleep(0.01)

        # Cleanup camera hardware when thread stops
        self.camera.stop()

    def stop(self) -> None:
        self._is_running = False
        self.wait(1500)


class MainWindow(QMainWindow):
    """
    Main Application Dashboard for GESTURA.
    """

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config

        self.setWindowTitle("GESTURA — Real-Time Hand Gesture Laptop Controller")
        self.setMinimumSize(1180, 750)
        self.setStyleSheet(DARK_THEME_QSS)

        # Initialize Core Engines
        self.camera = CameraManager(self.config.camera, mirror=self.config.gesture.mirror_preview)
        self.tracker = HandTracker(
            self.config.model_path,
            max_num_hands=2 if self.config.gesture.two_hand_zoom_enabled else 1,
        )
        self.actions = ActionController(self.config.smoothing)
        self.cursor = self.actions  # Alias for backward compatibility
        self.mapper = CoordinateMapper(
            zone_config=self.config.active_zone,
            screen_width=self.actions.screen_width,
            screen_height=self.actions.screen_height,
            sensitivity=self.config.gesture.sensitivity,
            edge_padding=0.08,
        )
        self.gestures = GestureRecognizer(self.config.gesture, self.actions)
        self.perf_monitor = PerformanceMonitor(window_size=30)

        # Setup GUI Widgets
        self._init_ui()

        # Connect Worker Thread
        self.worker = TrackingWorker(
            config=self.config,
            camera=self.camera,
            tracker=self.tracker,
            mapper=self.mapper,
            actions=self.actions,
            gesture_recognizer=self.gestures,
            perf_monitor=self.perf_monitor,
        )
        self.worker.frame_processed.connect(self._on_frame_processed)
        self.worker.camera_status_changed.connect(self._on_camera_status_changed)

        # Register Emergency Shortcuts
        self._register_hotkeys()

        # Start Camera Pipeline (Control starts in OFF mode for safety)
        self.worker.start()

    def _init_ui(self) -> None:
        central_widget = QWidget(self)
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # =====================================================================
        # 1. HEADER BAR
        # =====================================================================
        header = QFrame(self)
        header.setObjectName("HeaderBar")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 10, 18, 10)

        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(2)
        app_title = QLabel("GESTURA")
        app_title.setObjectName("AppTitle")
        app_sub = QLabel("Real-Time Hand Gesture Laptop Controller")
        app_sub.setObjectName("AppSubtitle")
        title_vbox.addWidget(app_title)
        title_vbox.addWidget(app_sub)
        header_layout.addLayout(title_vbox)

        header_layout.addStretch()

        # Emergency Pause Hotkey Badge
        hotkey_badge = QLabel("Emergency Pause: [F8]")
        hotkey_badge.setObjectName("HotkeyBadge")
        header_layout.addWidget(hotkey_badge)

        header_layout.addSpacing(16)

        # Control Buttons
        self.btn_start = QPushButton("▶ Start Gesture Control")
        self.btn_start.setObjectName("BtnStart")
        self.btn_start.clicked.connect(self.start_gesture_control)

        self.btn_stop = QPushButton("⏹ Stop Gesture Control")
        self.btn_stop.setObjectName("BtnStop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_gesture_control)

        header_layout.addWidget(self.btn_start)
        header_layout.addWidget(self.btn_stop)

        root_layout.addWidget(header)

        # =====================================================================
        # 2. TELEMETRY METRIC CARDS
        # =====================================================================
        telemetry_bar = QFrame(self)
        telemetry_bar.setStyleSheet("background-color: #12151D; border-bottom: 1px solid #1E2330;")
        tb_layout = QHBoxLayout(telemetry_bar)
        tb_layout.setContentsMargins(14, 8, 14, 8)
        tb_layout.setSpacing(10)

        # Telemetry Cards
        self.card_cam_fps = self._create_metric_card("CAM FPS", "0.0 FPS", "val_cam_fps")
        self.card_proc_fps = self._create_metric_card("TRACKING FPS", "0.0 FPS", "val_proc_fps")
        self.card_cursor_rate = self._create_metric_card("CURSOR RATE", "0.0 Hz", "val_cursor_rate")
        self.card_latency = self._create_metric_card("LATENCY (AVG / P95)", "0.0 / 0.0 ms", "val_lat")
        self.card_gesture = self._create_metric_card("ACTIVE GESTURE", "IDLE", "val_gesture", is_accent=True)
        self.card_state = self._create_metric_card("CONTROL STATUS", "DISABLED", "val_state")
        self.card_resources = self._create_metric_card("CPU / RAM", "0% / 0 MB", "val_resources")

        tb_layout.addWidget(self.card_cam_fps)
        tb_layout.addWidget(self.card_proc_fps)
        tb_layout.addWidget(self.card_cursor_rate)
        tb_layout.addWidget(self.card_latency)
        tb_layout.addWidget(self.card_gesture)
        tb_layout.addWidget(self.card_state)
        tb_layout.addWidget(self.card_resources)

        root_layout.addWidget(telemetry_bar)

        # =====================================================================
        # 3. MAIN CONTENT: Camera View + Settings Panel
        # =====================================================================
        content_frame = QWidget(self)
        content_layout = QHBoxLayout(content_frame)
        content_layout.setContentsMargins(14, 14, 14, 14)
        content_layout.setSpacing(14)

        # Camera Preview Box
        preview_box = QFrame(self)
        preview_box.setObjectName("PreviewContainer")
        pb_layout = QVBoxLayout(preview_box)
        pb_layout.setContentsMargins(4, 4, 4, 4)

        self.camera_view = CameraView(preview_box)
        pb_layout.addWidget(self.camera_view)
        content_layout.addWidget(preview_box, stretch=3)

        # Settings Panel
        self.settings_panel = SettingsPanel(self.config, content_frame)
        self.settings_panel.setFixedWidth(380)

        # Connect setting signals
        self.settings_panel.sensitivity_changed.connect(self._on_sensitivity_updated)
        self.settings_panel.smoothing_changed.connect(self._on_smoothing_updated)
        self.settings_panel.resolution_changed.connect(self._on_resolution_changed)
        self.settings_panel.fps_changed.connect(self._on_fps_changed)
        self.settings_panel.two_hand_zoom_toggled.connect(self._on_two_hand_toggled)

        content_layout.addWidget(self.settings_panel, stretch=1)
        root_layout.addWidget(content_frame, stretch=1)

        # Status Bar
        self.statusBar().showMessage("GESTURA Ready. Click 'Start Gesture Control' or press F8 to activate.")

    def _create_metric_card(self, title: str, default_val: str, attr_name: str, is_accent: bool = False) -> QFrame:
        card = QFrame()
        card.setProperty("class", "MetricCard")
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(6, 4, 6, 4)
        vbox.setSpacing(1)

        lbl_title = QLabel(title)
        lbl_title.setProperty("class", "MetricTitle")

        lbl_val = QLabel(default_val)
        lbl_val.setProperty("class", "MetricValueAccent" if is_accent else "MetricValue")
        setattr(self, attr_name, lbl_val)

        vbox.addWidget(lbl_title)
        vbox.addWidget(lbl_val)
        return card

    def _register_hotkeys(self) -> None:
        """Configures Windows Global F8 Hotkey + In-App Qt F8 Shortcut."""
        self.shortcut_f8 = QShortcut(QKeySequence(Qt.Key.Key_F8), self)
        self.shortcut_f8.activated.connect(self.toggle_gesture_control)

        try:
            user32 = ctypes.windll.user32
            user32.RegisterHotKey(int(self.winId()), HOTKEY_ID, 0x4000, VK_F8)
        except Exception as e:
            print(f"[MainWindow] Global hotkey registration note: {e}")

    def nativeEvent(self, eventType, message) -> tuple:
        """Catches Windows WM_HOTKEY events for global F8 emergency pause."""
        try:
            if eventType in (b"windows_generic_MSG", "windows_generic_MSG"):
                msg = wintypes.MSG.from_address(int(message))
                if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID:
                    self.toggle_gesture_control()
                    return True, 0
        except Exception:
            pass
        return super().nativeEvent(eventType, message)

    def start_gesture_control(self) -> None:
        """Activates real-time cursor positioning and gesture actions."""
        self.actions.enable()
        self.gestures.is_paused = False
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.val_state.setText("ENGAGED")
        self.val_state.setStyleSheet("color: #10B981;")
        self.statusBar().showMessage("Gesture control ACTIVE. Move index fingertip inside the active rectangle.")

    def stop_gesture_control(self) -> None:
        """Immediately halts cursor movement, releases held buttons, and freezes position."""
        self.actions.disable()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.val_state.setText("DISABLED")
        self.val_state.setStyleSheet("color: #EF4444;")
        self.statusBar().showMessage("Gesture control STOPPED. All actions released.")

    def toggle_gesture_control(self) -> None:
        """Emergency F8 Hotkey toggle."""
        if self.actions.is_active:
            self.stop_gesture_control()
        else:
            self.start_gesture_control()

    def _on_sensitivity_updated(self, sens: float) -> None:
        self.mapper.update_sensitivity(sens)

    def _on_smoothing_updated(self, min_cutoff: float, beta: float) -> None:
        self.actions.update_smoothing_parameters(min_cutoff=min_cutoff, beta=beta)

    def _on_resolution_changed(self, width: int, height: int) -> None:
        self.camera.change_resolution(width, height)

    def _on_fps_changed(self, fps: int) -> None:
        self.camera.change_fps(fps)

    def _on_two_hand_toggled(self, enabled: bool) -> None:
        self.tracker.set_max_hands(2 if enabled else 1)

    def _on_camera_status_changed(self, success: bool, message: str) -> None:
        if not success:
            QMessageBox.critical(self, "Camera Error", message)

    def _on_frame_processed(self, q_img: QImage, telemetry: dict) -> None:
        """Updates live viewport and telemetry cards."""
        self.camera_view.update_frame(q_img)
        self.worker.gui_busy = False

        # Camera & Processing FPS
        cam_fps = telemetry.get("cam_fps", 0.0)
        proc_fps = telemetry.get("fps", 0.0)
        self.val_cam_fps.setText(f"{cam_fps:.1f} FPS")
        self.val_proc_fps.setText(f"{proc_fps:.1f} FPS")

        # Cursor update rate
        cur_rate = telemetry.get("cursor_rate", 0.0)
        self.val_cursor_rate.setText(f"{cur_rate:.1f} Hz")

        # Software Latency (Average / P95)
        lat = telemetry.get("latency_ms", 0.0)
        p95 = telemetry.get("p95_latency_ms", 0.0)
        self.val_lat.setText(f"{lat:.1f} / {p95:.1f} ms")

        # Active Gesture
        gesture = telemetry.get("gesture_name", "IDLE")
        self.val_gesture.setText(gesture)

        # Control State
        is_active = telemetry.get("cursor_active", False)
        is_paused = telemetry.get("is_paused", False)
        is_inside = telemetry.get("is_inside", False)
        has_hand = telemetry.get("hand_detected", False)

        if not is_active:
            self.val_state.setText("DISABLED")
            self.val_state.setStyleSheet("color: #64748B;")
        elif is_paused:
            self.val_state.setText("PAUSED")
            self.val_state.setStyleSheet("color: #F59E0B;")
        elif not has_hand:
            self.val_state.setText("SEARCHING...")
            self.val_state.setStyleSheet("color: #94A3B8;")
        elif not is_inside:
            self.val_state.setText("FROZEN (OUTSIDE)")
            self.val_state.setStyleSheet("color: #F59E0B;")
        else:
            self.val_state.setText("ENGAGED")
            self.val_state.setStyleSheet("color: #10B981;")

        # CPU% and RAM
        cpu = telemetry.get("cpu_percent", 0.0)
        mem = telemetry.get("memory_mb", 0.0)
        self.val_resources.setText(f"{cpu:.0f}% / {mem:.0f} MB")

    def closeEvent(self, event) -> None:
        """Ensures all camera, threads, actions, and hotkeys are released safely."""
        try:
            ctypes.windll.user32.UnregisterHotKey(int(self.winId()), HOTKEY_ID)
        except Exception:
            pass

        self.actions.disable()

        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.stop()

        if hasattr(self, "tracker") and self.tracker:
            self.tracker.close()

        super().closeEvent(event)
