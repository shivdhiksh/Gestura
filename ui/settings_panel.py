"""
GESTURA - Settings Panel Widget
Comprehensive interactive control panel for sensitivity, smoothing, active zone geometry,
camera resolution/FPS, interaction durations, and individual gesture enable/disable mapping.
"""

from typing import Dict, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QPushButton,
    QScrollArea,
    QFrame,
    QComboBox,
    QCheckBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)

from config import AppConfig


class SettingsPanel(QFrame):
    """
    Settings sidebar providing interactive tuning of tracking responsiveness,
    hardware resolution, gesture triggers, and an interactive gesture mapping table.
    """

    # Signals emitted when parameters change
    sensitivity_changed = Signal(float)
    smoothing_changed = Signal(float, float)  # min_cutoff, beta
    active_zone_changed = Signal(float, float, float, float)
    resolution_changed = Signal(int, int)     # width, height
    fps_changed = Signal(int)                 # 30 or 60
    camera_index_changed = Signal(int)
    scroll_sensitivity_changed = Signal(float)
    drag_hold_duration_changed = Signal(float)
    two_hand_zoom_toggled = Signal(bool)
    gesture_enable_toggled = Signal(str, bool)

    def __init__(self, config: AppConfig, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("SettingsPanel")
        self.config = config
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Tab widget for organizing settings vs gesture mapping
        self.tabs = QTabWidget(self)
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #232838;
                background-color: #161922;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #12151D;
                color: #94A3B8;
                padding: 8px 14px;
                font-weight: 600;
                font-size: 11px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background: #1E2330;
                color: #38BDF8;
                border-bottom: 2px solid #38BDF8;
            }
        """)

        tab_controls = QWidget()
        self._build_controls_tab(tab_controls)
        self.tabs.addTab(tab_controls, "Controls & Tuning")

        tab_gestures = QWidget()
        self._build_gestures_tab(tab_gestures)
        self.tabs.addTab(tab_gestures, "Gesture Mapping")

        main_layout.addWidget(self.tabs)

    def _build_controls_tab(self, parent: QWidget) -> None:
        scroll = QScrollArea(parent)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(14)

        # =====================================================================
        # 1. CURSOR SENSITIVITY
        # =====================================================================
        sec_sens = QLabel("CURSOR SENSITIVITY")
        sec_sens.setProperty("class", "SectionHeader")
        layout.addWidget(sec_sens)

        sens_header = QHBoxLayout()
        sens_label = QLabel("Speed Multiplier")
        sens_label.setProperty("class", "SettingLabel")
        self.sens_val_lbl = QLabel(f"{self.config.gesture.sensitivity:.1f}x")
        self.sens_val_lbl.setProperty("class", "SettingValue")
        sens_header.addWidget(sens_label)
        sens_header.addStretch()
        sens_header.addWidget(self.sens_val_lbl)
        layout.addLayout(sens_header)

        self.slider_sens = QSlider(Qt.Orientation.Horizontal)
        self.slider_sens.setRange(5, 25)
        self.slider_sens.setValue(int(round(self.config.gesture.sensitivity * 10)))
        self.slider_sens.valueChanged.connect(self._on_sens_changed)
        layout.addWidget(self.slider_sens)

        # =====================================================================
        # 2. ADAPTIVE JITTER SMOOTHING (1 € Filter)
        # =====================================================================
        sec_smooth = QLabel("JITTER SUPPRESSION (1 € FILTER)")
        sec_smooth.setProperty("class", "SectionHeader")
        layout.addWidget(sec_smooth)

        preset_layout = QHBoxLayout()
        self.btn_preset_speed = QPushButton("Fast")
        self.btn_preset_speed.setProperty("class", "PresetBtn")
        self.btn_preset_speed.clicked.connect(lambda: self._apply_smoothing_preset(2.0, 0.040))

        self.btn_preset_balanced = QPushButton("Balanced")
        self.btn_preset_balanced.setProperty("class", "PresetBtn")
        self.btn_preset_balanced.clicked.connect(lambda: self._apply_smoothing_preset(1.2, 0.025))

        self.btn_preset_smooth = QPushButton("Smooth")
        self.btn_preset_smooth.setProperty("class", "PresetBtn")
        self.btn_preset_smooth.clicked.connect(lambda: self._apply_smoothing_preset(0.7, 0.012))

        preset_layout.addWidget(self.btn_preset_speed)
        preset_layout.addWidget(self.btn_preset_balanced)
        preset_layout.addWidget(self.btn_preset_smooth)
        layout.addLayout(preset_layout)

        cutoff_header = QHBoxLayout()
        cutoff_label = QLabel("Min Cutoff (Hz)")
        cutoff_label.setProperty("class", "SettingLabel")
        self.cutoff_val_lbl = QLabel(f"{self.config.smoothing.min_cutoff:.1f} Hz")
        self.cutoff_val_lbl.setProperty("class", "SettingValue")
        cutoff_header.addWidget(cutoff_label)
        cutoff_header.addStretch()
        cutoff_header.addWidget(self.cutoff_val_lbl)
        layout.addLayout(cutoff_header)

        self.slider_cutoff = QSlider(Qt.Orientation.Horizontal)
        self.slider_cutoff.setRange(3, 30)
        self.slider_cutoff.setValue(int(round(self.config.smoothing.min_cutoff * 10)))
        self.slider_cutoff.valueChanged.connect(self._on_smoothing_slider_changed)
        layout.addWidget(self.slider_cutoff)

        beta_header = QHBoxLayout()
        beta_label = QLabel("Velocity Beta")
        beta_label.setProperty("class", "SettingLabel")
        self.beta_val_lbl = QLabel(f"{self.config.smoothing.beta:.3f}")
        self.beta_val_lbl.setProperty("class", "SettingValue")
        beta_header.addWidget(beta_label)
        beta_header.addStretch()
        beta_header.addWidget(self.beta_val_lbl)
        layout.addLayout(beta_header)

        self.slider_beta = QSlider(Qt.Orientation.Horizontal)
        self.slider_beta.setRange(5, 50)
        self.slider_beta.setValue(int(round(self.config.smoothing.beta * 1000)))
        self.slider_beta.valueChanged.connect(self._on_smoothing_slider_changed)
        layout.addWidget(self.slider_beta)

        # =====================================================================
        # 3. WEBCAM HARDWARE SETTINGS
        # =====================================================================
        sec_cam = QLabel("WEBCAM HARDWARE CONFIG")
        sec_cam.setProperty("class", "SectionHeader")
        layout.addWidget(sec_cam)

        cam_form = QHBoxLayout()
        lbl_res = QLabel("Resolution:")
        lbl_res.setProperty("class", "SettingLabel")
        self.combo_res = QComboBox()
        self.combo_res.addItems(["640x480 (Recommended)", "320x240 (Fastest)"])
        self.combo_res.currentIndexChanged.connect(self._on_resolution_changed)

        lbl_fps = QLabel("FPS Target:")
        lbl_fps.setProperty("class", "SettingLabel")
        self.combo_fps = QComboBox()
        self.combo_fps.addItems(["60 FPS (Hardware Max)", "30 FPS (Standard)"])
        self.combo_fps.currentIndexChanged.connect(self._on_fps_target_changed)

        cam_form.addWidget(lbl_res)
        cam_form.addWidget(self.combo_res)
        cam_form.addWidget(lbl_fps)
        cam_form.addWidget(self.combo_fps)
        layout.addLayout(cam_form)

        # =====================================================================
        # 4. ACTIVE RECTANGLE TRACKING ZONE
        # =====================================================================
        sec_zone = QLabel("ACTIVE RECTANGLE ZONE")
        sec_zone.setProperty("class", "SectionHeader")
        layout.addWidget(sec_zone)

        # Zone Width
        zw_header = QHBoxLayout()
        zw_lbl = QLabel("Width Span")
        zw_lbl.setProperty("class", "SettingLabel")
        self.zw_val_lbl = QLabel(f"{int(round(self.config.active_zone.width_ratio * 100))}%")
        self.zw_val_lbl.setProperty("class", "SettingValue")
        zw_header.addWidget(zw_lbl)
        zw_header.addStretch()
        zw_header.addWidget(self.zw_val_lbl)
        layout.addLayout(zw_header)

        self.slider_zw = QSlider(Qt.Orientation.Horizontal)
        self.slider_zw.setRange(40, 90)
        self.slider_zw.setValue(int(round(self.config.active_zone.width_ratio * 100)))
        self.slider_zw.valueChanged.connect(self._on_zone_changed)
        layout.addWidget(self.slider_zw)

        # Zone Height
        zh_header = QHBoxLayout()
        zh_lbl = QLabel("Height Span")
        zh_lbl.setProperty("class", "SettingLabel")
        self.zh_val_lbl = QLabel(f"{int(round(self.config.active_zone.height_ratio * 100))}%")
        self.zh_val_lbl.setProperty("class", "SettingValue")
        zh_header.addWidget(zh_lbl)
        zh_header.addStretch()
        zh_header.addWidget(self.zh_val_lbl)
        layout.addLayout(zh_header)

        self.slider_zh = QSlider(Qt.Orientation.Horizontal)
        self.slider_zh.setRange(40, 90)
        self.slider_zh.setValue(int(round(self.config.active_zone.height_ratio * 100)))
        self.slider_zh.valueChanged.connect(self._on_zone_changed)
        layout.addWidget(self.slider_zh)

        # Horizontal Center Offset
        zox_header = QHBoxLayout()
        zox_lbl = QLabel("Center X Offset")
        zox_lbl.setProperty("class", "SettingLabel")
        self.zox_val_lbl = QLabel(f"{int(round(self.config.active_zone.center_x_offset * 100)):+d}%")
        self.zox_val_lbl.setProperty("class", "SettingValue")
        zox_header.addWidget(zox_lbl)
        zox_header.addStretch()
        zox_header.addWidget(self.zox_val_lbl)
        layout.addLayout(zox_header)

        self.slider_zox = QSlider(Qt.Orientation.Horizontal)
        self.slider_zox.setRange(-25, 25)
        self.slider_zox.setValue(int(round(self.config.active_zone.center_x_offset * 100)))
        self.slider_zox.valueChanged.connect(self._on_zone_changed)
        layout.addWidget(self.slider_zox)

        # Vertical Center Offset
        zoy_header = QHBoxLayout()
        zoy_lbl = QLabel("Center Y Offset")
        zoy_lbl.setProperty("class", "SettingLabel")
        self.zoy_val_lbl = QLabel(f"{int(round(self.config.active_zone.center_y_offset * 100)):+d}%")
        self.zoy_val_lbl.setProperty("class", "SettingValue")
        zoy_header.addWidget(zoy_lbl)
        zoy_header.addStretch()
        zoy_header.addWidget(self.zoy_val_lbl)
        layout.addLayout(zoy_header)

        self.slider_zoy = QSlider(Qt.Orientation.Horizontal)
        self.slider_zoy.setRange(-25, 25)
        self.slider_zoy.setValue(int(round(self.config.active_zone.center_y_offset * 100)))
        self.slider_zoy.valueChanged.connect(self._on_zone_changed)
        layout.addWidget(self.slider_zoy)

        btn_reset_zone = QPushButton("Reset Zone to Defaults")
        btn_reset_zone.setProperty("class", "PresetBtn")
        btn_reset_zone.clicked.connect(self._reset_zone_defaults)
        layout.addWidget(btn_reset_zone)

        # =====================================================================
        # 5. INTERACTION & SCROLLING PARAMETERS
        # =====================================================================
        sec_interact = QLabel("INTERACTION SENSITIVITY")
        sec_interact.setProperty("class", "SectionHeader")
        layout.addWidget(sec_interact)

        scroll_header = QHBoxLayout()
        scroll_lbl = QLabel("Scroll Sensitivity")
        scroll_lbl.setProperty("class", "SettingLabel")
        self.scroll_val_lbl = QLabel(f"{self.config.gesture.scroll_sensitivity:.1f}x")
        self.scroll_val_lbl.setProperty("class", "SettingValue")
        scroll_header.addWidget(scroll_lbl)
        scroll_header.addStretch()
        scroll_header.addWidget(self.scroll_val_lbl)
        layout.addLayout(scroll_header)

        self.slider_scroll = QSlider(Qt.Orientation.Horizontal)
        self.slider_scroll.setRange(5, 25)
        self.slider_scroll.setValue(int(round(self.config.gesture.scroll_sensitivity * 10)))
        self.slider_scroll.valueChanged.connect(self._on_scroll_sens_changed)
        layout.addWidget(self.slider_scroll)

        hold_header = QHBoxLayout()
        hold_lbl = QLabel("Pinch-to-Drag Hold Duration")
        hold_lbl.setProperty("class", "SettingLabel")
        self.hold_val_lbl = QLabel(f"{self.config.gesture.hold_duration_drag:.2f}s")
        self.hold_val_lbl.setProperty("class", "SettingValue")
        hold_header.addWidget(hold_lbl)
        hold_header.addStretch()
        hold_header.addWidget(self.hold_val_lbl)
        layout.addLayout(hold_header)

        self.slider_hold = QSlider(Qt.Orientation.Horizontal)
        self.slider_hold.setRange(20, 60)  # 0.20s to 0.60s
        self.slider_hold.setValue(int(round(self.config.gesture.hold_duration_drag * 100)))
        self.slider_hold.valueChanged.connect(self._on_drag_hold_changed)
        layout.addWidget(self.slider_hold)

        # Two-Hand Zoom Mode Toggle
        self.check_two_hand = QCheckBox("Enable Two-Hand Zoom Mode (Requires 2 Hands)")
        self.check_two_hand.setChecked(self.config.gesture.two_hand_zoom_enabled)
        self.check_two_hand.toggled.connect(self._on_two_hand_toggled)
        layout.addWidget(self.check_two_hand)

        layout.addStretch()
        scroll.setWidget(content)

        tab_layout = QVBoxLayout(parent)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)

    def _build_gestures_tab(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        lbl_desc = QLabel("Enable or disable specific hand gestures:")
        lbl_desc.setProperty("class", "SettingLabel")
        layout.addWidget(lbl_desc)

        # Table of all gestures
        self.gesture_table = QTableWidget(12, 3)
        self.gesture_table.setHorizontalHeaderLabels(["Hand Gesture", "Windows Action", "Active"])
        self.gesture_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.gesture_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.gesture_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.gesture_table.verticalHeader().setVisible(False)
        self.gesture_table.setStyleSheet("""
            QTableWidget {
                background-color: #12151D;
                gridline-color: #232838;
                border: 1px solid #232838;
                color: #F8FAFC;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #1A1E29;
                color: #94A3B8;
                padding: 6px;
                font-weight: 600;
                border: 1px solid #232838;
            }
        """)

        gesture_entries = [
            ("cursor_move", "Index finger only", "Move mouse cursor proportionally"),
            ("left_click", "Thumb + index pinch", "Left click / Double click"),
            ("drag_and_drop", "Hold thumb + index pinch", "Drag and drop selection"),
            ("right_click", "Thumb + middle pinch", "Right click context menu"),
            ("scrolling", "Two fingers (index + middle)", "Vertical and horizontal scroll"),
            ("volume_control", "Thumb up / Thumb down", "Increase / decrease volume"),
            ("brightness_control", "Three fingers (index+mid+ring)", "Screen brightness adjust"),
            ("media_controls", "Thumb + ring pinch", "Play / pause media playback"),
            ("media_controls", "Thumb + pinky pinch", "Next track (single) / Prev (double)"),
            ("app_switch", "Open-palm swipe right/left", "Switch next / previous application"),
            ("window_maximize", "Four fingers up, thumb folded", "Maximize / restore active window"),
            ("pause_resume", "Closed fist / Open palm", "Pause / resume tracking"),
        ]

        for row, (key, gesture_desc, action_desc) in enumerate(gesture_entries):
            item_gesture = QTableWidgetItem(gesture_desc)
            item_gesture.setFlags(Qt.ItemFlag.ItemIsEnabled)
            item_action = QTableWidgetItem(action_desc)
            item_action.setFlags(Qt.ItemFlag.ItemIsEnabled)

            cb = QCheckBox()
            is_enabled = self.config.gesture.enabled_gestures.get(key, True)
            cb.setChecked(is_enabled)
            cb.toggled.connect(lambda checked, k=key: self._on_gesture_check_toggled(k, checked))

            cell_widget = QWidget()
            cb_layout = QHBoxLayout(cell_widget)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            cb_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cb_layout.addWidget(cb)

            self.gesture_table.setItem(row, 0, item_gesture)
            self.gesture_table.setItem(row, 1, item_action)
            self.gesture_table.setCellWidget(row, 2, cell_widget)

        layout.addWidget(self.gesture_table)

    def _on_gesture_check_toggled(self, key: str, checked: bool) -> None:
        self.config.gesture.enabled_gestures[key] = checked
        self.gesture_enable_toggled.emit(key, checked)

    def _on_sens_changed(self, value: int) -> None:
        sens = value / 10.0
        self.sens_val_lbl.setText(f"{sens:.1f}x")
        self.config.gesture.sensitivity = sens
        self.sensitivity_changed.emit(sens)

    def _apply_smoothing_preset(self, min_cutoff: float, beta: float) -> None:
        self.slider_cutoff.blockSignals(True)
        self.slider_beta.blockSignals(True)

        self.slider_cutoff.setValue(int(round(min_cutoff * 10)))
        self.slider_beta.setValue(int(round(beta * 1000)))

        self.cutoff_val_lbl.setText(f"{min_cutoff:.1f} Hz")
        self.beta_val_lbl.setText(f"{beta:.3f}")

        self.slider_cutoff.blockSignals(False)
        self.slider_beta.blockSignals(False)

        self.config.smoothing.min_cutoff = min_cutoff
        self.config.smoothing.beta = beta
        self.smoothing_changed.emit(min_cutoff, beta)

    def _on_smoothing_slider_changed(self) -> None:
        min_cutoff = self.slider_cutoff.value() / 10.0
        beta = self.slider_beta.value() / 1000.0

        self.cutoff_val_lbl.setText(f"{min_cutoff:.1f} Hz")
        self.beta_val_lbl.setText(f"{beta:.3f}")

        self.config.smoothing.min_cutoff = min_cutoff
        self.config.smoothing.beta = beta
        self.smoothing_changed.emit(min_cutoff, beta)

    def _on_resolution_changed(self, index: int) -> None:
        if index == 0:
            w, h = 640, 480
        else:
            w, h = 320, 240
        self.resolution_changed.emit(w, h)

    def _on_fps_target_changed(self, index: int) -> None:
        fps = 60 if index == 0 else 30
        self.fps_changed.emit(fps)

    def _on_scroll_sens_changed(self, value: int) -> None:
        sens = value / 10.0
        self.scroll_val_lbl.setText(f"{sens:.1f}x")
        self.config.gesture.scroll_sensitivity = sens
        self.scroll_sensitivity_changed.emit(sens)

    def _on_drag_hold_changed(self, value: int) -> None:
        hold_s = value / 100.0
        self.hold_val_lbl.setText(f"{hold_s:.2f}s")
        self.config.gesture.hold_duration_drag = hold_s
        self.drag_hold_duration_changed.emit(hold_s)

    def _on_two_hand_toggled(self, checked: bool) -> None:
        self.config.gesture.two_hand_zoom_enabled = checked
        self.config.gesture.enabled_gestures["two_hand_zoom"] = checked
        self.two_hand_zoom_toggled.emit(checked)

    def _on_zone_changed(self) -> None:
        w_ratio = self.slider_zw.value() / 100.0
        h_ratio = self.slider_zh.value() / 100.0
        off_x = self.slider_zox.value() / 100.0
        off_y = self.slider_zoy.value() / 100.0

        self.zw_val_lbl.setText(f"{int(w_ratio * 100)}%")
        self.zh_val_lbl.setText(f"{int(h_ratio * 100)}%")
        self.zox_val_lbl.setText(f"{int(off_x * 100):+d}%")
        self.zoy_val_lbl.setText(f"{int(off_y * 100):+d}%")

        self.config.active_zone.width_ratio = w_ratio
        self.config.active_zone.height_ratio = h_ratio
        self.config.active_zone.center_x_offset = off_x
        self.config.active_zone.center_y_offset = off_y

        self.active_zone_changed.emit(w_ratio, h_ratio, off_x, off_y)

    def _reset_zone_defaults(self) -> None:
        self.slider_zw.setValue(65)
        self.slider_zh.setValue(70)
        self.slider_zox.setValue(0)
        self.slider_zoy.setValue(0)
