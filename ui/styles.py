"""
GESTURA - Modern Dark Theme Stylesheet
Clean, professional desktop interface with high-contrast accents and subtle styling.
"""

DARK_THEME_QSS = """
/* Global Window Styling */
QMainWindow, QWidget#CentralWidget {
    background-color: #0F1117;
    color: #E2E8F0;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
}

/* Header Area */
QFrame#HeaderBar {
    background-color: #161922;
    border-bottom: 1px solid #232838;
    padding: 8px 16px;
}

QLabel#AppTitle {
    color: #FFFFFF;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 1.5px;
}

QLabel#AppSubtitle {
    color: #94A3B8;
    font-size: 12px;
}

QLabel#HotkeyBadge {
    background-color: #262B3D;
    color: #38BDF8;
    border: 1px solid #38BDF8;
    border-radius: 6px;
    padding: 4px 10px;
    font-weight: 600;
    font-size: 11px;
}

/* Primary Action Buttons */
QPushButton#BtnStart {
    background-color: #059669;
    color: #FFFFFF;
    border: 1px solid #10B981;
    border-radius: 8px;
    padding: 10px 22px;
    font-size: 14px;
    font-weight: 600;
}
QPushButton#BtnStart:hover {
    background-color: #10B981;
}
QPushButton#BtnStart:pressed {
    background-color: #047857;
}
QPushButton#BtnStart:disabled {
    background-color: #1E293B;
    color: #64748B;
    border-color: #334155;
}

QPushButton#BtnStop {
    background-color: #DC2626;
    color: #FFFFFF;
    border: 1px solid #EF4444;
    border-radius: 8px;
    padding: 10px 22px;
    font-size: 14px;
    font-weight: 600;
}
QPushButton#BtnStop:hover {
    background-color: #EF4444;
}
QPushButton#BtnStop:pressed {
    background-color: #B91C1C;
}
QPushButton#BtnStop:disabled {
    background-color: #1E293B;
    color: #64748B;
    border-color: #334155;
}

/* Metric Cards */
QFrame.MetricCard {
    background-color: #161922;
    border: 1px solid #232838;
    border-radius: 8px;
    padding: 10px;
}

QLabel.MetricTitle {
    color: #94A3B8;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

QLabel.MetricValue {
    color: #F8FAFC;
    font-size: 18px;
    font-weight: 700;
}

QLabel.MetricValueAccent {
    color: #10B981;
    font-size: 18px;
    font-weight: 700;
}

/* Camera Preview Viewport */
QFrame#PreviewContainer {
    background-color: #090B10;
    border: 1px solid #232838;
    border-radius: 10px;
}

/* Settings Side Panel */
QFrame#SettingsPanel {
    background-color: #161922;
    border: 1px solid #232838;
    border-radius: 10px;
    padding: 14px;
}

QLabel.SectionHeader {
    color: #F1F5F9;
    font-size: 14px;
    font-weight: 700;
    margin-top: 6px;
    margin-bottom: 4px;
}

QLabel.SettingLabel {
    color: #CBD5E1;
    font-size: 12px;
    font-weight: 500;
}

QLabel.SettingValue {
    color: #38BDF8;
    font-size: 12px;
    font-weight: 600;
}

/* Sliders */
QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background: #232838;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #4F46E5;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #818CF8;
    border: 2px solid #FFFFFF;
    width: 14px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #A5B4FC;
}

/* Preset Buttons */
QPushButton.PresetBtn {
    background-color: #1E2330;
    color: #CBD5E1;
    border: 1px solid #2E3547;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 11px;
    font-weight: 600;
}
QPushButton.PresetBtn:hover {
    background-color: #2E3547;
    color: #FFFFFF;
}
QPushButton.PresetBtn:checked {
    background-color: #4338CA;
    border-color: #6366F1;
    color: #FFFFFF;
}

/* Scroll Area */
QScrollArea {
    border: none;
    background: transparent;
}
QScrollArea > QWidget > QWidget {
    background: transparent;
}

/* Status Bar */
QStatusBar {
    background-color: #12151D;
    color: #94A3B8;
    border-top: 1px solid #1E2330;
    font-size: 12px;
}
"""

