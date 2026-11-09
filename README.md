# GESTURA — Real-Time Hand Gesture Laptop Controller

**GESTURA** is a high-performance Windows desktop application engineered to control your laptop cursor and trigger native Windows actions using natural hand gestures captured by a standard webcam. It prioritizes ultra-low software processing latency (<30 ms), true 30+ FPS capture, zero cursor lag, jitter-free cursor stabilization via the **1 € Filter**, and proportional coordinate mapping constrained within an **Active Rectangle Tracking Zone**.

---

## Technology Stack

- **Language & Runtime**: Python 3.10 – 3.13 (Tested on Python 3.13.7 64-bit)
- **Computer Vision & Video Capture**: OpenCV (`opencv-python>=4.10.0`) with Microsoft Media Foundation (`CAP_MSMF`) and DirectShow (`CAP_DSHOW`) fallback
- **Hand Tracking & ML Inference**: Google MediaPipe (`mediapipe>=1.0.0`) HandLandmarker running in continuous video mode (`RunningMode.VIDEO`) on CPU (TensorFlow Lite XNNPACK delegate)
- **Graphical User Interface**: PySide6 (`PySide6>=6.5.0`) with hardware-accelerated `QPainter` video rendering, multi-tab configuration panels, and modern dark theme
- **Mathematical Modeling & Filtering**: NumPy (`numpy>=1.24.0`) and 1 € Filter adaptive velocity-based low-pass filtering
- **Windows System Integration**: Native Win32 APIs via Python `ctypes` (`user32.dll` for `SetCursorPos`, `mouse_event`, `keybd_event`, `RegisterHotKey`, `ShowWindow`), plus `screen-brightness-control`
- **Telemetry & Benchmarking**: `psutil` process monitoring, rolling statistical estimators, and high-resolution performance counters (`time.perf_counter`)
- **Testing Suite**: `pytest>=7.0.0` with mock Win32 dispatchers, synthetic hand landmark generators, and pipeline timing benchmarks

---

## Key Features & Architecture

```
Webcam Hardware (Integrated / USB Sensor, 30 FPS)
        │
        ▼ (Microsoft Media Foundation cv2.CAP_MSMF, Buffer=1)
CameraManager (Dedicated Capture Thread)
        │ Latest Frame Buffer (zero queue buildup)
        ▼
TrackingWorker (Dedicated QThread)
        │
        ├─► HandTracker (MediaPipe HandLandmarker in VIDEO Mode, CPU ~65–75 FPS capability)
        │         │ 21 Normalized 3D Landmarks
        │         ▼
        ├─► GestureRecognizer (HandFeatures Extractor & 16-Gesture State Machine)
        │         │ Disambiguation, Hysteresis, Hold Timers, Cooldowns, Per-Gesture Toggles
        │         ▼
        ├─► ActionController / CursorController (Native Win32 ctypes API)
        │         │ SetCursorPos + 1 € Filter + mouse_event + keybd_event + ShowWindow
        │         ▼
        └─► Visual HUD Renderer (Active Rectangle, Hand Skeleton, State Badges)
                  │ QImage + Telemetry Dictionary
                  ▼
PySide6 GUI Thread (MainWindow)
        ├─► CameraView (Mirrored QPainter High-DPI Viewport)
        ├─► Telemetry Cards (Cam FPS, Pipeline FPS, Cursor Rate, Latency, CPU%, RAM MB)
        └─► SettingsPanel (Tabbed: "Controls & Tuning" & "Gesture Mapping" Table)
```

1. **Active Rectangle Tracking Zone (Primary Interaction Area)**:
   - A clearly demarcated rectangular boundary in the webcam viewport proportionally maps to the laptop's entire screen resolution.
   - **Boundary Freeze Safety**: The cursor moves **only** when your index fingertip (MediaPipe Landmark #8) is inside the rectangle. Exiting the zone immediately freezes cursor position, eliminating erratic jumps or unintended clicks.
   - **Customizable Geometry**: Adjust width span, height span, and horizontal/vertical offsets directly from the settings panel.

2. **Ultra-Low Latency & High FPS Camera Pipeline**:
   - **Auto-Selected MSMF Backend**: Automatically initializes Microsoft Media Foundation (`cv2.CAP_MSMF`) to achieve full hardware frame rate (30.0+ FPS), eliminating the 7.5 FPS bottleneck inherent in DirectShow uncompressed video streams.
   - **Latest-Frame-Only Buffer**: Buffer size 1 (`cv2.CAP_PROP_BUFFERSIZE = 1`) on a dedicated capture thread drops stale frames and prevents queue buildup.
   - **MediaPipe HandLandmarker in VIDEO Mode**: 21-point hand tracking running at **~65–75 FPS capability (~13–15 ms per frame)** on CPU (AMD Ryzen 7 4800H).

3. **1 € Adaptive Low-Latency Smoothing (Jitter Suppression)**:
   - Based on the CHI 2012 algorithm by Casiez et al.
   - **At rest or low velocity**: Low cutoff frequency (`min_cutoff = 1.2 Hz`) heavily suppresses micro-tremors, keeping cursor position rock steady.
   - **At high velocity**: Dynamically expands cutoff frequency with velocity (`beta = 0.02`), eliminating drag and lag.
   - Quick presets: *Fast (Low Latency)*, *Balanced (Default)*, *Smooth (Ultra Jitter Suppression)*.

4. **Comprehensive 16-Gesture State Machine**:
   - Geometric feature extraction disambiguates finger pinches, curls, and extensions.
   - Confidence thresholds, hysteresis, configurable hold durations, and debouncing prevent false activations.

5. **Safety & Hardware Control**:
   - **Default OFF**: Gesture control starts in a safe disabled state upon launch.
   - **System-Wide Emergency Hotkey `[F8]`**: Press F8 anytime from any application on your PC to immediately freeze or resume cursor control.
   - **Automatic Action Release**: When tracking is lost or control is stopped, all held mouse buttons and keys are automatically released (`release_all()`).

---

## Gesture Mapping Reference

All gestures map directly to native Windows input events:

| Hand Gesture | Action | Implementation Mechanism |
|---|---|---|
| **Index finger only** | Move mouse cursor | Native Win32 `SetCursorPos` + 1 € Filter |
| **Thumb + index pinch** | Left click | Win32 `mouse_event(MOUSEEVENTF_LEFTDOWN / UP)` |
| **Hold thumb + index pinch (>0.35s)** | Drag and drop | Win32 `MOUSEEVENTF_LEFTDOWN` hold until pinch release |
| **Thumb + middle pinch** | Right click | Win32 `mouse_event(MOUSEEVENTF_RIGHTDOWN / UP)` |
| **Two fingers (index + middle)** | Vertical & horizontal scroll | Win32 `MOUSEEVENTF_WHEEL` & `MOUSEEVENTF_HWHEEL` |
| **Thumb up** | Increase volume | Win32 `keybd_event(VK_VOLUME_UP)` |
| **Thumb down** | Decrease volume | Win32 `keybd_event(VK_VOLUME_DOWN)` |
| **Three fingers (index + middle + ring)** | Adjust screen brightness | `screen-brightness-control` API (+/- 5%) |
| **Thumb + ring pinch** | Play / Pause media | Win32 `keybd_event(VK_MEDIA_PLAY_PAUSE)` |
| **Thumb + little finger pinch** | Next track | Win32 `keybd_event(VK_MEDIA_NEXT_TRACK)` |
| **Double thumb + little finger pinch** | Previous track | Win32 `keybd_event(VK_MEDIA_PREV_TRACK)` |
| **Open-palm horizontal swipe right** | Switch to next application | Win32 `Alt + Tab` sequence |
| **Open-palm horizontal swipe left** | Switch to previous application | Win32 `Alt + Shift + Tab` sequence |
| **Four fingers held up, thumb folded** | Maximize / restore window | Win32 `ShowWindow(SW_MAXIMIZE / RESTORE)` |
| **Closed fist held (>0.4s)** | Pause gesture control | State machine pause + `release_all()` |
| **Open palm held (>0.4s)** | Resume gesture control | State machine unpause |
| **Two hands index-pinch spread (Optional)** | Zoom in / Zoom out | Win32 `Ctrl + MOUSEEVENTF_WHEEL` |

*Every gesture can be individually enabled or disabled in the UI under Settings > Gesture Mapping.*

---

## Hardware & System Requirements

- **Operating System**: Windows 10 or Windows 11 (64-bit)
- **Processor (CPU)**: Intel Core i5 / AMD Ryzen 5 or higher (Tested on AMD Ryzen 7 4800H 8-Core)
- **Memory (RAM)**: 4 GB minimum (8+ GB recommended; application consumes ~250–350 MB)
- **Camera**: Standard integrated laptop webcam or external USB webcam (640x480 resolution at 30 FPS minimum)
- **Permissions**: Standard user permissions; camera access enabled in Windows Privacy Settings

---

## Measured Performance Benchmarks

*Empirical measurements taken on AMD Ryzen 7 4800H, Windows 11, Integrated Webcam (640x480):*

| Metric | Measured Baseline (DirectShow) | Optimized (MSMF Auto) | Target Requirement |
|---|---|---|---|
| **Camera Capture FPS** | 7.5 FPS | **30.0 – 31.0 FPS** | Maximum sensor capability |
| **MediaPipe Inference** | ~70 FPS (14.2 ms) | **~65 – 75 FPS (13.5 ms)** | Single-hand video mode |
| **Cursor Dispatch Frequency** | 7.6 Hz | **30.0 Hz** | Synchronous with camera |
| **Average End-to-End Latency** | 133.0 ms | **23.4 ms** | <50.0 ms |
| **95th Percentile Latency (P95)** | 145.0 ms | **35.6 ms** | Low variance |
| **CPU Utilization** | ~15% | **25% – 35%** | Efficient on CPU |
| **RAM Footprint** | ~180 MB | **~250 – 350 MB** | Lightweight desktop app |

> **Hardware Limitation Note**: The integrated laptop camera hardware sensor is physically locked at 30 FPS across 640x480, 320x240, and 1280x720 resolutions. While GESTURA requests 60 FPS and the MediaPipe inference engine easily processes frames at 70+ FPS, camera capture runs at the physical sensor maximum of 30.0 FPS.

---

## Testing Results

The test suite covers unit and integration tests across all modules:

| Test Module | Tests | Description | Status |
|---|---|---|---|
| `test_action_controller.py` | 5 | Win32 cursor positioning, clicks, drag lifecycle, `release_all()`, boundary clamping | **PASSED** |
| `test_camera.py` | 2 | Backend negotiation (MSMF/DSHOW), buffer size, frame acquisition, resolution hot-swap | **PASSED** |
| `test_coordinate_mapper.py` | 6 | Zone boundary math, proportional mapping, outside freeze, directional logic | **PASSED** |
| `test_cursor_controller.py` | 5 | Cursor enable/disable toggle, initial state, position update, boundary clamping | **PASSED** |
| `test_gesture_recognizer.py` | 8 | Feature extraction, fist pause, open-palm resume, clicks, drag, right click, scroll, volume | **PASSED** |
| `test_hand_tracker.py` | 3 | HandLandmarker initialization, empty frame handling, HUD landmark drawing | **PASSED** |
| `test_performance_monitor.py` | 2 | Rolling FPS, average and P95 latency, cursor update rate, psutil CPU/RAM metrics | **PASSED** |
| `test_smoothing.py` | 7 | Low-pass filter reset, zero-dt handling, 1 € Filter jitter reduction, high-speed tracking | **PASSED** |
| `test_ui_and_pipeline.py` | 2 | End-to-end pipeline benchmark (<50 ms), MainWindow lifecycle, F8 safety toggle | **PASSED** |
| **Total** | **40** | **Comprehensive Full Project Test Suite** | **100% PASSED** |

---

## Project Structure

```
GESTURA/
│
├── .gitignore                # Comprehensive exclusions (venv, caches, logs, IDE files)
├── config.py                 # Dataclasses for Active Zone, Camera, Smoothing, and Gestures
├── main.py                   # Desktop application entry point (PySide6 QApplication)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── assets/
│   └── models/
│       └── hand_landmarker.task  # MediaPipe HandLandmarker neural network model asset
│
├── core/
│   ├── __init__.py           # Package exports
│   ├── camera.py             # Dedicated background camera thread (MSMF/DirectShow auto-selection)
│   ├── hand_tracker.py       # MediaPipe HandLandmarker wrapper, multi-hand support & HUD drawing
│   ├── gesture_recognizer.py # Feature extractor & 16-gesture debounced state machine
│   ├── action_controller.py  # Native Win32 ctypes cursor, clicks, drags, volume, media, windows
│   ├── cursor_controller.py  # Backward-compatible alias/subclass of ActionController
│   ├── coordinate_mapper.py  # Active rectangle boundary math, proportional mapping & HUD drawer
│   └── smoothing.py          # 1 € Filter adaptive velocity-based low-pass filter (1D & 2D)
│
├── ui/
│   ├── __init__.py           # Package exports
│   ├── main_window.py        # Central dashboard, telemetry cards, and worker thread orchestration
│   ├── camera_view.py        # High-DPI QPainter viewport rendering the mirrored video stream
│   ├── settings_panel.py     # Tabbed settings: "Controls & Tuning" & "Gesture Mapping" table
│   └── styles.py             # Sleek modern dark theme stylesheet
│
├── utils/
│   ├── __init__.py           # Package exports
│   └── performance_monitor.py # Rolling window FPS, average/P95 latency, CPU%, RAM, and cursor rate
│
└── tests/
    ├── test_action_controller.py
    ├── test_camera.py
    ├── test_coordinate_mapper.py
    ├── test_cursor_controller.py
    ├── test_gesture_recognizer.py
    ├── test_hand_tracker.py
    ├── test_performance_monitor.py
    ├── test_smoothing.py
    └── test_ui_and_pipeline.py
```

---

## Installation & Setup

### 1. Clone the Repository
```powershell
git clone https://github.com/shivdhiksh/Gestura.git
cd Gestura
```

### 2. Create Virtual Environment & Install Dependencies
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run All Unit & Integration Tests
```powershell
.\.venv\Scripts\python.exe -m pytest -v tests/
```

### 4. Launch GESTURA
```powershell
.\.venv\Scripts\python.exe main.py
```

---

## Keyboard Shortcuts
- **`[F8]`**: Global Emergency Pause / Resume Hotkey (system-wide hook; works even when GESTURA is minimized or running behind other windows).
