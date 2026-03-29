# Real-Time Face & Hand Detection

A modular Python project for real-time webcam-based face and hand detection using **MediaPipe** and **OpenCV**.

## Project Structure

```text
real_time_face_hand_detection/
├── config/
│   └── settings.py
├── src/
│   ├── detectors/
│   │   ├── face_detector.py
│   │   └── hand_detector.py
│   ├── utils/
│   │   ├── drawing.py
│   │   └── fps.py
│   ├── main.py
│   └── startup_checker.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.7+
- A working webcam

## Setup

1. Clone or download this project.
2. (Recommended) Create and activate a virtual environment:
   - `python -m venv venv`
   - Activate:
     - macOS/Linux: `source venv/bin/activate`
     - Windows (PowerShell): `venv\Scripts\Activate.ps1`
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Adjust runtime settings in `config/settings.py` if needed:
   - `CAMERA_ID`
   - frame size
   - confidence thresholds
   - detector toggles

## Run

```bash
python src/main.py
```

`main.py` now performs preflight validation via `startup_checker.py` before detection starts. The app only enters the realtime loop if every startup check passes.

## Controls

- Press `ESC` to exit.

## Troubleshooting

- If the camera does not open, try a different `CAMERA_ID` in `config/settings.py`.
- If detection is slow, reduce `FRAME_WIDTH` / `FRAME_HEIGHT`.
- You can disable one detector (`USE_FACE_DETECTION` or `USE_HAND_DETECTION`) to improve performance.
- If startup says MediaPipe Solutions APIs are unavailable, install a compatible Python version (3.11 is recommended) and reinstall dependencies.
