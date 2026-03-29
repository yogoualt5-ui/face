"""Realtime face and hand detection entry point."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.startup_checker import run_startup_checks


def main() -> int:
    if not run_startup_checks():
        return 1

    import cv2

    from config import settings
    from src.detectors.face_detector import detect_faces
    from src.detectors.hand_detector import detect_hands
    from src.utils.drawing import (
        draw_face_bbox,
        draw_hand_bbox,
        draw_hand_landmarks,
        draw_text,
    )
    from src.utils.fps import FPSCounter

    cap = cv2.VideoCapture(settings.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"Error: Unable to open camera with CAMERA_ID={settings.CAMERA_ID}")
        return 1

    fps_counter = FPSCounter(averaging_window=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame. Exiting loop.")
            break

        frame = cv2.resize(frame, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = detect_faces(rgb_frame) if settings.USE_FACE_DETECTION else []
        hands = detect_hands(rgb_frame) if settings.USE_HAND_DETECTION else []

        for bbox in faces:
            draw_face_bbox(frame, bbox, settings.DRAW_BBOX_COLOR)

        for hand in hands:
            draw_hand_bbox(frame, hand["bbox"], settings.DRAW_BBOX_COLOR)
            draw_hand_landmarks(frame, hand["landmarks"], settings.DRAW_LANDMARK_COLOR)

        fps = fps_counter.update()
        draw_text(frame, f"FPS: {fps:.1f}", (10, 30), 0.8, (0, 255, 255))
        draw_text(frame, f"Faces: {len(faces)} Hands: {len(hands)}", (10, 60), 0.7, (255, 255, 255))

        cv2.imshow("Face & Hand Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == settings.ESC_KEY:
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
