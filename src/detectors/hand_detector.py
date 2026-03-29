"""Hand detection utilities powered by MediaPipe."""

from __future__ import annotations

from typing import Dict, List, Tuple

from config import settings
from src.detectors.mp_compat import get_solutions_namespace

_solutions = get_solutions_namespace()
_hands = _solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=settings.MAX_HANDS,
    min_detection_confidence=settings.HAND_DETECTION_CONFIDENCE,
    min_tracking_confidence=settings.HAND_TRACKING_CONFIDENCE,
)


def detect_hands(rgb_frame) -> List[Dict[str, object]]:
    """Detect hands and return bounding boxes plus landmarks.

    Returns:
        List of dicts with keys:
            - bbox: (xmin, ymin, xmax, ymax)
            - landmarks: list[(x, y)]
    """
    results = _hands.process(rgb_frame)
    if not results.multi_hand_landmarks:
        return []

    frame_height, frame_width = rgb_frame.shape[:2]
    detected_hands: List[Dict[str, object]] = []

    for hand_landmarks in results.multi_hand_landmarks:
        landmarks: List[Tuple[int, int]] = []
        xs: List[int] = []
        ys: List[int] = []

        for landmark in hand_landmarks.landmark:
            x = min(max(int(landmark.x * frame_width), 0), frame_width - 1)
            y = min(max(int(landmark.y * frame_height), 0), frame_height - 1)
            landmarks.append((x, y))
            xs.append(x)
            ys.append(y)

        bbox = (min(xs), min(ys), max(xs), max(ys))
        detected_hands.append({"bbox": bbox, "landmarks": landmarks})

    return detected_hands
