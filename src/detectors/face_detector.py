"""Face detection utilities powered by MediaPipe."""

from __future__ import annotations

from typing import List, Tuple

from config import settings
from src.detectors.mp_compat import get_solutions_namespace

_solutions = get_solutions_namespace()
_face_detection = _solutions.face_detection.FaceDetection(
    min_detection_confidence=settings.FACE_DETECTION_CONFIDENCE,
)


def detect_faces(rgb_frame) -> List[Tuple[int, int, int, int]]:
    """Detect faces and return absolute pixel bounding boxes.

    Returns:
        List of tuples in the form (xmin, ymin, xmax, ymax).
    """
    results = _face_detection.process(rgb_frame)
    if not results.detections:
        return []

    frame_height, frame_width = rgb_frame.shape[:2]
    boxes: List[Tuple[int, int, int, int]] = []

    for detection in results.detections:
        rel_box = detection.location_data.relative_bounding_box
        xmin = max(int(rel_box.xmin * frame_width), 0)
        ymin = max(int(rel_box.ymin * frame_height), 0)
        xmax = min(int((rel_box.xmin + rel_box.width) * frame_width), frame_width - 1)
        ymax = min(int((rel_box.ymin + rel_box.height) * frame_height), frame_height - 1)
        boxes.append((xmin, ymin, xmax, ymax))

    return boxes
