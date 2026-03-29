"""Drawing helpers for OpenCV overlays."""

import cv2


def draw_face_bbox(frame, bbox, color, thickness=2):
    """Draw face bounding box in-place."""
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)


def draw_hand_bbox(frame, bbox, color, thickness=2):
    """Draw hand bounding box in-place."""
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)


def draw_hand_landmarks(frame, landmarks, color, radius=3):
    """Draw hand landmarks in-place."""
    for x, y in landmarks:
        cv2.circle(frame, (x, y), radius, color, -1)


def draw_text(frame, text, position, font_scale, color, thickness=2):
    """Draw text in-place."""
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
