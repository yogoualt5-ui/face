"""Compatibility helpers for locating MediaPipe solution APIs."""

from __future__ import annotations

import importlib


class MediaPipeSolutionsNotFound(RuntimeError):
    """Raised when MediaPipe face/hand solution APIs cannot be located."""


def get_solutions_namespace():
    """Return a namespace exposing `face_detection` and `hands` modules.

    Supports environments where `mediapipe.solutions` is available as:
    - an attribute on the top-level `mediapipe` package, or
    - an importable submodule (`mediapipe.solutions`), or
    - legacy path (`mediapipe.python.solutions`).
    """
    mp = importlib.import_module("mediapipe")

    # Common path: attribute on top-level package.
    solutions = getattr(mp, "solutions", None)
    if solutions is not None and hasattr(solutions, "face_detection") and hasattr(solutions, "hands"):
        return solutions

    # Some builds expose it only as an importable module.
    try:
        solutions = importlib.import_module("mediapipe.solutions")
        if hasattr(solutions, "face_detection") and hasattr(solutions, "hands"):
            return solutions
    except Exception:  # noqa: BLE001
        pass

    # Legacy fallback path.
    try:
        solutions = importlib.import_module("mediapipe.python.solutions")
        if hasattr(solutions, "face_detection") and hasattr(solutions, "hands"):
            return solutions
    except Exception:  # noqa: BLE001
        pass

    raise MediaPipeSolutionsNotFound(
        "Could not locate MediaPipe Solutions APIs (face_detection/hands). "
        "Use Python 3.9-3.12 with an official mediapipe wheel, then reinstall dependencies."
    )
