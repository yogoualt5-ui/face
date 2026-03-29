"""Preflight validation checks for runtime dependencies and setup."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run_check(name: str, fn: Callable[[], Tuple[bool, str]]) -> bool:
    ok, details = fn()
    status = "[PASS]" if ok else "[FAIL]"
    print(f"{status} {name}: {details}")
    return ok


def _check_required_paths() -> Tuple[bool, str]:
    required = [
        PROJECT_ROOT / "config",
        PROJECT_ROOT / "src",
        SRC_ROOT / "detectors",
        SRC_ROOT / "utils",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "README.md",
        SRC_ROOT / "main.py",
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
    if missing:
        return False, f"Missing: {', '.join(missing)}"
    return True, "Required folders/files are present"


def _check_dependency_imports() -> Tuple[bool, str]:
    failed = []
    for module_name in ("cv2", "mediapipe"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{module_name} ({exc})")
    if failed:
        return False, f"Import failures: {', '.join(failed)}"
    return True, "cv2 and mediapipe import successfully"


def _check_mediapipe_discovery() -> Tuple[bool, str]:
    try:
        mp = importlib.import_module("mediapipe")
        location = getattr(mp, "__file__", "unknown location")
    except Exception as exc:  # noqa: BLE001
        return False, f"Unable to import mediapipe ({exc})"
    return True, f"mediapipe discovered at {location}"


def _check_mediapipe_api_exposure() -> Tuple[bool, str]:
    try:
        mp = importlib.import_module("mediapipe")
    except Exception as exc:  # noqa: BLE001
        return False, f"mediapipe import failed ({exc})"

    has_solutions = hasattr(mp, "solutions")

    try:
        importlib.import_module("mediapipe.python.solutions")
        has_python_solutions = True
    except Exception:
        has_python_solutions = False

    if not has_solutions and not has_python_solutions:
        return False, "Neither mediapipe.solutions nor mediapipe.python.solutions is exposed"

    return True, (
        f"mediapipe.solutions={'yes' if has_solutions else 'no'}, "
        f"mediapipe.python.solutions={'yes' if has_python_solutions else 'no'}"
    )


def _check_face_and_hands_presence() -> Tuple[bool, str]:
    try:
        mp = importlib.import_module("mediapipe")
        solutions = getattr(mp, "solutions", None)
        if solutions is None:
            solutions = importlib.import_module("mediapipe.python.solutions")

        has_face = hasattr(solutions, "face_detection")
        has_hands = hasattr(solutions, "hands")
    except Exception as exc:  # noqa: BLE001
        return False, f"Unable to inspect APIs ({exc})"

    if not has_face or not has_hands:
        return False, f"face_detection={has_face}, hands={has_hands}"

    return True, "face_detection and hands APIs are available"


def _check_local_module_shadowing() -> Tuple[bool, str]:
    suspicious = []
    for name in ("cv2.py", "mediapipe.py"):
        if (PROJECT_ROOT / name).exists() or (SRC_ROOT / name).exists():
            suspicious.append(name)
    if suspicious:
        return False, f"Potential module shadowing: {', '.join(suspicious)}"
    return True, "No local cv2/mediapipe module shadowing detected"


def _check_camera_open_read() -> Tuple[bool, str]:
    try:
        import cv2

        from config import settings

        cap = cv2.VideoCapture(settings.CAMERA_ID)
        if not cap.isOpened():
            cap.release()
            return False, f"Camera {settings.CAMERA_ID} failed to open"
        ret, _ = cap.read()
        cap.release()
        if not ret:
            return False, "Camera opened but failed to read a frame"
        return True, "Camera opened and frame read succeeded"
    except Exception as exc:  # noqa: BLE001
        return False, f"Camera check failed ({exc})"


def _check_detector_imports() -> Tuple[bool, str]:
    failed = []
    for module_name in ("src.detectors.face_detector", "src.detectors.hand_detector"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{module_name} ({exc})")
    if failed:
        return False, f"Detector import failures: {', '.join(failed)}"
    return True, "Detector modules import successfully"


def run_startup_checks() -> bool:
    """Run all startup checks and return True only if all pass."""
    checks: List[Tuple[str, Callable[[], Tuple[bool, str]]]] = [
        ("Required project paths", _check_required_paths),
        ("Dependency imports", _check_dependency_imports),
        ("MediaPipe discovery/import", _check_mediapipe_discovery),
        ("MediaPipe API exposure", _check_mediapipe_api_exposure),
        ("MediaPipe face/hands availability", _check_face_and_hands_presence),
        ("Local module shadowing", _check_local_module_shadowing),
        ("Camera open/read", _check_camera_open_read),
        ("Detector module imports", _check_detector_imports),
    ]

    print("Running startup checks...\n")
    results = [_run_check(name, fn) for name, fn in checks]
    passed = all(results)

    print("\nStartup checks complete.")
    if passed:
        print("All checks passed. Starting realtime detection...")
    else:
        print("One or more checks failed. Fix issues before running detection.")
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if run_startup_checks() else 1)
