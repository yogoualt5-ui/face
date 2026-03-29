"""Simple FPS counter utility."""

from __future__ import annotations

import time
from collections import deque


class FPSCounter:
    """Compute instantaneous FPS with optional moving average."""

    def __init__(self, averaging_window: int = 10):
        self.prev_time = time.time()
        self.samples = deque(maxlen=max(1, averaging_window))

    def update(self) -> float:
        current_time = time.time()
        delta = current_time - self.prev_time
        self.prev_time = current_time

        fps = 0.0 if delta <= 0 else 1.0 / delta
        self.samples.append(fps)

        return sum(self.samples) / len(self.samples)
