"""Deterministic backend used by tests and UI demos."""

from __future__ import annotations

from .base import VisionBackend
from .types import BBox, DetectionResult


class FakeBackend(VisionBackend):
    key = "fake"
    label = "Fake backend"
    local = True

    def __init__(
        self,
        *,
        present: bool = True,
        confidence: float = 0.91,
        reason: str = "Fake detection",
        boxes: tuple[BBox, ...] = (),
    ) -> None:
        self.present = present
        self.confidence = confidence
        self.reason = reason
        self.boxes = boxes
        self.calls: list[tuple[bytes, str, float]] = []

    async def prepare(self) -> None:
        return None

    async def check(
        self,
        image_jpeg: bytes,
        target: str,
        *,
        threshold: float = 0.35,
    ) -> DetectionResult:
        self.calls.append((image_jpeg, target, threshold))
        return DetectionResult(
            present=self.present,
            label=target,
            confidence=self.confidence,
            reason=self.reason,
            boxes=self.boxes,
            backend=self.key,
            target=target,
        )