"""Small, dependency-free value types shared by every vision backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """A detection rectangle in original-frame pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0
    label: str = ""

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class DetectionResult:
    """Normalised answer from any backend, local or remote."""

    present: bool
    label: str
    confidence: float = 0.0
    reason: str = ""
    boxes: tuple[BBox, ...] = ()
    raw: str = ""
    latency_ms: float = 0.0
    backend: str = ""
    target: str = ""


@dataclass(frozen=True)
class BackendInfo:
    key: str
    label: str
    local: bool
    detail: str = ""


class VisionError(RuntimeError):
    """An error that can be rendered directly in the GUI."""

    def __init__(
        self,
        message: str,
        *,
        kind: str = "error",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.status_code = status_code