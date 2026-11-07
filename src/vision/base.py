"""The backend contract that keeps local and hosted models interchangeable."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import DetectionResult


class VisionBackend(ABC):
    """A model that can answer "is this object in the image?".

    Implementations may run locally (ONNX, llama.cpp server) or call a hosted
    OpenAI-compatible API.  The rest of the application only depends on this
    interface, so adding a provider never requires touching the UI.
    """

    key: str = "base"
    label: str = "Backend"
    local: bool = True

    @property
    def ready(self) -> bool:
        return True

    async def prepare(self) -> None:
        """Load weights, validate credentials, warm caches, etc."""

    @abstractmethod
    async def check(
        self,
        image_jpeg: bytes,
        target: str,
        *,
        threshold: float = 0.35,
    ) -> DetectionResult:
        """Return whether *target* is visible in *image_jpeg*."""

    async def close(self) -> None:
        """Release network clients or native resources."""