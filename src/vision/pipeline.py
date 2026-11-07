"""Motion-gated recognition orchestration.

This module is deliberately free of Flet so the cooldown and warning rules
can be tested independently of the GUI.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from .base import VisionBackend
from .types import DetectionResult


class RecognitionPipeline:
    def __init__(
        self,
        backend: VisionBackend,
        *,
        threshold: float = 0.35,
        check_cooldown: float = 3.0,
        warn_cooldown: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend = backend
        self.threshold = threshold
        self.check_cooldown = check_cooldown
        self.warn_cooldown = warn_cooldown
        self._clock = clock
        self._lock = asyncio.Lock()
        self._last_check = float("-inf")
        self._last_warning = float("-inf")
        self.last_result: DetectionResult | None = None

    @property
    def running(self) -> bool:
        return self._lock.locked()

    def reset(self) -> None:
        self._last_check = float("-inf")
        self._last_warning = float("-inf")
        self.last_result = None

    async def on_motion(
        self,
        image_jpeg: bytes,
        target: str,
    ) -> DetectionResult | None:
        """Run one check if the cooldown allows it.

        Returns ``None`` when the check was skipped.  The lock prevents a
        slow remote model from stacking up requests while motion continues.
        """

        if self._lock.locked():
            return None
        now = self._clock()
        if now - self._last_check < self.check_cooldown:
            return None

        async with self._lock:
            self._last_check = now
            result = await self.backend.check(
                image_jpeg,
                target,
                threshold=self.threshold,
            )
            self.last_result = result
            return result

    def should_warn(self, result: DetectionResult) -> bool:
        if not result.present:
            return False
        now = self._clock()
        if now - self._last_warning < self.warn_cooldown:
            return False
        self._last_warning = now
        return True