"""Integration tests for the motion -> recognition -> warning path.

These do not need a Flutter host: they use the real Dashboard controls with a
tiny fake page object, so the GUI state transitions are still exercised.
"""

from __future__ import annotations

import numpy as np

import main
from ui import Dashboard
from vision.fake import FakeBackend
from vision.pipeline import RecognitionPipeline
from vision.types import BBox


class FakePage:
    def __init__(self) -> None:
        self.bgcolor = "#FFFFFF"
        self.updates = 0

    def update(self) -> None:
        self.updates += 1


def _dashboard(page: FakePage) -> Dashboard:
    return Dashboard(
        page,
        on_toggle_camera=lambda _e=None: None,
        on_toggle_detection=lambda _e=None: None,
        on_apply_settings=lambda _e=None: None,
        on_dismiss_warning=lambda _e=None: None,
    )


def _state(present: bool) -> main.MotionState:
    state = main.MotionState()
    state.backend = FakeBackend(
        present=present,
        confidence=0.91,
        boxes=(
            BBox(10, 20, 120, 160, confidence=0.91, label="cell phone"),
        )
        if present
        else (),
    )
    state.backend.key = "fake"
    state.backend_ready = True
    state.pipeline = RecognitionPipeline(
        state.backend,
        threshold=0.35,
        check_cooldown=0.0,
        warn_cooldown=0.0,
    )
    state.latest_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    return state


async def test_trigger_check_shows_warning():
    page = FakePage()
    dashboard = _dashboard(page)
    dashboard.alert_switch.value = False
    state = _state(present=True)

    await main.trigger_check(page, dashboard, state)

    assert dashboard.warning_card.visible is True
    assert "SMARTPHONE" in dashboard.warning_title.value
    assert "91%" in dashboard.warning_sub.value
    assert dashboard.status.value == "Smartphone detected"
    assert state.checking is False


async def test_trigger_check_without_object_is_quiet():
    page = FakePage()
    dashboard = _dashboard(page)
    dashboard.alert_switch.value = False
    state = _state(present=False)

    await main.trigger_check(page, dashboard, state)

    assert dashboard.warning_card.visible is False
    assert "no smartphone found" in dashboard.status.value.lower()
    assert state.checking is False


async def test_embedded_backend_rejects_custom_object():
    page = FakePage()
    dashboard = _dashboard(page)
    dashboard.object_dropdown.value = "custom"
    dashboard.custom_field.value = "wallet"
    state = _state(present=True)
    state.backend.key = "embedded"

    await main.trigger_check(page, dashboard, state)

    assert dashboard.warning_card.visible is False
    assert "local model only knows COCO" in dashboard.status.value