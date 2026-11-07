import flet.testing as ftt

import detection


async def test_ui_initial_state(flet_app: ftt.FletTestApp):
    """The motion detector UI renders with camera off."""
    tester = flet_app.tester
    await tester.pump_and_settle()

    assert (await tester.find_by_key("start_camera")).count == 1
    assert (await tester.find_by_key("start_detection")).count == 1
    assert (await tester.find_by_key("camera_box")).count == 1
    assert (await tester.find_by_key("interval_slider")).count == 1
    assert (await tester.find_by_key("object_select")).count == 1
    assert (await tester.find_by_key("provider_select")).count == 1
    assert (await tester.find_by_key("ai_status")).count == 1
    assert (await tester.find_by_text("Camera off")).count == 1


async def test_detection_button_disabled_until_camera_starts(flet_app: ftt.FletTestApp):
    """Start Detection starts disabled and is not enabled without a camera."""
    tester = flet_app.tester
    await tester.pump_and_settle()

    btn = await tester.find_by_key("start_detection")
    assert btn.count == 1
    assert btn.control.disabled is True


async def test_motion_ratio_identical_frames_is_zero():
    frame = detection.preprocess_frame(__black_frame())
    assert detection.motion_ratio(frame, frame) == 0.0


async def test_motion_ratio_different_frames_is_high():
    black = __black_frame()
    white = black.copy()
    white[:] = (255, 255, 255)
    frame = detection.preprocess_frame(black)
    other = detection.preprocess_frame(white)
    assert detection.motion_ratio(frame, other) > 0.9


def __black_frame():
    import numpy as np

    return np.zeros((480, 640, 3), dtype=np.uint8)