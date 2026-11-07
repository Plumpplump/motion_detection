"""Motion Sentinel — motion-triggered object recognition with Flet.

Pipeline
--------
1. A camera source produces frames:
   - Desktop: an OpenCV ``VideoCapture`` thread.
   - Web: the ``flet_camera`` control (streaming or snapshot polling).
2. ``detection.preprocess_frame`` + ``motion_ratio`` detect motion by comparing
   the current frame with one from the sensitivity interval.
3. Only after motion is confirmed, the current frame is sent to the selected
   vision backend:
   - ``embedded``: YOLOX-Tiny INT8 ONNX in ``assets/models`` (default, local).
   - ``openrouter`` / ``ollama`` / ``custom``: an OpenAI-compatible endpoint.
4. If the chosen object is present, the dashboard shows a red warning banner
   and optionally flashes and beeps.

The backend contract lives in ``vision/base.py``.  Adding a provider means
adding one adapter or one provider preset, not changing this file.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
if sys.platform == "darwin":
    # OpenCV's AVFoundation backend cannot request permission from a worker
    # thread.  Skip the broken implicit request; macOS still enforces the
    # permission granted to the terminal/IDE/app in System Settings.
    os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment issue
    raise SystemExit(
        "OpenCV is not installed in this Python environment.\n"
        "Run Motion Sentinel with the project environment:\n"
        "    uv run flet run --web\n\n"
        "A bare `flet run` uses whichever virtualenv is active and may not\n"
        "have the project dependencies (cv2, onnxruntime, ...)."
    ) from exc

import numpy as np
import flet as ft
from dotenv import load_dotenv

from detection import motion_ratio, preprocess_frame
from ui import (
    ACCENT,
    BACKGROUND,
    Dashboard,
)
from vision.factory import create_backend
from vision.image import annotate_jpeg, decode_jpeg, encode_jpeg
from vision.model_store import ensure_model
from vision.pipeline import RecognitionPipeline
from vision.providers import BackendConfig, config_from_env
from vision.types import VisionError

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

CAMERA_INDEX = 0
WINDOW_SECONDS = 5.0
SAMPLE_INTERVAL = 0.25
WEB_POLL_INTERVAL = 0.8
MOTION_RATIO_THRESHOLD = 0.008
DEBOUNCE_FRAMES = 3
CHECK_COOLDOWN = 3.0
WARN_COOLDOWN = 15.0
INTERVAL_MIN = 0.5
INTERVAL_MAX = 3.0
WARMUP_SECONDS = 2.0
ANALYSIS_MAX_SIDE = 768
PREVIEW_WIDTH = 320
PREVIEW_HEIGHT = 180


def use_cv2_camera(page: ft.Page) -> bool:
    """Desktop uses OpenCV; web and mobile use the Flet camera control."""

    if page.web:
        return False
    return page.platform not in {
        ft.PagePlatform.ANDROID,
        ft.PagePlatform.IOS,
    }


class MotionState:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.camera_on = False
        self.camera_error: str | None = None
        self.detecting = False

        # Frames
        self.latest_jpeg: bytes | None = None
        self.latest_frame: np.ndarray | None = None
        self.latest_gray: np.ndarray | None = None
        self._last_rendered: bytes | None = None
        self.history: deque[tuple[float, np.ndarray]] = deque()

        # Motion
        self.hits = 0
        self.last_alert = 0.0
        self.first_frame_time: float | None = None

        # Recognition
        self.backend = None
        self.backend_ready = False
        self.backend_error: str | None = None
        self.pipeline: RecognitionPipeline | None = None
        self.ai_loading = False
        self.checking = False
        self.last_check = 0.0

        # Web camera
        self.web_cam = None
        self.web_overlay = None
        self.web_cam_initialized = False
        self.frames_received = 0
        self.last_frame_at = 0.0
        self.polling_started = False

        self.camera_stop = threading.Event()
        self.camera_generation = 0
        self.capture_lock = asyncio.Lock()
        self.loops_started = False


def beep() -> None:
    """Best-effort alert sound across desktop platforms."""

    try:
        if sys.platform.startswith("win"):
            import winsound

            for _ in range(2):
                winsound.Beep(1000, 180)
                time.sleep(0.05)
        elif sys.platform == "darwin":
            subprocess.run(
                ["afplay", "/System/Library/Sounds/Glass.aiff"],
                check=False,
                timeout=3,
            )
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


def camera_error_message() -> str:
    if sys.platform == "darwin":
        return (
            "Camera access denied. Open System Settings → Privacy & Security → "
            "Camera, enable the app running this project (Terminal, iTerm, "
            "VS Code or Python), then restart the app."
        )
    if sys.platform.startswith("win"):
        return (
            "Could not open the camera. Check Windows camera privacy settings "
            "and close other apps that may be using it."
        )
    return "Could not open the camera. Check that another app is not using it."


def open_camera():
    """Open the first working camera, preferring the native macOS backend."""

    indices = [CAMERA_INDEX] + [i for i in (0, 1, 2) if i != CAMERA_INDEX]
    last = None
    for index in indices:
        if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
        last = cap
        cap.release()
    return last


def capture_loop(state: MotionState, generation: int) -> None:
    cap = open_camera()
    try:
        if cap is None or not cap.isOpened():
            state.camera_error = camera_error_message()
            print(f"Camera error: {state.camera_error}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        state.camera_on = True
        failed_reads = 0

        while (
            not state.stop.is_set()
            and not state.camera_stop.is_set()
            and state.camera_generation == generation
        ):
            ok, frame = cap.read()
            if not ok:
                failed_reads += 1
                if failed_reads >= 10:
                    state.camera_error = (
                        "Camera opened but no frames arrived. "
                        + camera_error_message()
                    )
                    break
                time.sleep(0.1)
                continue

            failed_reads = 0
            state.latest_frame = frame
            state.latest_gray = preprocess_frame(frame)
            state.frames_received += 1
            state.last_frame_at = time.monotonic()

            small = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
            ok_jpeg, buffer = cv2.imencode(
                ".jpg",
                small,
                [int(cv2.IMWRITE_JPEG_QUALITY), 70],
            )
            if ok_jpeg:
                state.latest_jpeg = buffer.tobytes()
            time.sleep(0.03)
    finally:
        if cap is not None:
            cap.release()
        if state.camera_generation == generation:
            state.camera_on = False


def ingest_web_frame(state: MotionState, raw_bytes: bytes) -> None:
    try:
        frame = decode_jpeg(raw_bytes)
        state.latest_frame = frame
        state.latest_gray = preprocess_frame(frame)
        state.latest_jpeg = raw_bytes
        state.frames_received += 1
        state.last_frame_at = time.monotonic()
    except Exception:
        pass


async def detection_loop(
    page: ft.Page,
    state: MotionState,
    dashboard: Dashboard,
) -> None:
    while not state.stop.is_set():
        now = time.monotonic()
        gray = state.latest_gray

        if gray is not None:
            # Only append when a new frame arrived; the sampler runs faster
            # than a web snapshot loop and would otherwise store duplicates.
            if not state.history or state.history[-1][1] is not gray:
                state.history.append((now, gray.copy()))
            cutoff = now - WINDOW_SECONDS
            while state.history and state.history[0][0] < cutoff:
                state.history.popleft()

        if state.detecting and gray is not None:
            if state.first_frame_time is None:
                state.first_frame_time = now
            if now - state.first_frame_time < WARMUP_SECONDS:
                await asyncio.sleep(SAMPLE_INTERVAL)
                continue

            try:
                target_time = now - dashboard.get_interval()
                target = min(
                    state.history,
                    key=lambda entry: abs(entry[0] - target_time),
                )
                if gray.shape != target[1].shape:
                    state.hits = 0
                    await asyncio.sleep(SAMPLE_INTERVAL)
                    continue

                ratio = motion_ratio(gray, target[1])
                if ratio > MOTION_RATIO_THRESHOLD:
                    state.hits += 1
                else:
                    state.hits = 0

                if (
                    state.hits >= DEBOUNCE_FRAMES
                    and (now - state.last_alert) >= CHECK_COOLDOWN
                    and not state.checking
                ):
                    state.last_alert = now
                    state.last_check = now
                    state.hits = 0
                    asyncio.create_task(
                        trigger_check(page, dashboard, state)
                    )
            except Exception:
                state.hits = 0

        await asyncio.sleep(SAMPLE_INTERVAL)


async def display_loop(
    page: ft.Page,
    state: MotionState,
    raw: ft.RawImage,
) -> None:
    while not state.stop.is_set():
        jpeg = state.latest_jpeg
        if jpeg is not None and jpeg is not state._last_rendered:
            state._last_rendered = jpeg
            try:
                await raw.render_encoded(jpeg)
            except Exception:
                pass
        await asyncio.sleep(0.05)


async def web_poll_camera(
    state: MotionState,
    web_cam,
    ingest,
    dashboard: Dashboard | None = None,
) -> None:
    failures = 0
    while not state.stop.is_set() and state.camera_on:
        try:
            # Never overlap with the snapshot taken by a motion check.
            async with state.capture_lock:
                data = await web_cam.take_picture()
            if data:
                ingest(data)
                failures = 0
        except Exception:
            failures += 1
            if failures >= 5:
                if dashboard is not None:
                    dashboard.set_status(
                        "Live preview is on, but this browser cannot provide "
                        "detection frames. Try Chrome or Edge, or switch "
                        "to the desktop app.",
                        "warn",
                    )
                return
            await asyncio.sleep(0.5)
            continue
        # Polling more slowly than the motion sampler keeps the preview alive.
        await asyncio.sleep(WEB_POLL_INTERVAL)


async def watch_stream_feed(
    page: ft.Page,
    state: MotionState,
    dashboard: Dashboard,
    web_cam,
    ingest,
    timeout: float = 4.0,
) -> None:
    """Fall back to snapshots if image streaming produces no frames."""

    await asyncio.sleep(timeout)
    if not state.camera_on or state.frames_received > 0 or state.polling_started:
        return
    state.polling_started = True
    dashboard.set_status(
        "Camera preview is live — using snapshots for motion detection",
        "warn",
    )
    page.run_task(
        web_poll_camera,
        state,
        web_cam,
        ingest,
        dashboard,
    )


async def snapshot_jpeg(state: MotionState) -> bytes | None:
    """Take the frame that triggered motion without disturbing the preview.

    A fresh frame from the live feed is preferred.  ``take_picture()`` is only
    used when the feed is stale, and it is serialised with the polling loop.
    """

    frame = state.latest_frame
    if frame is not None and (time.monotonic() - state.last_frame_at) < 1.0:
        return await asyncio.to_thread(
            encode_jpeg,
            frame.copy(),
            ANALYSIS_MAX_SIDE,
            85,
        )

    if state.web_cam is not None and state.camera_on:
        try:
            async with state.capture_lock:
                data = await state.web_cam.take_picture()
            if data:
                frame = await asyncio.to_thread(decode_jpeg, data)
                return await asyncio.to_thread(
                    encode_jpeg,
                    frame,
                    ANALYSIS_MAX_SIDE,
                    85,
                )
        except Exception:
            pass

    if frame is None:
        return None
    return await asyncio.to_thread(
        encode_jpeg,
        frame.copy(),
        ANALYSIS_MAX_SIDE,
        85,
    )


async def trigger_check(
    page: ft.Page,
    dashboard: Dashboard,
    state: MotionState,
) -> None:
    if state.checking:
        return

    target = dashboard.get_target()
    if target is None:
        dashboard.set_status("Choose an object to watch first", "warn")
        return

    if state.backend is None or not state.backend_ready:
        message = state.backend_error or "AI model is still loading"
        dashboard.set_status(f"Motion detected — {message}", "warn")
        return

    if state.backend.key == "embedded" and not target.supported_locally:
        dashboard.set_status(
            "The local model only knows COCO objects. "
            "Switch to OpenRouter for this custom object.",
            "warn",
        )
        return

    state.checking = True
    dashboard.set_checking(True)
    dashboard.set_status(
        f"Motion detected — checking for {target.label.lower()}…",
        "info",
    )

    try:
        image = await snapshot_jpeg(state)
        if image is None:
            dashboard.set_status("Camera frame unavailable", "error")
            return

        # Motion is confirmed: flash the camera container while the model runs.
        if dashboard.alert_enabled():
            asyncio.create_task(dashboard.flash_camera())

        pipeline = state.pipeline
        if pipeline is None:
            dashboard.set_status("AI pipeline is not ready", "error")
            return

        pipeline.threshold = dashboard.get_threshold()
        target_text = (
            target.model_label
            if state.backend.key == "embedded"
            else target.label
        )
        result = await pipeline.on_motion(image, target_text)
        if result is None:
            return

        if result.present:
            if pipeline.should_warn(result):
                thumbnail = (
                    await asyncio.to_thread(annotate_jpeg, image, result.boxes)
                    if result.boxes
                    else image
                )
                display_result = replace(result, label=target.label)
                dashboard.show_warning(display_result, thumbnail)
                dashboard.set_status(f"{target.label} detected", "warn")
                if dashboard.alert_enabled():
                    asyncio.create_task(asyncio.to_thread(beep))
            else:
                dashboard.set_status(f"{target.label} still visible", "info")
        else:
            dashboard.set_status(
                f"Motion detected — no {target.label.lower()} found",
                "ok",
            )
    except VisionError as exc:
        dashboard.set_status(exc.message, "error")
    except Exception as exc:
        dashboard.set_status(f"Detection error: {exc}", "error")
    finally:
        state.checking = False
        dashboard.set_checking(False)
        try:
            page.update()
        except Exception:
            pass


def _backend_name(backend) -> str:
    if backend is None:
        return "AI"
    if backend.key == "embedded":
        return "Local AI"
    return backend.label


async def load_backend(
    page: ft.Page,
    dashboard: Dashboard,
    state: MotionState,
    config: BackendConfig,
) -> None:
    state.ai_loading = True
    preparing = (
        "Preparing local model…"
        if config.provider == "embedded"
        else f"Connecting to {config.preset.label}…"
    )
    dashboard.set_ai_status(preparing, "info")

    try:
        if config.provider == "embedded":
            await asyncio.to_thread(ensure_model)
        backend = create_backend(config)
        await backend.prepare()

        previous = state.backend
        state.backend = backend
        state.backend_ready = True
        state.backend_error = None
        state.pipeline = RecognitionPipeline(
            backend,
            threshold=config.threshold,
            check_cooldown=CHECK_COOLDOWN,
            warn_cooldown=WARN_COOLDOWN,
        )
        if previous is not None and previous is not backend:
            try:
                await previous.close()
            except Exception:
                pass

        name = _backend_name(backend)
        detail = "on-device" if backend.local else "remote"
        dashboard.set_ai_status(f"{name} ready · {detail}", "ok" if backend.local else "cloud")
        if state.camera_on:
            dashboard.set_start_detection_enabled(True)
    except VisionError as exc:
        state.backend_error = exc.message
        if state.backend is None:
            state.backend_ready = False
        dashboard.set_ai_status(exc.message, "error")
        dashboard.set_status("Local AI unavailable — see AI settings", "error")
    except Exception as exc:
        state.backend_error = str(exc)
        if state.backend is None:
            state.backend_ready = False
        dashboard.set_ai_status(f"Could not start AI: {exc}", "error")
        dashboard.set_status("Local AI unavailable — see AI settings", "error")
    finally:
        state.ai_loading = False


async def main(page: ft.Page) -> None:
    camera_mode = "OpenCV" if use_cv2_camera(page) else "flet_camera"
    print(
        f"Motion Sentinel | Python: {sys.executable} | camera: {camera_mode}",
        flush=True,
    )

    project_python = (
        Path(__file__).resolve().parent.parent
        / ".venv"
        / ("Scripts" if os.name == "nt" else "bin")
        / ("python.exe" if os.name == "nt" else "python")
    )
    if project_python.exists() and Path(sys.executable).resolve() != project_python.resolve():
        print(
            "WARNING: this is not the project virtualenv.\n"
            f"         project env: {project_python}\n"
            "         start with:  ./run_web.sh   (or uv run flet run --web)",
            flush=True,
        )

    page.title = "Motion Sentinel"
    page.bgcolor = BACKGROUND
    page.theme_mode = ft.ThemeMode.LIGHT
    page.theme = ft.Theme(color_scheme_seed=ACCENT, use_material3=True)
    if use_cv2_camera(page):
        try:
            page.window.width = 1180
            page.window.height = 860
            page.window.min_width = 980
            page.window.min_height = 700
        except Exception:
            pass

    state = MotionState()

    async def start_detection(_event=None) -> None:
        if not state.camera_on:
            dashboard.set_status("Start the camera first", "warn")
            return
        state.detecting = True
        state.first_frame_time = None
        state.hits = 0
        state.last_alert = 0.0
        if state.pipeline is not None:
            state.pipeline.reset()
        dashboard.hide_warning()
        dashboard.set_monitoring(True)
        if state.backend_ready:
            dashboard.set_status(
                "Monitoring — recognising after motion",
                "info",
            )
        else:
            dashboard.set_status(
                "Monitoring motion — AI is not ready yet. "
                "Check AI settings.",
                "warn",
            )

    async def stop_detection(_event=None) -> None:
        state.detecting = False
        state.hits = 0
        state.first_frame_time = None
        dashboard.set_monitoring(False)
        dashboard.hide_warning()
        dashboard.set_status(
            "Monitoring paused — camera still on" if state.camera_on else "Camera off",
            "idle",
        )

    async def toggle_detection(_event=None) -> None:
        if state.detecting:
            await stop_detection()
        else:
            await start_detection()

    async def stop_camera(_event=None) -> None:
        if not state.camera_on:
            return

        state.detecting = False
        state.hits = 0
        state.first_frame_time = None
        state.camera_on = False
        state.frames_received = 0
        state.polling_started = False
        state.latest_frame = None
        state.latest_gray = None
        state.latest_jpeg = None
        dashboard.set_monitoring(False)
        dashboard.hide_warning()
        dashboard.set_status("Stopping camera…", "info")

        if use_cv2_camera(page):
            state.camera_generation += 1
            state.camera_stop.set()
            # Let the old capture thread leave cap.read() and release the device.
            await asyncio.sleep(0.3)
            state.camera_stop.clear()
        else:
            try:
                await state.web_cam.stop_image_stream()
            except Exception:
                pass
            try:
                await state.web_cam.pause_preview()
            except Exception:
                pass
            try:
                state.web_cam.preview_enabled = False
                state.web_cam.update()
            except Exception:
                pass

        dashboard.set_camera_started(False)
        dashboard.set_status("Camera off", "idle")

    async def start_camera(_event=None) -> None:
        if state.camera_on:
            return
        state.stop.clear()
        state.camera_error = None
        dashboard.start_camera_btn.disabled = True
        dashboard.set_status("Starting camera…", "info")

        # Desktop uses OpenCV; web and native mobile use flet_camera.
        if use_cv2_camera(page):
            state.camera_stop.clear()
            state.camera_generation += 1
            generation = state.camera_generation
            state.camera_on = True
            threading.Thread(
                target=capture_loop,
                args=(state, generation),
                daemon=True,
            ).start()
            if not state.loops_started:
                state.loops_started = True
                page.run_task(detection_loop, page, state, dashboard)
                page.run_task(
                    display_loop,
                    page,
                    state,
                    dashboard.preview_img,
                )

            for _ in range(100):
                if state.latest_jpeg is not None or state.camera_error:
                    break
                await asyncio.sleep(0.05)

            if state.camera_error:
                state.camera_on = False
                dashboard.set_status(state.camera_error, "error")
                dashboard.set_camera_started(False)
            else:
                dashboard.set_camera_started(True)
                dashboard.set_status(
                    "Camera on — watching for motion",
                    "ok",
                )
                dashboard.set_start_detection_enabled(True)
            return

        import flet_camera as fc

        resumed = False
        if state.web_cam_initialized:
            try:
                state.web_cam.preview_enabled = True
                state.web_cam.update()
                await state.web_cam.resume_preview()
                resumed = True
            except Exception:
                resumed = False

        if not resumed:
            try:
                cameras = await state.web_cam.get_available_cameras()
            except Exception:
                cameras = None
                state.camera_error = (
                    "Camera timeout. Allow camera access in the browser, "
                    "then press Start Camera again."
                )
            if not cameras:
                state.camera_error = state.camera_error or "No camera found"
                dashboard.set_status(state.camera_error, "error")
                dashboard.start_camera_btn.disabled = False
                return

            try:
                await state.web_cam.initialize(
                    description=cameras[0],
                    resolution_preset=fc.ResolutionPreset.MEDIUM,
                    enable_audio=False,
                    image_format_group=fc.ImageFormatGroup.JPEG,
                )
                state.web_cam_initialized = True
            except Exception as exc:
                state.camera_on = False
                dashboard.set_status(
                    f"Camera initialization failed: {exc}",
                    "error",
                )
                dashboard.start_camera_btn.disabled = False
                return

        state.camera_on = True
        state.camera_error = None
        state.frames_received = 0
        state.last_frame_at = 0.0
        if state.web_overlay is not None:
            state.web_overlay.visible = False
        dashboard.set_camera_started(True)

        if not state.loops_started:
            state.loops_started = True
            page.run_task(detection_loop, page, state, dashboard)

        # Give the preview a moment to start before asking for frames.
        await asyncio.sleep(0.5)

        try:
            supports_streaming = await state.web_cam.supports_image_streaming()
            stream_started = False
            if supports_streaming:
                try:
                    await state.web_cam.start_image_stream()
                    stream_started = True
                except Exception:
                    stream_started = False

            if stream_started:
                page.run_task(
                    watch_stream_feed,
                    page,
                    state,
                    dashboard,
                    state.web_cam,
                    lambda data: ingest_web_frame(state, data),
                )
                dashboard.set_status(
                    "Camera streaming — watching for motion",
                    "ok",
                )
            else:
                state.polling_started = True
                page.run_task(
                    web_poll_camera,
                    state,
                    state.web_cam,
                    lambda data: ingest_web_frame(state, data),
                    dashboard,
                )
                dashboard.set_status(
                    "Camera on — snapshot mode",
                    "ok",
                )
            dashboard.set_start_detection_enabled(True)
        except Exception as exc:
            state.camera_on = False
            dashboard.set_status(
                f"Camera stream failed: {exc}",
                "error",
            )
            dashboard.set_camera_started(False)

    async def toggle_camera(_event=None) -> None:
        if state.camera_on:
            await stop_camera()
        else:
            await start_camera()

    async def apply_settings(_event=None) -> None:
        config = dashboard.get_config()
        await load_backend(page, dashboard, state, config)
        if state.backend_ready and state.pipeline is not None:
            state.pipeline.reset()
        if state.backend_error and state.backend is not None:
            dashboard.set_status(
                "Backend switch failed — still using the previous model",
                "warn",
            )

    async def dismiss_warning(_event=None) -> None:
        dashboard.hide_warning()
        if state.detecting:
            dashboard.set_status("Monitoring for motion…", "info")
        elif state.camera_on:
            dashboard.set_status("Camera on — watching for motion", "ok")

    dashboard = Dashboard(
        page,
        on_toggle_camera=toggle_camera,
        on_toggle_detection=toggle_detection,
        on_apply_settings=apply_settings,
        on_dismiss_warning=dismiss_warning,
    )

    if not use_cv2_camera(page):
        import flet_camera as fc

        def on_stream_image(event) -> None:
            if getattr(event, "bytes", None):
                ingest_web_frame(state, event.bytes)

        state.web_overlay = ft.Container(
            alignment=ft.Alignment.CENTER,
            content=ft.Column(
                [
                    ft.Icon(
                        ft.Icons.CENTER_FOCUS_STRONG,
                        color=ft.Colors.WHITE_70,
                        size=42,
                    ),
                    ft.Text(
                        "Waiting for camera",
                        color=ft.Colors.WHITE_70,
                        size=13,
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=8,
            ),
        )
        state.web_cam = fc.Camera(
            preview_enabled=True,
            content=state.web_overlay,
            on_stream_image=on_stream_image,
            expand=True,
        )
        dashboard.camera_surface.content = state.web_cam

    page.add(dashboard.build())
    dashboard.apply_config(config_from_env())

    def shutdown(_event=None) -> None:
        state.stop.set()
        if state.backend is not None:
            try:
                asyncio.create_task(state.backend.close())
            except Exception:
                pass

    page.on_disconnect = shutdown
    page.run_task(
        load_backend,
        page,
        dashboard,
        state,
        dashboard.get_config(),
    )


if __name__ == "__main__":
    ft.run(main)