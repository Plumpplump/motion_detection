"""Motion Detector — Flet UI + platform camera pipelines.

Architecture
------------
src/
  main.py     This file. Flet app: UI, camera/platform pipelines, alert logic.
  detection.py Pure helpers (no Flet): preprocess_frame(), motion_ratio().
tests/
  test_main.py  UI smoke tests (injected `flet_app` fixture) + detection unit tests.

Third-party Python libraries
-----------------------------
- flet (>=0.86.5)          UI framework.
- flet-camera (>=0.86.5)   `fc.Camera` control — web/mobile camera preview + frames.
- opencv-python (>=5.0)    `cv2` — Webcam capture, frame math, JPEG encode.
- numpy                    Array handling under OpenCV.
- stdlib only elsewhere    threading, asyncio, time, collections, winsound (stdlib),
                           os, pathlib — no other packages needed.

Data flow (all platforms share one pipeline)
--------------------------------------------
1. A camera source produces frames:
   - Windows desktop: `capture_loop` thread reads cv2.VideoCapture(0).
   - Web: the flet_camera `Camera` control; frames arrive via `on_stream_image`
     (or a `web_poll_camera` take_picture() fallback when streaming is unsupported).
2. Frames are downscaled/grayscaled by detection.preprocess_frame() into a small
   numpy array stored as state.latest_gray.
3. `detection_loop` keeps a rolling history deque of (timestamp, gray frame) for
   the last WINDOW_SECONDS. When detecting, it compares latest_gray against the
   history frame closest to `now - slider.value` (the sensitivity interval),
   via detection.motion_ratio(). 3 consecutive hits trigger an alert.
4. Alert: winsound beep (background task) + red/white page.bgcolor flash.
5. Desktop live view: display_loop pushes JPEGs to an ft.RawImage via
   render_encoded(). Web uses the Camera control's own preview.

Why the camera was black / how it was solved
--------------------------------------------
- OpenCV 5 + Windows MSMF: frames grab but come out black unless the env var
  OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0 is set before `import cv2`.
- Desktop preview: ft.Image.src + raw bytes did not render reliably on Windows;
  switching to ft.RawImage.render_encoded() (dedicated streaming control) fixed it.
- Web timeout on get_available_cameras(): the Camera control was created with
  visible=False, so Flet never built its web widget -> no invoke-method listener,
  -> 10s TimeoutException. Fix: mount the Camera control visible from the start
  and overlay a "Camera off" placeholder via its `content` argument (docs pattern).
  If image streaming is unsupported, fall back to polling web_cam.take_picture().
"""

import asyncio
import os
import threading
import time
from collections import deque

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np
import flet as ft

from detection import preprocess_frame, motion_ratio

CAMERA_INDEX = 0
WINDOW_SECONDS = 5.0
SAMPLE_INTERVAL = 0.25
MOTION_RATIO_THRESHOLD = 0.008
DEBOUNCE_FRAMES = 3
ALERT_COOLDOWN = 5.0
INTERVAL_MIN = 0.5
INTERVAL_MAX = 3.0
WARMUP_SECONDS = 2.0
UI_WIDTH = 390


class MotionState:
    def __init__(self):
        self.stop = threading.Event()
        self.camera_on = False
        self.camera_error = None
        self.detecting = False
        self.flashing = False
        self.latest_jpeg: bytes | None = None
        self.latest_gray: np.ndarray | None = None
        self._last_rendered: bytes | None = None
        self.history: deque[tuple[float, np.ndarray]] = deque()
        self.hits = 0
        self.last_alert = 0.0
        self.first_frame_time: float | None = None


def beep() -> None:
    try:
        import winsound

        for _ in range(3):
            winsound.Beep(1000, 200)
            time.sleep(0.05)
    except Exception:
        pass


def capture_loop(state: MotionState) -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        state.camera_error = "Could not open camera"
        return
    state.camera_on = True
    failed_reads = 0
    while not state.stop.is_set():
        ok, frame = cap.read()
        if not ok:
            failed_reads += 1
            if failed_reads >= 10:
                state.camera_error = "Camera opened but no frames arrived"
                break
            time.sleep(0.1)
            continue
        failed_reads = 0
        state.latest_gray = preprocess_frame(frame)
        small = cv2.resize(frame, (320, 180))
        okj, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if okj:
            state.latest_jpeg = buf.tobytes()
        time.sleep(0.03)
    cap.release()
    state.camera_on = False


async def detection_loop(page, state, slider, status) -> None:
    while not state.stop.is_set():
        now = time.monotonic()
        gray = state.latest_gray
        if gray is not None:
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
                target_t = now - float(slider.value)
                target = min(state.history, key=lambda e: abs(e[0] - target_t))
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
                    and (now - state.last_alert) >= ALERT_COOLDOWN
                ):
                    state.last_alert = now
                    state.hits = 0
                    await trigger_alert(page, status, state)
            except Exception:
                state.hits = 0
        await asyncio.sleep(SAMPLE_INTERVAL)


async def display_loop(page, state, raw) -> None:
    while not state.stop.is_set():
        jpeg = state.latest_jpeg
        if jpeg is not None and jpeg is not state._last_rendered:
            state._last_rendered = jpeg
            try:
                await raw.render_encoded(jpeg)
            except Exception:
                pass
        await asyncio.sleep(0.05)


async def web_poll_camera(state, web_cam, ingest) -> None:
    while not state.stop.is_set():
        try:
            data = await web_cam.take_picture()
            if data:
                ingest(data)
        except Exception:
            pass
        await asyncio.sleep(SAMPLE_INTERVAL)


async def flash_background(page) -> None:
    for _ in range(6):
        page.bgcolor = ft.Colors.RED
        page.update()
        await asyncio.sleep(0.15)
        page.bgcolor = ft.Colors.WHITE
        page.update()
        await asyncio.sleep(0.15)
    page.bgcolor = None
    page.update()


async def trigger_alert(page, status, state) -> None:
    status.value = "MOTION DETECTED!"
    status.color = ft.Colors.RED
    page.update()
    asyncio.create_task(asyncio.to_thread(beep))
    if not state.flashing:
        state.flashing = True
        try:
            await flash_background(page)
        finally:
            state.flashing = False


async def main(page: ft.Page):
    state = MotionState()
    page.title = "Motion Detector"

    status = ft.Text("Camera off", key="status", size=16)
    slider = ft.Slider(
        min=INTERVAL_MIN,
        max=INTERVAL_MAX,
        value=1.0,
        divisions=10,
        label="1.0s",
        key="interval_slider",
    )
    preview_img = ft.RawImage(fit=ft.BoxFit.CONTAIN)
    camera_box = ft.Container(
        key="camera_box",
        height=320,
        bgcolor=ft.Colors.BLACK,
        border_radius=8,
        alignment=ft.Alignment.CENTER,
        content=preview_img,
    )
    start_camera_btn = ft.FilledButton(
        "Start Camera", key="start_camera", icon=ft.Icons.VIDEOCAM
    )
    start_detection_btn = ft.FilledButton(
        "Start Detection", key="start_detection", disabled=True, icon=ft.Icons.MOTION_PHOTOS_ON
    )

    web_cam = None
    web_overlay = None
    if page.web:
        import flet_camera as fc

        def ingest_web_frame(raw_bytes) -> None:
            try:
                arr = np.frombuffer(raw_bytes, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    state.latest_gray = preprocess_frame(img)
            except Exception:
                pass

        def on_stream_image(e):
            ingest_web_frame(e.bytes)

        web_overlay = ft.Container(
            alignment=ft.Alignment.CENTER,
            content=ft.Column(
                [
                    ft.Icon(ft.Icons.CENTER_FOCUS_STRONG, color=ft.Colors.WHITE_70, size=48),
                    ft.Text("Camera off", color=ft.Colors.WHITE_70),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=8,
            ),
        )
        web_cam = fc.Camera(
            preview_enabled=True,
            content=web_overlay,
            on_stream_image=on_stream_image,
        )
        camera_box.content = web_cam

    def shutdown(e=None):
        state.stop.set()

    page.on_disconnect = shutdown

    async def start_camera(e):
        if state.camera_on:
            return
        state.stop.clear()
        start_camera_btn.disabled = True
        status.value = "Starting camera…"
        status.color = None
        page.update()
        if page.web:
            import flet_camera as fc

            try:
                cameras = await web_cam.get_available_cameras()
            except Exception:
                cameras = None
                state.camera_error = (
                    "Camera timeout. Make sure you ALLOW camera access "
                    "in the browser (and try once before denying), then click Start Camera again."
                )
            if not cameras:
                if not state.camera_error:
                    state.camera_error = "No camera found"
                status.value = state.camera_error
                status.color = ft.Colors.RED
            else:
                try:
                    await web_cam.initialize(
                        description=cameras[0],
                        resolution_preset=fc.ResolutionPreset.LOW,
                        enable_audio=False,
                        image_format_group=fc.ImageFormatGroup.JPEG,
                    )
                    state.camera_on = True
                    if web_overlay is not None:
                        web_overlay.visible = False
                    camera_box.update()
                    page.run_task(detection_loop, page, state, slider, status)
                    supports_streaming = await web_cam.supports_image_streaming()
                    if supports_streaming:
                        await web_cam.start_image_stream()
                        status.value = "Camera streaming"
                    else:
                        page.run_task(web_poll_camera, state, web_cam, ingest_web_frame)
                        status.value = "Camera on (snapshot polling)"
                    status.color = None
                    status.update()
                except Exception:
                    state.camera_error = "Camera initialization failed"
                    status.value = state.camera_error
                    status.color = ft.Colors.RED
        else:
            state.camera_on = True
            threading.Thread(target=capture_loop, args=(state,), daemon=True).start()
            page.run_task(detection_loop, page, state, slider, status)
            page.run_task(display_loop, page, state, preview_img)
            for _ in range(100):
                if state.latest_jpeg is not None or state.camera_error:
                    break
                await asyncio.sleep(0.05)
            if state.camera_error:
                state.camera_on = False
                status.value = state.camera_error
            else:
                status.value = "Camera on"

        if state.camera_error:
            status.color = ft.Colors.RED
        else:
            start_detection_btn.disabled = False
            status.color = None
        start_camera_btn.disabled = False
        page.update()

    def start_detection(e):
        state.detecting = True
        start_detection_btn.disabled = True
        status.value = "Detecting motion…"
        status.color = ft.Colors.WHITE
        page.update()

    ui = ft.Column(
        [
            ft.Row([camera_box]),
            ft.Row(
                [
                    start_camera_btn,
                    start_detection_btn,
                ],
                wrap=True,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            ft.Row([ft.Text("Compare to frame from (s):"), slider]),
            ft.Row([status], alignment=ft.MainAxisAlignment.CENTER),
        ]
    )

    start_camera_btn.on_click = start_camera
    start_detection_btn.on_click = start_detection

    page.add(
        ft.SafeArea(
            expand=True,
            content=ft.Row(
                [
                    ft.Container(
                        content=ui,
                        width=UI_WIDTH,
                        border=ft.Border.all(2, ft.Colors.BLACK),
                        padding=16,
                    )
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
        )
    )


if __name__ == "__main__":
    ft.run(main)