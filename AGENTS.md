# AGENTS.md

## Project

- Motion Sentinel is a **Flet** (Python) app that detects motion, then asks a
  vision model whether a selected object is visible. It is designed to run in
  the browser (`flet run --web`) for camera access; desktop uses OpenCV.
- Package manager is **uv** (`pyproject.toml`, uv.lock). Use `uv run ...`,
  `uv add`, etc. — never pip directly.

## Run

```bash
./run_web.sh            # recommended: uv run flet run --web
uv run flet run --web   # browser camera control
uv run flet run         # desktop: OpenCV camera
```

- Always use `uv run` (or `run_web.sh` / `run_web.bat`). A bare `flet run`
  from another virtualenv will miss `cv2` / `onnxruntime`.
- App entrypoint is `src/main.py` (set via `[tool.flet.app] path = "src"`).
- `use_cv2_camera(page)` chooses the camera path: OpenCV on desktop,
  `flet_camera` on web and native mobile.
- Web camera access is handled by `flet_camera`: camera control mounted
  visible from the start, optional streaming, snapshot fallback,
  `take_picture()` for the recognition snapshot.
- The camera button toggles start/stop; the detection button toggles
  start/stop on the same control.
- macOS desktop camera access needs permission for the terminal/IDE in
  System Settings → Privacy & Security → Camera.

## Architecture

```
src/
  main.py        Flet entrypoint, camera loops, motion-triggered orchestration
  ui.py          two-column dashboard (settings left, camera centre)
  detection.py   pure motion helpers (preprocess_frame, motion_ratio)
  vision/
    base.py      VisionBackend interface
    embedded.py  YOLOX-Tiny INT8 ONNX backend (default, local)
    remote.py    OpenAI-compatible backend (OpenRouter/Ollama/custom)
    providers.py provider presets + BackendConfig
    factory.py   config -> backend
    model_store.py model paths, SHA-256, download
    pipeline.py  motion-gated cooldowns and warning rules
    catalog.py   object names <-> COCO labels
    image.py     letterbox, JPEG, box drawing
    parsing.py   robust model reply parsing
    prompt.py    hosted-model prompt
    fake.py      deterministic test backend
  assets/models/ model, coco.names, LICENSE, NOTICE
```

- Recognition only runs after motion. Do not add a manual "take picture" flow
  unless the user asks for it.
- Adding a provider: implement `VisionBackend.check()` or add a preset in
  `vision/providers.py`; the UI reads provider presets dynamically.
- The default model is in `src/assets/models/yolox_tiny_ptq_int8.onnx`
  (5.3 MB, Apache-2.0, 80 COCO classes). Re-download with
  `uv run python scripts/fetch_model.py`.
- Runtime model resolution checks `VISION_MODEL_DIR`, `FLET_ASSETS_DIR`,
  `src/assets/models`, then `FLET_APP_STORAGE_DATA/models`.

## Test

- Tests live in `tests/` (`[tool.pytest.ini_options] testpaths`), async mode is auto.
- Pure logic/integration tests: `uv run pytest tests/test_vision.py tests/test_engine.py`.
- Full Flet UI tests use the injected `flet_app` fixture (`tests/test_main.py`);
  run `uv run flet test` or `uv run pytest`. The Flet test host needs a Flutter
  desktop toolchain. Do not write your own `flet_app` fixture.
- Use `tester.pump_and_settle()`, `find_by_key`, and `find_by_text` as in
  `tests/test_main.py`.

## Build

- `uv run flet build web` / `macos` / `windows` / `linux` / `apk` / `ipa`.
- For a packaged macOS app with OpenCV, add `NSCameraUsageDescription` to the
  app's Info.plist.

## Git / repo layout

- This folder has **no `.git`** — the git repo root is the Windows home dir
  (`C:\Users\david`), which must never be staged/committed. Do not `git init`,
  `git add`, or commit here; just edit files.

## Tooling

- No CI, lint, or typecheck setup exists; don't add tooling that isn't already
  present in sibling Python projects.
- `.gitignore` covers Python caches, `.venv`, `.flet/`, `build/`, `storage/`.

## Dependencies

- Runtime: `flet`, `flet-camera`, `opencv-python`, `numpy` (via OpenCV),
  `onnxruntime`, `httpx`, `python-dotenv`.
- Dev: `flet-cli`, `flet-desktop`, `flet-web`, `flet[test]`.
- No PyTorch / Transformers / Ultralytics.