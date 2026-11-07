# Motion Sentinel

A Flet app that watches a camera for motion, then asks a vision model whether a
chosen object is visible. Recognition only runs after motion is detected, so the
app stays quiet and light when nothing is happening.

- **Local-first:** the default detector is a 5.3 MB YOLOX-Tiny INT8 ONNX model
  running on this computer through ONNX Runtime.
- **Switchable:** the same UI can use OpenRouter, a local Ollama server, or any
  OpenAI-compatible endpoint.
- **Automatic:** no manual shutter button. A motion event takes the snapshot and
  runs the check.
- **Visible:** motion flashes the camera container red; when the watched object
  is found, the GUI adds a red warning banner and an optional beep. The flash is
  scoped to the camera card so the browser camera preview stays alive.

## Run

The camera works best in the browser, using Flet's camera control:

```bash
./run_web.sh
# or
uv run flet run --web
```

Open the URL Flet prints and allow camera access when the browser asks.

> **Use `uv run` (or the launcher script).** A bare `flet run --web` uses
> whichever virtualenv is active and may be missing `cv2` or `onnxruntime`.
> The project environment is defined in `pyproject.toml` / `uv.lock`. The app
> prints its Python path at startup so you can confirm the environment.

Desktop mode uses OpenCV instead:

```bash
uv run flet run
```

On macOS, OpenCV needs camera permission for the terminal/IDE that launched the
app: **System Settings → Privacy & Security → Camera**. If the camera still
fails, quit and reopen the terminal, then try again. On Windows, check
**Settings → Privacy → Camera**.

## How it works

```
camera frame
   ↓
motion check (frame difference, existing detection.py)
   ↓  motion confirmed
snapshot of the triggering frame
   ↓
vision backend
   ├─ embedded YOLOX-Tiny INT8  (default, local)
   └─ OpenRouter / Ollama / custom OpenAI-compatible
   ↓
object present?
   ├─ yes → red warning banner + optional flash/beep
   └─ no  → quiet status update
```

If motion continues, checks are throttled (about once every 3 seconds) and the
same warning is not repeated for 15 seconds.

## Choosing the object

The left sidebar has an object selector with friendly names mapped to the
COCO labels the local model knows:

| UI name | Model label |
|---|---|
| Smartphone | `cell phone` |
| Laptop | `laptop` |
| Book | `book` |
| Person | `person` |
| Cat / Dog | `cat` / `dog` |
| Car / Bicycle | `car` / `bicycle` |
| Bottle / Cup / Chair | `bottle` / `cup` / `chair` |
| Keyboard / Mouse | `keyboard` / `mouse` |
| TV / Backpack / Handbag | `tv` / `backpack` / `handbag` |
| Clock / Vase / Scissors / Toothbrush | matching COCO labels |

The embedded model only supports these COCO classes. Select **Custom object…**
and switch the AI backend to OpenRouter (or another remote model) for arbitrary
objects such as “wallet”, “keys”, or “red mug”.

## AI backends

Open **AI backend** in the left sidebar. The default is **Local (embedded)**.

### OpenRouter

Set the key in the sidebar, or use an environment variable:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

Optional `.env`:

```bash
VISION_PROVIDER=openrouter
VISION_MODEL=google/gemini-2.5-flash
OPENROUTER_API_KEY=sk-or-...
```

The model field is editable, so any vision-capable OpenRouter model can be used.

### Ollama (local server)

```bash
ollama serve
ollama pull qwen2.5vl:7b
```

Select **Ollama (local server)** in the sidebar. The default endpoint is
`http://localhost:11434/v1`.

### Custom OpenAI-compatible endpoint

Select **Custom OpenAI-compatible** and fill in the model and base URL. This
works with LM Studio, vLLM, llama.cpp server, SGLang, and similar services.

### Configuration reference

| Variable | Meaning |
|---|---|
| `VISION_PROVIDER` | `embedded`, `openrouter`, `ollama`, `custom` |
| `VISION_MODEL` | Model name passed to the provider |
| `VISION_BASE_URL` | OpenAI-compatible base URL (ends in `/v1`) |
| `VISION_API_KEY` | Generic API key (or use `OPENROUTER_API_KEY`) |
| `VISION_TIMEOUT` | Request timeout in seconds (default 45) |
| `VISION_THRESHOLD` | Local confidence threshold (default 0.35) |
| `VISION_MODEL_DIR` | Override the directory containing the ONNX model |

## Embedded model

The model lives at:

```
src/assets/models/yolox_tiny_ptq_int8.onnx
```

- YOLOX-Tiny, INT8 post-training quantization
- 5.3 MB, Apache-2.0
- COCO 80 classes, 416×416 input
- ~20–45 ms CPU inference in testing
- Source: <https://huggingface.co/jianlee/yolox-tiny-quantized>
- SHA-256: `0d0418035dd5e6ddf4cb108b253490f6fade8292a4fb217d3eaf570f30aa8efb`

Re-download or verify it with:

```bash
uv run python scripts/fetch_model.py
```

At runtime the loader checks `FLET_ASSETS_DIR`, the project `src/assets/models`
directory, and finally a writable `FLET_APP_STORAGE_DATA/models` fallback. If the
file is missing, it downloads and hash-verifies it there.

## Architecture

```
src/
  main.py                  camera loops, motion → recognition orchestration, Flet entrypoint
  ui.py                    two-column dashboard, warning banner, settings sidebar
  detection.py             pure motion helpers (preprocess_frame, motion_ratio)
  vision/
    base.py                VisionBackend interface
    embedded.py            YOLOX INT8 ONNX backend
    remote.py              OpenAI-compatible backend (OpenRouter/Ollama/custom)
    providers.py           provider presets + BackendConfig
    factory.py             config → backend
    model_store.py         model paths, SHA-256, download
    pipeline.py            motion-gated cooldowns and warning rules
    catalog.py             object names ↔ COCO labels
    image.py               letterbox, JPEG, bounding-box drawing
    parsing.py             robust JSON reply parsing
    prompt.py              hosted-model prompt
    fake.py                deterministic backend for tests
  assets/
    models/                model, COCO labels, licence, notice
tests/
  test_main.py             Flet UI smoke tests
  test_vision.py           pure vision/backends tests
  test_engine.py           motion → warning integration tests
scripts/
  fetch_model.py           reproducible model download
```

Adding a backend means implementing `VisionBackend.check()` and registering a
provider preset; the UI and motion pipeline do not need to change.

## Tests

Pure logic and integration tests (no Flutter toolchain needed):

```bash
uv run pytest tests/test_vision.py tests/test_engine.py
```

Full Flet UI tests use the injected `flet_app` fixture:

```bash
uv run flet test
```

The Flet test host requires a Flutter desktop toolchain; if it is not installed,
the UI tests cannot start. The web app can be verified manually with
`uv run flet run --web`.

## Privacy

With **Local (embedded)** selected, frames never leave the computer. When a
remote backend is selected, snapshots are sent to that provider; the app labels
remote backends clearly in the status chip.

## Build

```bash
uv run flet build macos    # or windows, linux, web, apk, ipa
```

For camera access in a packaged macOS app, add `NSCameraUsageDescription` to the
app's Info.plist. The development web flow does not need it.