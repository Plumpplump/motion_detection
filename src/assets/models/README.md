# Embedded vision model

Default object detector for the app. It runs locally on the CPU through
ONNX Runtime; no network access or API key is needed at inference time.

| Property | Value |
|---|---|
| Model | YOLOX-Tiny, INT8 post-training quantization (32 calibration images) |
| Source | https://huggingface.co/jianlee/yolox-tiny-quantized |
| File | `yolox_tiny_ptq_int8.onnx` |
| Size | 5.3 MB |
| SHA-256 | `0d0418035dd5e6ddf4cb108b253490f6fade8292a4fb217d3eaf570f30aa8efb` |
| License | Apache-2.0 (see `LICENSE` and `NOTICE`) |
| Input | `images`: float32 `[1, 3, 416, 416]`, BGR, 0-255, NCHW |
| Output | `output`: float32 `[1, 3549, 85]` |
| Classes | 80 COCO classes, see `coco.names` |

The model output is `[cx, cy, width, height, objectness, class_scores...]`
with box decoding already inside the ONNX graph. Non-maximum suppression is
applied by the application.

## Re-download

```bash
uv run python scripts/fetch_model.py
```

The script verifies the SHA-256 above before accepting the file.