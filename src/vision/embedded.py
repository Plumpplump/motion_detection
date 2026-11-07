"""Local YOLOX-Tiny INT8 detector running through ONNX Runtime."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import VisionBackend
from .catalog import COCO_LABELS
from .image import decode_jpeg, letterbox, unletterbox_box
from .types import BBox, DetectionResult, VisionError


class EmbeddedYoloBackend(VisionBackend):
    """YOLOX-Tiny (COCO, 416 px) compiled to INT8 ONNX.

    The model output is ``[cx, cy, width, height, objectness, class…]`` in the
    letterboxed 416x416 coordinate space.  Non-maximum suppression is applied
    here; everything else is already inside the ONNX graph.
    """

    key = "embedded"
    label = "Local (embedded)"
    local = True

    def __init__(
        self,
        model_path: str | Path,
        labels: list[str] | tuple[str, ...] | None = None,
        *,
        input_size: int = 416,
        nms_iou: float = 0.45,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels = list(labels or COCO_LABELS)
        self.input_size = input_size
        self.nms_iou = nms_iou
        self._session: Any | None = None
        self._input_name = "images"
        self._output_name = "output"
        self._lock = threading.Lock()
        self._label_to_id = {label.lower(): i for i, label in enumerate(self.labels)}

    @property
    def ready(self) -> bool:
        return self._session is not None

    async def prepare(self) -> None:
        if self._session is not None:
            return
        await asyncio.to_thread(self._load_session)

    def _load_session(self) -> None:
        with self._lock:
            if self._session is not None:
                return
            if not self.model_path.is_file():
                raise VisionError(
                    f"Local model not found at {self.model_path}",
                    kind="model",
                )
            try:
                import onnxruntime as ort
            except ImportError as exc:  # pragma: no cover - environment issue
                raise VisionError(
                    f"onnxruntime is not installed for {sys.executable}. "
                    "Start the app with `uv run flet run --web` "
                    "(or `uv add onnxruntime`).",
                    kind="dependency",
                ) from exc
            try:
                session = ort.InferenceSession(
                    str(self.model_path),
                    providers=["CPUExecutionProvider"],
                )
            except Exception as exc:
                raise VisionError(
                    f"Could not load the local model: {exc}",
                    kind="model",
                ) from exc
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if inputs:
                self._input_name = inputs[0].name
            if outputs:
                self._output_name = outputs[0].name
            self._session = session

    async def check(
        self,
        image_jpeg: bytes,
        target: str,
        *,
        threshold: float = 0.35,
    ) -> DetectionResult:
        if self._session is None:
            await self.prepare()
        target = target.strip()
        class_id = self._label_to_id.get(target.lower())
        if class_id is None:
            raise VisionError(
                f"The local model does not know '{target}'. "
                "Switch to OpenRouter in AI settings for custom objects.",
                kind="unsupported",
            )
        return await asyncio.to_thread(
            self._check_sync,
            image_jpeg,
            target,
            class_id,
            float(threshold),
        )

    def _check_sync(
        self,
        image_jpeg: bytes,
        target: str,
        class_id: int,
        threshold: float,
    ) -> DetectionResult:
        started = time.perf_counter()
        frame = decode_jpeg(image_jpeg)
        height, width = frame.shape[:2]
        padded, scale, pad_x, pad_y = letterbox(frame, self.input_size)

        # YOLOX expects BGR, 0-255, NCHW.
        tensor = padded.astype(np.float32).transpose(2, 0, 1)[None, ...]
        assert self._session is not None
        output = self._session.run(
            [self._output_name],
            {self._input_name: tensor},
        )[0]
        predictions = np.asarray(output).reshape(-1, 85)

        class_scores = predictions[:, 5 + class_id]
        scores = predictions[:, 4] * class_scores
        mask = scores >= threshold
        if not np.any(mask):
            best = float(scores.max()) if scores.size else 0.0
            return DetectionResult(
                present=False,
                label=target,
                confidence=best,
                reason=f"No {target} above {threshold:.0%} confidence.",
                latency_ms=(time.perf_counter() - started) * 1000,
                backend=self.key,
                target=target,
            )

        selected = predictions[mask]
        selected_scores = scores[mask]
        cx, cy, bw, bh = (
            selected[:, 0],
            selected[:, 1],
            selected[:, 2],
            selected[:, 3],
        )
        boxes_xyxy = np.stack(
            [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
            axis=1,
        )
        keep = _nms(boxes_xyxy, selected_scores, self.nms_iou)
        boxes: list[BBox] = []
        for index in keep:
            mapped = unletterbox_box(
                tuple(float(v) for v in boxes_xyxy[index]),
                scale,
                pad_x,
                pad_y,
                width,
                height,
            )
            boxes.append(
                BBox(
                    x1=mapped[0],
                    y1=mapped[1],
                    x2=mapped[2],
                    y2=mapped[3],
                    confidence=float(selected_scores[index]),
                    label=target,
                )
            )
        best_confidence = max((box.confidence for box in boxes), default=0.0)
        latency = (time.perf_counter() - started) * 1000
        return DetectionResult(
            present=True,
            label=target,
            confidence=best_confidence,
            reason=f"{len(boxes)} matching {target} box(es) found.",
            boxes=tuple(boxes),
            latency_ms=latency,
            backend=self.key,
            target=target,
        )


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Greedy non-maximum suppression with numpy only."""

    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_threshold]

    return keep