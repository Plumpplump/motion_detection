"""Image helpers shared by the embedded detector and remote backends."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from .types import BBox


def decode_jpeg(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode the captured image")
    return frame


def resize_max(frame: np.ndarray, max_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame
    scale = max_side / longest
    return cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def encode_jpeg(
    frame: np.ndarray,
    max_side: int = 768,
    quality: int = 85,
) -> bytes:
    frame = resize_max(frame, max_side)
    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise ValueError("Could not encode the image as JPEG")
    return buffer.tobytes()


def prepare_jpeg(image_jpeg: bytes, max_side: int = 768, quality: int = 85) -> bytes:
    """Normalise a snapshot before sending it to a remote model."""

    try:
        return encode_jpeg(decode_jpeg(image_jpeg), max_side=max_side, quality=quality)
    except Exception:
        return image_jpeg


def to_data_url(image_jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode("ascii")


def letterbox(
    frame: np.ndarray,
    size: int = 416,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, int, int]:
    """Resize while preserving aspect ratio and pad to *size* x *size*.

    Returns the padded image plus the scale and padding offsets needed to map
    boxes back to the original frame.
    """

    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    pad_x = (size - new_width) // 2
    pad_y = (size - new_height) // 2
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return canvas, scale, pad_x, pad_y


def unletterbox_box(
    box: tuple[float, float, float, float],
    scale: float,
    pad_x: int,
    pad_y: int,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale
    x1 = min(max(x1, 0.0), float(frame_width))
    y1 = min(max(y1, 0.0), float(frame_height))
    x2 = min(max(x2, 0.0), float(frame_width))
    y2 = min(max(y2, 0.0), float(frame_height))
    return x1, y1, x2, y2


def draw_boxes(
    frame: np.ndarray,
    boxes: tuple[BBox, ...] | list[BBox],
    color: tuple[int, int, int] = (34, 197, 94),
) -> np.ndarray:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    thickness = max(2, round(max(height, width) / 320))
    font_scale = max(0.45, max(height, width) / 1400)
    for box in boxes:
        p1 = (round(box.x1), round(box.y1))
        p2 = (round(box.x2), round(box.y2))
        cv2.rectangle(annotated, p1, p2, color, thickness)
        caption = box.label or "object"
        if box.confidence:
            caption = f"{caption} {box.confidence:.0%}"
        text_y = max(18, p1[1] - 8)
        cv2.putText(
            annotated,
            caption,
            (p1[0], text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
    return annotated


def annotate_jpeg(
    image_jpeg: bytes,
    boxes: tuple[BBox, ...] | list[BBox],
) -> bytes:
    if not boxes:
        return image_jpeg
    try:
        frame = decode_jpeg(image_jpeg)
        return encode_jpeg(draw_boxes(frame, boxes), max_side=768, quality=85)
    except Exception:
        return image_jpeg