"""Locate, verify, and (if needed) download the embedded ONNX model.

Flet exposes the development assets directory through ``FLET_ASSETS_DIR``.
Packaged apps may not be able to read their bundled assets from Python, so a
writable copy under ``FLET_APP_STORAGE_DATA`` is used as a fallback.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable

import httpx

from .catalog import COCO_LABELS
from .types import VisionError

MODEL_FILENAME = "yolox_tiny_ptq_int8.onnx"
LABELS_FILENAME = "coco.names"
MODEL_URL = (
    "https://huggingface.co/jianlee/yolox-tiny-quantized/resolve/main/"
    "yolox_tiny_ptq_int8_calib32.onnx"
)
MODEL_SHA256 = "0d0418035dd5e6ddf4cb108b253490f6fade8292a4fb217d3eaf570f30aa8efb"
LICENSE_URL = (
    "https://huggingface.co/jianlee/yolox-tiny-quantized/resolve/main/LICENSE"
)
NOTICE_URL = (
    "https://huggingface.co/jianlee/yolox-tiny-quantized/resolve/main/NOTICE"
)

ProgressCallback = Callable[[int, int], None]


def asset_models_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "models"


def candidate_model_dirs() -> list[Path]:
    dirs: list[Path] = []

    explicit = os.getenv("VISION_MODEL_DIR", "").strip()
    if explicit:
        dirs.append(Path(explicit).expanduser())

    flet_assets = os.getenv("FLET_ASSETS_DIR", "").strip()
    if flet_assets:
        dirs.append(Path(flet_assets).expanduser() / "models")

    dirs.append(asset_models_dir())

    data_dir = os.getenv("FLET_APP_STORAGE_DATA", "").strip()
    if data_dir:
        dirs.append(Path(data_dir).expanduser() / "models")

    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: list[Path] = []
    for directory in dirs:
        key = str(directory)
        if key not in seen:
            seen.add(key)
            unique.append(directory)
    return unique


def writable_models_dir() -> Path:
    data_dir = os.getenv("FLET_APP_STORAGE_DATA", "").strip()
    if data_dir:
        root = Path(data_dir).expanduser()
    else:
        root = Path.home() / ".motion_sentinel"
    directory = root / "models"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def find_model() -> Path | None:
    for directory in candidate_model_dirs():
        path = directory / MODEL_FILENAME
        if path.is_file():
            return path
    return None


def find_labels() -> Path | None:
    for directory in candidate_model_dirs():
        path = directory / LABELS_FILENAME
        if path.is_file():
            return path
    return None


def load_labels() -> list[str]:
    path = find_labels()
    if path is not None:
        labels = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(labels) == 80:
            return labels
    return list(COCO_LABELS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path) -> bool:
    try:
        return sha256_file(path) == MODEL_SHA256
    except OSError:
        return False


def download_model(
    destination: Path,
    progress: ProgressCallback | None = None,
    *,
    force: bool = False,
) -> Path:
    """Download and verify the model into *destination*."""

    destination.mkdir(parents=True, exist_ok=True)
    target = destination / MODEL_FILENAME
    if target.is_file() and not force and verify_model(target):
        _ensure_aux_files(destination)
        return target

    _download(MODEL_URL, target, progress)
    if not verify_model(target):
        target.unlink(missing_ok=True)
        raise VisionError(
            "The downloaded model failed its SHA-256 check. "
            "Please retry or download it manually.",
            kind="model",
        )
    _ensure_aux_files(destination)
    return target


def ensure_model(progress: ProgressCallback | None = None) -> Path:
    """Return a verified model path, downloading it if necessary."""

    existing = find_model()
    if existing is not None:
        if verify_model(existing):
            return existing
        # A corrupted bundled file should not silently disable the app.
        if not os.access(existing, os.W_OK):
            raise VisionError(
                f"Local model checksum mismatch at {existing}. "
                "Re-run scripts/fetch_model.py or delete the file.",
                kind="model",
            )

    return download_model(writable_models_dir(), progress)


def _ensure_aux_files(directory: Path) -> None:
    for filename, url in (
        ("LICENSE", LICENSE_URL),
        ("NOTICE", NOTICE_URL),
    ):
        target = directory / filename
        if not target.is_file():
            try:
                _download(url, target, None)
            except Exception:
                # Attribution files are nice to have, not required to infer.
                pass


def _download(url: str, target: Path, progress: ProgressCallback | None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with httpx.stream(
            "GET",
            url,
            follow_redirects=True,
            timeout=httpx.Timeout(120.0),
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", "0") or "0")
            downloaded = 0
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(64 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(downloaded, total)
        os.replace(temporary, target)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, VisionError):
            raise
        raise VisionError(
            f"Could not download {url}: {exc}",
            kind="network",
        ) from exc