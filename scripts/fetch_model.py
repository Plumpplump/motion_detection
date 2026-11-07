#!/usr/bin/env python3
"""Download the embedded YOLOX-Tiny INT8 model into src/assets/models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision.catalog import COCO_LABELS
from vision.model_store import (
    MODEL_FILENAME,
    MODEL_SHA256,
    download_model,
)


def _progress(downloaded: int, total: int) -> None:
    if total > 0:
        ratio = downloaded / total
        bar_width = 28
        filled = int(bar_width * ratio)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(
            f"\r  {bar} {ratio:6.1%}  {downloaded / 1e6:5.1f} MB",
            end="",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"\r  {downloaded / 1e6:5.1f} MB",
            end="",
            file=sys.stderr,
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        default=str(PROJECT_ROOT / "src" / "assets" / "models"),
        help="destination directory (default: src/assets/models)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="download even if a valid file already exists",
    )
    args = parser.parse_args()

    destination = Path(args.dest).expanduser().resolve()
    print(f"Downloading {MODEL_FILENAME} to {destination}")
    print(f"Expected SHA-256: {MODEL_SHA256}")

    try:
        path = download_model(destination, _progress, force=args.force)
    except Exception as error:
        print(f"\nDownload failed: {error}", file=sys.stderr)
        return 1

    print()
    print(f"Model:    {path}")
    print(f"Size:     {path.stat().st_size / 1e6:.2f} MB")
    print("SHA-256:  verified")

    labels_path = destination / "coco.names"
    labels_path.write_text(
        "\n".join(COCO_LABELS) + "\n",
        encoding="utf-8",
    )
    print(f"Labels:   {labels_path} ({len(COCO_LABELS)} classes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())