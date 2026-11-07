#!/usr/bin/env bash
# Launch the web app with the project environment (cv2 + onnxruntime included).
set -euo pipefail
cd "$(dirname "$0")"
exec uv run flet run --web "$@"