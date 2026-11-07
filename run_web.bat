@echo off
REM Launch the web app with the project environment (cv2 + onnxruntime included).
cd /d "%~dp0"
uv run flet run --web %*