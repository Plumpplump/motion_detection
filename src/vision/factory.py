"""Create a backend from user configuration.

This is the single place that knows how a provider name maps to an
implementation.  The UI only edits a :class:`BackendConfig`.
"""

from __future__ import annotations

from pathlib import Path

from . import model_store
from .base import VisionBackend
from .embedded import EmbeddedYoloBackend
from .providers import PROVIDERS, BackendConfig
from .remote import OpenAICompatibleBackend
from .types import VisionError


def create_backend(
    config: BackendConfig,
    *,
    model_path: str | Path | None = None,
    labels: list[str] | None = None,
) -> VisionBackend:
    preset = config.preset

    if preset.key == "embedded":
        path = Path(model_path) if model_path is not None else model_store.find_model()
        if path is None:
            raise VisionError(
                "The local model is not available. "
                "Run `uv run python scripts/fetch_model.py`.",
                kind="model",
            )
        return EmbeddedYoloBackend(path, labels or model_store.load_labels())

    api_key = config.resolved_api_key
    if preset.requires_key and not api_key:
        raise VisionError(
            f"{preset.label} needs an API key. "
            f"Set {preset.api_key_env} or paste it in AI settings.",
            kind="auth",
        )

    extra_headers: dict[str, str] = {}
    if preset.key == "openrouter":
        extra_headers = {
            "HTTP-Referer": "https://github.com/local/motion-sentinel",
            "X-Title": "Motion Sentinel",
        }

    local = preset.key in {"ollama"} or "localhost" in config.resolved_base_url
    return OpenAICompatibleBackend(
        key=preset.key,
        label=preset.label,
        base_url=config.resolved_base_url,
        model=config.resolved_model,
        api_key=api_key,
        timeout=config.timeout,
        requires_key=preset.requires_key,
        extra_headers=extra_headers,
        json_mode=config.json_mode,
        local=local,
    )


def provider_presets():
    return PROVIDERS