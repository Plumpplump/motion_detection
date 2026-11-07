"""Provider presets and user-editable backend configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ProviderPreset:
    key: str
    label: str
    emoji: str
    local: bool
    base_url: str = ""
    default_model: str = ""
    api_key_env: str = ""
    requires_key: bool = False
    description: str = ""


PROVIDERS: dict[str, ProviderPreset] = {
    "embedded": ProviderPreset(
        key="embedded",
        label="Local (embedded)",
        emoji="🟢",
        local=True,
        description="YOLOX-Tiny INT8 · 5.3 MB · runs on this computer, no network",
    ),
    "openrouter": ProviderPreset(
        key="openrouter",
        label="OpenRouter",
        emoji="☁️",
        local=False,
        base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-2.5-flash",
        api_key_env="OPENROUTER_API_KEY",
        requires_key=True,
        description="Cloud vision model · works for any object you type",
    ),
    "ollama": ProviderPreset(
        key="ollama",
        label="Ollama (local server)",
        emoji="🦙",
        local=True,
        base_url="http://localhost:11434/v1",
        default_model="qwen2.5vl:7b",
        description="OpenAI-compatible local server · no API key",
    ),
    "custom": ProviderPreset(
        key="custom",
        label="Custom OpenAI-compatible",
        emoji="🔧",
        local=False,
        base_url="",
        default_model="",
        api_key_env="VISION_API_KEY",
        description="LM Studio, vLLM, llama.cpp server, or another endpoint",
    ),
}

DEFAULT_PROVIDER = "embedded"


@dataclass
class BackendConfig:
    provider: str = DEFAULT_PROVIDER
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    timeout: float = 45.0
    threshold: float = 0.35
    json_mode: bool = True

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            self.provider = DEFAULT_PROVIDER
        self.threshold = max(0.05, min(0.95, float(self.threshold)))

    @property
    def preset(self) -> ProviderPreset:
        return PROVIDERS[self.provider]

    @property
    def resolved_model(self) -> str:
        return (self.model or self.preset.default_model).strip()

    @property
    def resolved_base_url(self) -> str:
        return (self.base_url or self.preset.base_url).strip().rstrip("/")

    @property
    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key.strip()
        if self.preset.api_key_env:
            return os.getenv(self.preset.api_key_env, "").strip()
        return ""

    def with_provider(self, provider: str) -> "BackendConfig":
        return replace(
            self,
            provider=provider,
            model="",
            base_url="",
        )


def config_from_env() -> BackendConfig:
    provider = os.getenv("VISION_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if provider not in PROVIDERS:
        provider = DEFAULT_PROVIDER

    def _float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError:
            return default

    return BackendConfig(
        provider=provider,
        model=os.getenv("VISION_MODEL", "").strip(),
        base_url=os.getenv("VISION_BASE_URL", "").strip(),
        api_key=os.getenv("VISION_API_KEY", "").strip(),
        timeout=_float("VISION_TIMEOUT", 45.0),
        threshold=_float("VISION_THRESHOLD", 0.35),
    )