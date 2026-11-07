"""Pluggable local and remote vision backends."""

from .base import VisionBackend
from .catalog import (
    COCO_OBJECTS,
    CUSTOM_OBJECT_KEY,
    DEFAULT_OBJECT_KEY,
    ObjectSpec,
    find_object,
    make_custom_target,
    resolve_target,
)
from .embedded import EmbeddedYoloBackend
from .factory import create_backend
from .fake import FakeBackend
from .pipeline import RecognitionPipeline
from .providers import PROVIDERS, BackendConfig, config_from_env
from .remote import OpenAICompatibleBackend
from .types import BBox, DetectionResult, VisionError

__all__ = [
    "BackendConfig",
    "BBox",
    "COCO_OBJECTS",
    "CUSTOM_OBJECT_KEY",
    "DEFAULT_OBJECT_KEY",
    "DetectionResult",
    "EmbeddedYoloBackend",
    "FakeBackend",
    "ObjectSpec",
    "OpenAICompatibleBackend",
    "PROVIDERS",
    "RecognitionPipeline",
    "VisionBackend",
    "VisionError",
    "config_from_env",
    "create_backend",
    "find_object",
    "make_custom_target",
    "resolve_target",
]