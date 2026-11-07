from __future__ import annotations

import httpx
import numpy as np
import pytest

from vision.catalog import (
    COCO_OBJECTS,
    find_object,
    resolve_target,
)
from vision.embedded import EmbeddedYoloBackend, _nms
from vision.factory import create_backend
from vision.fake import FakeBackend
from vision.image import encode_jpeg, letterbox, unletterbox_box
from vision.model_store import find_model, load_labels
from vision.parsing import parse_detection_reply
from vision.pipeline import RecognitionPipeline
from vision.providers import BackendConfig, PROVIDERS
from vision.remote import OpenAICompatibleBackend
from vision.types import BBox, VisionError


def test_catalog_contains_expected_objects():
    assert find_object("smartphone").model_label == "cell phone"
    assert find_object("laptop").model_label == "laptop"
    assert find_object("book").model_label == "book"
    assert len(COCO_OBJECTS) >= 20
    assert all(obj.supported_locally for obj in COCO_OBJECTS)


def test_custom_target():
    target = resolve_target("custom", "  wallet  ")
    assert target.label == "wallet"
    assert target.model_label == "wallet"
    assert not target.supported_locally


def test_letterbox_round_trip():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    padded, scale, pad_x, pad_y = letterbox(frame, 416)
    assert padded.shape == (416, 416, 3)
    original = (10.0, 20.0, 100.0, 200.0)
    # Map into the letterboxed frame, then back.
    letterboxed = (
        original[0] * scale + pad_x,
        original[1] * scale + pad_y,
        original[2] * scale + pad_x,
        original[3] * scale + pad_y,
    )
    restored = unletterbox_box(letterboxed, scale, pad_x, pad_y, 640, 480)
    assert restored == pytest.approx(original, abs=1e-4)


def test_parse_clean_json():
    present, confidence, reason, raw = parse_detection_reply(
        '{"present": true, "confidence": 0.87, "reason": "A phone is visible."}',
        "smartphone",
    )
    assert present is True
    assert confidence == pytest.approx(0.87)
    assert "phone" in reason
    assert raw.startswith("{")


def test_parse_fenced_json():
    present, confidence, _, _ = parse_detection_reply(
        '```json\n{"present": false, "confidence": 92, "reason": "Not visible"}\n```',
        "laptop",
    )
    assert present is False
    assert confidence == pytest.approx(0.92)


def test_parse_synonyms_and_percentage():
    present, confidence, reason, _ = parse_detection_reply(
        '{"found": "yes", "score": 87, "explanation": "A book is on the desk."}',
        "book",
    )
    assert present is True
    assert confidence == pytest.approx(0.87)
    assert "book" in reason


def test_parse_invalid_reply_raises():
    with pytest.raises(VisionError):
        parse_detection_reply("I am not sure what you mean.", "book")


def test_nms_suppresses_overlapping_boxes():
    boxes = np.array(
        [
            [0, 0, 100, 100],
            [5, 5, 105, 105],
            [200, 200, 300, 300],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = _nms(boxes, scores, 0.45)
    assert keep == [0, 2]


async def test_pipeline_cooldowns():
    clock = [0.0]
    backend = FakeBackend(present=True)
    pipeline = RecognitionPipeline(
        backend,
        check_cooldown=3.0,
        warn_cooldown=15.0,
        clock=lambda: clock[0],
    )

    first = await pipeline.on_motion(b"frame", "smartphone")
    assert first is not None and first.present
    assert pipeline.should_warn(first) is True
    assert pipeline.should_warn(first) is False

    clock[0] = 1.0
    assert await pipeline.on_motion(b"frame", "smartphone") is None

    clock[0] = 3.0
    second = await pipeline.on_motion(b"frame", "smartphone")
    assert second is not None
    assert pipeline.should_warn(second) is False

    clock[0] = 18.0
    assert pipeline.should_warn(second) is True


async def test_fake_backend_records_calls():
    backend = FakeBackend(present=False, confidence=0.12)
    result = await backend.check(b"abc", "cell phone", threshold=0.4)
    assert result.present is False
    assert backend.calls == [(b"abc", "cell phone", 0.4)]


def test_provider_presets():
    assert PROVIDERS["embedded"].local is True
    assert PROVIDERS["openrouter"].requires_key is True
    assert PROVIDERS["openrouter"].api_key_env == "OPENROUTER_API_KEY"
    assert PROVIDERS["ollama"].base_url.startswith("http://localhost")


def test_factory_openrouter_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(VisionError) as error:
        create_backend(BackendConfig(provider="openrouter"))
    assert error.value.kind == "auth"


def test_factory_embedded_uses_model_path():
    path = find_model()
    if path is None:
        pytest.skip("embedded model is not present")
    backend = create_backend(
        BackendConfig(provider="embedded"),
        model_path=path,
        labels=load_labels(),
    )
    assert isinstance(backend, EmbeddedYoloBackend)
    assert backend.model_path == path


async def test_remote_backend_parses_reply():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"present": true, "confidence": 0.88, '
                                '"reason": "A phone is visible."}'
                            )
                        }
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        backend = OpenAICompatibleBackend(
            key="openrouter",
            label="OpenRouter",
            base_url="https://example.test/v1",
            model="test-model",
            api_key="test-key",
            client=client,
        )
        result = await backend.check(b"not-a-real-jpeg", "smartphone")
        assert result.present is True
        assert result.confidence == pytest.approx(0.88)
        assert result.backend == "openrouter"
    finally:
        await client.aclose()


async def test_remote_backend_maps_auth_error():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "Invalid API key"}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        backend = OpenAICompatibleBackend(
            key="openrouter",
            label="OpenRouter",
            base_url="https://example.test/v1",
            model="test-model",
            api_key="bad-key",
            requires_key=True,
            client=client,
        )
        with pytest.raises(VisionError) as error:
            await backend.check(b"not-a-real-jpeg", "smartphone")
        assert error.value.kind == "auth"
        assert error.value.status_code == 401
    finally:
        await client.aclose()


async def test_embedded_model_smoke():
    path = find_model()
    if path is None:
        pytest.skip("embedded model is not present")

    backend = EmbeddedYoloBackend(path, load_labels())
    await backend.prepare()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    jpeg = encode_jpeg(blank, 640, 85)
    result = await backend.check(jpeg, "cell phone")
    assert result.present is False
    assert result.latency_ms > 0


def test_bbox_geometry():
    box = BBox(10, 20, 30, 50, confidence=0.5, label="book")
    assert box.width == 20
    assert box.height == 30
    assert box.as_tuple() == (10, 20, 30, 50)