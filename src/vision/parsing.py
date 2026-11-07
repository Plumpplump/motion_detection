"""Defensive parsing for model replies.

Hosted VLMs are usually good at JSON, but not perfectly consistent.  This
module accepts fenced JSON, prose wrapped around one object, and common key
synonyms, then normalises the result.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .types import VisionError

_TRUE = {"true", "yes", "y", "1", "detected", "present", "found"}
_FALSE = {"false", "no", "n", "0", "absent", "not detected", "not found", "none"}


def parse_detection_reply(text: str, target: str) -> tuple[bool, float, str, str]:
    """Return ``(present, confidence, reason, raw)`` from a model reply."""

    raw = text or ""
    if not raw.strip():
        raise VisionError("The AI model returned an empty answer.", kind="parse")

    data = _extract_json(raw)
    if isinstance(data, list) and data:
        data = data[0] if isinstance(data[0], dict) else None
    if isinstance(data, dict):
        present = _pick_bool(data)
        if present is not None:
            confidence = _pick_confidence(data)
            reason = _pick_reason(data)
            if confidence is None:
                confidence = 0.75 if present else 0.25
            return present, confidence, reason, raw

    lowered = raw.lower()
    match = re.search(
        r'"?present"?\s*[:=]\s*"?(true|false|yes|no|1|0)"?',
        lowered,
    )
    if match:
        present = match.group(1) in _TRUE
        confidence = _as_confidence(
            _search_number(lowered, ("confidence", "score", "probability"))
        )
        return (
            present,
            confidence if confidence is not None else (0.75 if present else 0.25),
            "Parsed from a non-standard reply.",
            raw,
        )

    raise VisionError(
        "The AI model did not return a usable JSON answer. "
        "Try a more capable vision model.",
        kind="parse",
    )


def _extract_json(text: str) -> Any | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : index + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
        start = cleaned.find("{", start + 1)
    return None


def _pick_bool(data: dict) -> bool | None:
    for key in ("present", "found", "detected", "is_present", "exists", "visible"):
        if key in data:
            return _as_bool(data[key])
    for value in data.values():
        if isinstance(value, dict):
            nested = _pick_bool(value)
            if nested is not None:
                return nested
    return None


def _pick_confidence(data: dict) -> float | None:
    for key in ("confidence", "score", "probability", "certainty"):
        if key in data:
            return _as_confidence(data[key])
    return None


def _pick_reason(data: dict) -> str:
    for key in ("reason", "explanation", "description", "message", "answer"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:400]
    return "No reason provided."


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
    return None


def _as_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence > 1.0:
        confidence = confidence / 100.0 if confidence <= 100.0 else 1.0
    return max(0.0, min(1.0, confidence))


def _search_number(text: str, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        match = re.search(rf'"?{key}"?\s*[:=]\s*([0-9]*\.?[0-9]+)', text)
        if match:
            return float(match.group(1))
    return None