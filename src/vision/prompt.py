"""Prompt used with hosted vision-language models."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a strict visual object verifier. You receive exactly one image "
    "and one target object. Decide whether the target object is clearly "
    "visible in the image. Do not guess: if the object is absent, tiny, "
    "heavily occluded, or ambiguous, answer false. "
    "Reply with JSON only, without markdown fences or commentary."
)


def build_user_prompt(target: str) -> str:
    return (
        f'Target object: "{target}".\n'
        "Is the target object clearly visible in this image?\n"
        "Return exactly this JSON shape:\n"
        '{"present": true or false, "confidence": 0.0-1.0, '
        '"reason": "one short sentence"}\n'
        "The confidence is your certainty about the presence decision."
    )


def build_messages(target: str, image_data_url: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_user_prompt(target)},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url},
                },
            ],
        },
    ]