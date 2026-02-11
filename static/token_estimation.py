"""
Shared token-estimation helpers.

These utilities intentionally use a lightweight byte heuristic
(~4 bytes/token) for relative comparisons and telemetry.
"""

from __future__ import annotations


_ESTIMATED_TOKEN_RATIO = 4


def estimate_tokens_from_bytes(payload_bytes: int) -> int:
    if payload_bytes <= 0:
        return 0
    return max(1, payload_bytes // _ESTIMATED_TOKEN_RATIO)


def estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    return estimate_tokens_from_bytes(len(text.encode("utf-8")))
