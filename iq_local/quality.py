"""Honest, deterministic IQ generation canaries (not broad language evaluation)."""

from __future__ import annotations

import re

_CANARY = re.compile(
    r"(?:4(?:\.0+)?|four|(?:the answer is|answer:|it is)\s+(?:4(?:\.0+)?|four))[.! ]*",
    re.IGNORECASE,
)
_REPETITION = re.compile(r"([a-zA-Z]{3,10})\1{2,}")


def check_arithmetic_canary(output: str) -> tuple[bool, str]:
    """Verify a narrow '2 + 2 = 4' completion, not general language quality."""
    if not isinstance(output, str) or not output.strip():
        return False, "empty or non-text response"
    text = output.strip()
    if len(text) > 80:
        return False, "response too long for a digit-only canary"
    if _REPETITION.search(text):
        return False, "repeated subwords"
    if _CANARY.fullmatch(text) is None:
        return False, "incorrect arithmetic response"
    return True, "2+2 canary passed"
