from __future__ import annotations

from typing import Protocol

from .types import ModelRequest, ModelTurn


class ModelBackend(Protocol):
    """Adapter implemented by IQ inference or any other model backend."""

    def generate(self, request: ModelRequest) -> ModelTurn:
        ...
