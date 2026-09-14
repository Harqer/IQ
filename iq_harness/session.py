from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Protocol

from .types import Message


class Session(Protocol):
    def load(self) -> list[Message]:
        ...

    def append(self, message: Message) -> None:
        ...


class InMemorySession:
    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._lock = RLock()

    def load(self) -> list[Message]:
        with self._lock:
            return list(self._messages)

    def append(self, message: Message) -> None:
        with self._lock:
            self._messages.append(message)


class JsonlSession:
    """Small durable session store; one JSON object per message."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def load(self) -> list[Message]:
        if not self.path.exists():
            return []
        with self._lock, self.path.open("r", encoding="utf-8") as handle:
            return [Message(**json.loads(line)) for line in handle if line.strip()]

    def append(self, message: Message) -> None:
        record = {
            "role": message.role,
            "content": message.content,
            "name": message.name,
            "tool_call_id": message.tool_call_id,
        }
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
