from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


JSON = Any


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, JSON] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolDescriptor:
    name: str
    description: str
    input_schema: Mapping[str, JSON]


@dataclass(frozen=True)
class ModelRequest:
    system_prompt: str
    messages: Sequence[Message]
    tools: Sequence[ToolDescriptor]


@dataclass(frozen=True)
class ModelTurn:
    content: str = ""
    tool_calls: Sequence[ToolCall] = field(default_factory=tuple)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    instructions: str
    tools: tuple[str, ...] = ()
    skills: tuple[str, ...] | None = None
    delegates: tuple[str, ...] = ()
    handoffs: tuple[str, ...] = ()
    description: str = ""
    max_turns: int = 24


@dataclass(frozen=True)
class RunResult:
    output: str
    last_agent: str
    run_id: str
    messages: tuple[Message, ...]
    turns: int


@dataclass(frozen=True)
class TraceEvent:
    event: str
    run_id: str
    agent: str
    payload: Mapping[str, JSON]
    parent_run_id: str | None = None
