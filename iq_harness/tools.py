from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol

from .types import ToolCall, ToolDescriptor


ToolHandler = Callable[[Mapping[str, Any]], Any]
ApprovalHandler = Callable[[ToolCall, "Tool"], bool]


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    handler: ToolHandler
    requires_approval: bool = False

    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(self.name, self.description, self.input_schema)


class ToolProvider(Protocol):
    """Integration point for MCP/connectors/plugin-like tool providers."""

    def tools(self) -> Iterable[Tool]:
        ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def register_provider(self, provider: ToolProvider) -> None:
        for tool in provider.tools():
            self.register(tool)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"unknown tool: {name}") from exc

    def descriptors(self, names: Iterable[str]) -> tuple[ToolDescriptor, ...]:
        return tuple(self.get(name).descriptor() for name in names)

    @staticmethod
    def _validate(schema: Mapping[str, Any], args: Mapping[str, Any]) -> None:
        if schema.get("type") not in (None, "object"):
            raise ToolError("top-level tool schema must be an object")
        required = schema.get("required", [])
        for key in required:
            if key not in args:
                raise ToolError(f"missing required tool argument: {key}")
        properties = schema.get("properties", {})
        expected_types = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        for key, value in args.items():
            spec = properties.get(key)
            if not spec:
                if schema.get("additionalProperties") is False:
                    raise ToolError(f"unexpected tool argument: {key}")
                continue
            kind = spec.get("type")
            expected = expected_types.get(kind)
            if kind in ("integer", "number") and isinstance(value, bool):
                raise ToolError(f"invalid type for {key}: expected {kind}")
            if expected and not isinstance(value, expected):
                raise ToolError(f"invalid type for {key}: expected {kind}")

    def execute(
        self,
        call: ToolCall,
        *,
        approval: ApprovalHandler | None = None,
        preapproved: Iterable[str] = (),
    ) -> str:
        tool = self.get(call.name)
        self._validate(tool.input_schema, call.arguments)
        if tool.requires_approval and tool.name not in set(preapproved):
            if approval is None or not approval(call, tool):
                raise ToolError(f"approval denied for tool: {tool.name}")
        result = tool.handler(call.arguments)
        return result if isinstance(result, str) else repr(result)
