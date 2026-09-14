from __future__ import annotations

from dataclasses import dataclass

from .types import Message, ToolCall


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    reason: str = ""


class Guardrail:
    """Override the checks needed by a deployment. Default is allow."""

    def check_input(self, agent: str, messages: tuple[Message, ...]) -> GuardrailDecision:
        return GuardrailDecision(True)

    def check_tool(self, agent: str, call: ToolCall) -> GuardrailDecision:
        return GuardrailDecision(True)

    def check_output(self, agent: str, output: str) -> GuardrailDecision:
        return GuardrailDecision(True)
