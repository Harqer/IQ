from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from iq_harness import (
    AgentRuntime,
    AgentSpec,
    Guardrail,
    GuardrailDecision,
    InMemorySession,
    ModelRequest,
    ModelTurn,
    SkillCatalog,
    Tool,
    ToolCall,
    ToolRegistry,
)


class ScriptedModel:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = list(turns)
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        if not self.turns:
            raise AssertionError("unexpected model call")
        return self.turns.pop(0)


class HarnessTests(unittest.TestCase):
    def test_skill_progressive_disclosure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "skills"
            skill = root / "repo-audit"
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text(
                "---\nname: repo-audit\ndescription: Audit repositories. Use for code review.\n"
                "allowed-tools: repo.read\n---\nAlways inspect tests before editing.\n",
                encoding="utf-8",
            )
            catalog = SkillCatalog([root])
            model = ScriptedModel([
                ModelTurn(tool_calls=[ToolCall("1", "skills.load", {"name": "repo-audit"})]),
                ModelTurn(content="done"),
            ])
            runtime = AgentRuntime(model, [AgentSpec("main", "You are IQ.", skills=("repo-audit",))], skills=catalog)
            result = runtime.run("main", "audit this repo")
            self.assertEqual(result.output, "done")
            self.assertIn("repo-audit", model.requests[0].system_prompt)
            self.assertNotIn("Always inspect tests before editing.", model.requests[0].system_prompt)
            self.assertIn("Always inspect tests before editing.", model.requests[1].system_prompt)

    def test_tool_execution_and_session_memory(self) -> None:
        registry = ToolRegistry()
        registry.register(Tool(
            "math.add",
            "Add two integers.",
            {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            lambda args: str(args["a"] + args["b"]),
        ))
        model = ScriptedModel([
            ModelTurn(tool_calls=[ToolCall("1", "math.add", {"a": 2, "b": 3})]),
            ModelTurn(content="5"),
        ])
        session = InMemorySession()
        runtime = AgentRuntime(model, [AgentSpec("main", "Use tools.", tools=("math.add",))], tools=registry)
        result = runtime.run("main", "2+3", session=session)
        self.assertEqual(result.output, "5")
        self.assertTrue(any(message.role == "tool" and message.content == "5" for message in session.load()))

    def test_manager_delegation(self) -> None:
        class RoutingModel:
            def generate(self, request: ModelRequest) -> ModelTurn:
                if "You are the manager" in request.system_prompt:
                    if not any(m.role == "tool" and m.name == "agent.delegate" for m in request.messages):
                        return ModelTurn(tool_calls=[ToolCall("d1", "agent.delegate", {"agent": "researcher", "input": "research x"})])
                    return ModelTurn(content="combined")
                return ModelTurn(content="research result")

        runtime = AgentRuntime(
            RoutingModel(),
            [
                AgentSpec("manager", "You are the manager.", delegates=("researcher",)),
                AgentSpec("researcher", "You are the researcher."),
            ],
        )
        result = runtime.run("manager", "do it")
        self.assertEqual(result.output, "combined")
        self.assertEqual(result.last_agent, "manager")

    def test_handoff_transfers_control(self) -> None:
        class HandoffModel:
            def generate(self, request: ModelRequest) -> ModelTurn:
                if "triage" in request.system_prompt:
                    return ModelTurn(tool_calls=[ToolCall("h1", "agent.handoff", {"agent": "coder", "reason": "coding"})])
                return ModelTurn(content="coded")

        runtime = AgentRuntime(
            HandoffModel(),
            [
                AgentSpec("triage", "triage", handoffs=("coder",)),
                AgentSpec("coder", "coder"),
            ],
        )
        result = runtime.run("triage", "fix it")
        self.assertEqual(result.output, "coded")
        self.assertEqual(result.last_agent, "coder")

    def test_skill_resource_requires_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "skills"
            skill = root / "repo-audit"
            (skill / "references").mkdir(parents=True)
            (skill / "SKILL.md").write_text(
                "---\nname: repo-audit\ndescription: Audit repositories. Use for code review.\n---\nRead references/checks.md.\n",
                encoding="utf-8",
            )
            (skill / "references" / "checks.md").write_text("CHECKS", encoding="utf-8")
            catalog = SkillCatalog([root])
            model = ScriptedModel([
                ModelTurn(tool_calls=[ToolCall("1", "skills.load", {"name": "repo-audit"})]),
                ModelTurn(tool_calls=[ToolCall("2", "skills.read", {"name": "repo-audit", "path": "references/checks.md"})]),
                ModelTurn(content="done"),
            ])
            runtime = AgentRuntime(model, [AgentSpec("main", "You are IQ.", skills=("repo-audit",))], skills=catalog)
            result = runtime.run("main", "audit")
            self.assertEqual(result.output, "done")
            self.assertTrue(any(m.role == "tool" and m.content == "CHECKS" for m in result.messages))

    def test_output_guardrail_can_block(self) -> None:
        class BlockOutput(Guardrail):
            def check_output(self, agent: str, output: str) -> GuardrailDecision:
                return GuardrailDecision(False, "blocked")

        runtime = AgentRuntime(
            ScriptedModel([ModelTurn(content="unsafe")]),
            [AgentSpec("main", "test", skills=())],
            guardrails=[BlockOutput()],
        )
        with self.assertRaisesRegex(Exception, "blocked"):
            runtime.run("main", "go")

    def test_delegate_guardrail_blocks_before_subagent_runs(self) -> None:
        calls = {"child": 0}

        class RoutingModel:
            def generate(self, request: ModelRequest) -> ModelTurn:
                if "manager" in request.system_prompt:
                    if any(m.role == "tool" for m in request.messages):
                        return ModelTurn(content="finished")
                    return ModelTurn(tool_calls=[ToolCall("d1", "agent.delegate", {"agent": "worker", "input": "work"})])
                calls["child"] += 1
                return ModelTurn(content="child")

        class BlockDelegate(Guardrail):
            def check_tool(self, agent: str, call: ToolCall) -> GuardrailDecision:
                if call.name == "agent.delegate":
                    return GuardrailDecision(False, "no delegation")
                return GuardrailDecision(True)

        runtime = AgentRuntime(
            RoutingModel(),
            [
                AgentSpec("manager", "manager", delegates=("worker",), skills=()),
                AgentSpec("worker", "worker", skills=()),
            ],
            guardrails=[BlockDelegate()],
        )
        result = runtime.run("manager", "go")
        self.assertEqual(result.output, "finished")
        self.assertEqual(calls["child"], 0)

    def test_tool_result_guardrail_screens_context(self) -> None:
        registry = ToolRegistry()
        registry.register(Tool("external.read", "Read external content.", {"type": "object"}, lambda _: "INJECTION"))

        class ResultModel:
            def generate(self, request: ModelRequest) -> ModelTurn:
                if not any(m.role == "tool" for m in request.messages):
                    return ModelTurn(tool_calls=[ToolCall("t1", "external.read", {})])
                tool_message = next(m for m in request.messages if m.role == "tool")
                self.seen = tool_message.content
                return ModelTurn(content="done")

        class BlockInjection(Guardrail):
            def check_tool_result(self, agent: str, call: ToolCall, result: str) -> GuardrailDecision:
                if "INJECTION" in result:
                    return GuardrailDecision(False, "suspect external content")
                return GuardrailDecision(True)

        model = ResultModel()
        runtime = AgentRuntime(
            model,
            [AgentSpec("main", "main", tools=("external.read",), skills=())],
            tools=registry,
            guardrails=[BlockInjection()],
        )
        runtime.run("main", "read")
        self.assertNotIn("INJECTION", model.seen)
        self.assertIn("blocked", model.seen)


if __name__ == "__main__":
    unittest.main()
