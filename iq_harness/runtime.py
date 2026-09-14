from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping
from uuid import uuid4

from .guardrails import Guardrail
from .model import ModelBackend
from .session import Session
from .skills import SkillCatalog, SkillError
from .tools import ApprovalHandler, ToolError, ToolRegistry
from .types import AgentSpec, Message, ModelRequest, RunResult, ToolCall, ToolDescriptor, TraceEvent


TraceHook = Callable[[TraceEvent], None]


_SKILL_TOOL = ToolDescriptor(
    "skills.load",
    "Load the full instructions for an available Agent Skill when it is relevant to the task.",
    {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    },
)
_SKILL_READ_TOOL = ToolDescriptor(
    "skills.read",
    "Read a file inside an already activated Agent Skill, such as references or assets.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["name", "path"],
        "additionalProperties": False,
    },
)
_DELEGATE_TOOL = ToolDescriptor(
    "agent.delegate",
    "Run a bounded specialist subagent and return its result while the current agent remains in control.",
    {
        "type": "object",
        "properties": {
            "agent": {"type": "string"},
            "input": {"type": "string"},
        },
        "required": ["agent", "input"],
        "additionalProperties": False,
    },
)
_HANDOFF_TOOL = ToolDescriptor(
    "agent.handoff",
    "Transfer control of the active task to a specialist agent.",
    {
        "type": "object",
        "properties": {
            "agent": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["agent"],
        "additionalProperties": False,
    },
)


class AgentRuntimeError(RuntimeError):
    pass


@dataclass
class _RunState:
    run_id: str
    parent_run_id: str | None
    messages: list[Message]
    current_agent: str
    depth: int
    activated_skills: dict[str, set[str]] = field(default_factory=dict)
    turns: int = 0


class AgentRuntime:
    """Model-agnostic agent loop with skills, tools, delegation, handoffs, memory and traces."""

    def __init__(
        self,
        model: ModelBackend,
        agents: Iterable[AgentSpec],
        *,
        tools: ToolRegistry | None = None,
        skills: SkillCatalog | None = None,
        approval: ApprovalHandler | None = None,
        hooks: Iterable[TraceHook] = (),
        guardrails: Iterable[Guardrail] = (),
        max_delegation_depth: int = 4,
    ) -> None:
        self.model = model
        self.agents = {agent.name: agent for agent in agents}
        if not self.agents:
            raise AgentRuntimeError("at least one agent is required")
        self.tools = tools or ToolRegistry()
        self.skills = skills or SkillCatalog()
        self.approval = approval
        self.hooks = tuple(hooks)
        self.guardrails = tuple(guardrails)
        self.max_delegation_depth = max_delegation_depth
        self._validate_agents()

    def _validate_agents(self) -> None:
        for agent in self.agents.values():
            if agent.max_turns <= 0:
                raise AgentRuntimeError(f"agent {agent.name}: max_turns must be positive")
            for target in (*agent.delegates, *agent.handoffs):
                if target not in self.agents:
                    raise AgentRuntimeError(f"agent {agent.name}: unknown target {target}")
            for tool in agent.tools:
                self.tools.get(tool)
            if agent.skills is not None:
                for skill in agent.skills:
                    self.skills.get(skill)

    def _trace(self, state: _RunState, event: str, payload: Mapping[str, object]) -> None:
        item = TraceEvent(event, state.run_id, state.current_agent, payload, state.parent_run_id)
        for hook in self.hooks:
            hook(item)

    def _enforce_guardrails(
        self,
        phase: str,
        agent: str,
        value: object,
        *,
        call: ToolCall | None = None,
    ) -> None:
        for guardrail in self.guardrails:
            if phase == "input":
                decision = guardrail.check_input(agent, value)
            elif phase == "tool":
                decision = guardrail.check_tool(agent, value)
            elif phase == "tool_result":
                if call is None:
                    raise AgentRuntimeError("tool_result guardrail requires a tool call")
                decision = guardrail.check_tool_result(agent, call, str(value))
            elif phase == "output":
                decision = guardrail.check_output(agent, str(value))
            else:
                raise AgentRuntimeError(f"unknown guardrail phase: {phase}")
            if not decision.allowed:
                raise AgentRuntimeError(decision.reason or f"{phase} guardrail blocked execution")

    def _system_prompt(self, agent: AgentSpec, state: _RunState) -> str:
        parts = [agent.instructions.strip()]
        allowed_skills = self.skills.names() if agent.skills is None else agent.skills
        if allowed_skills:
            parts.extend([
                "\nAgent Skills are available. Load a skill only when relevant, using skills.load.",
                self.skills.catalog_text(allowed_skills),
            ])
        active = state.activated_skills.get(agent.name, set())
        for name in sorted(active):
            skill = self.skills.get(name)
            parts.append(f"\n<active_skill name=\"{name}\">\n{skill.body}\n</active_skill>")
        if agent.delegates:
            parts.append("\nDelegatable specialists:\n" + "\n".join(
                f"- {name}: {self.agents[name].description or self.agents[name].instructions.splitlines()[0]}"
                for name in agent.delegates
            ))
        if agent.handoffs:
            parts.append("\nHandoff specialists:\n" + "\n".join(
                f"- {name}: {self.agents[name].description or self.agents[name].instructions.splitlines()[0]}"
                for name in agent.handoffs
            ))
        return "\n".join(parts).strip()

    def _descriptors(self, agent: AgentSpec) -> tuple[ToolDescriptor, ...]:
        items = list(self.tools.descriptors(agent.tools))
        enabled_skills = self.skills.names() if agent.skills is None else agent.skills
        if enabled_skills:
            items.extend((_SKILL_TOOL, _SKILL_READ_TOOL))
        if agent.delegates:
            items.append(_DELEGATE_TOOL)
        if agent.handoffs:
            items.append(_HANDOFF_TOOL)
        return tuple(items)

    def _allowed_skill(self, agent: AgentSpec, name: str) -> bool:
        enabled = self.skills.names() if agent.skills is None else agent.skills
        return name in enabled

    def _preapproved_tools(self, state: _RunState, agent: AgentSpec) -> set[str]:
        allowed: set[str] = set()
        for name in state.activated_skills.get(agent.name, set()):
            allowed.update(self.skills.get(name).allowed_tools)
        return allowed

    def _delegate(self, parent: _RunState, target: str, user_input: str) -> str:
        if parent.depth >= self.max_delegation_depth:
            raise AgentRuntimeError("maximum delegation depth exceeded")
        child = _RunState(
            run_id=str(uuid4()),
            parent_run_id=parent.run_id,
            messages=[Message("user", user_input)],
            current_agent=target,
            depth=parent.depth + 1,
        )
        self._trace(parent, "delegate.start", {"target": target, "child_run_id": child.run_id})
        result = self._drive(child, persist=None)
        self._trace(parent, "delegate.end", {"target": target, "child_run_id": child.run_id})
        return result.output

    def _execute_delegate_batch(self, state: _RunState, agent: AgentSpec, calls: list[ToolCall]) -> dict[str, str]:
        def execute(call: ToolCall) -> tuple[str, str]:
            target = str(call.arguments.get("agent", ""))
            if target not in agent.delegates:
                raise AgentRuntimeError(f"agent {agent.name} cannot delegate to {target}")
            user_input = str(call.arguments.get("input", ""))
            return call.id, self._delegate(state, target, user_input)

        workers = min(len(calls), 8)
        if workers <= 1:
            return dict(execute(call) for call in calls)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="iq-subagent") as pool:
            return dict(pool.map(execute, calls))

    def _drive(self, state: _RunState, persist: Session | None) -> RunResult:
        while True:
            agent = self.agents[state.current_agent]
            if state.turns >= agent.max_turns:
                raise AgentRuntimeError(f"agent {agent.name} exceeded max_turns={agent.max_turns}")

            request = ModelRequest(
                system_prompt=self._system_prompt(agent, state),
                messages=tuple(state.messages),
                tools=self._descriptors(agent),
            )
            self._trace(state, "model.request", {"turn": state.turns + 1, "tool_count": len(request.tools)})
            turn = self.model.generate(request)
            state.turns += 1
            self._trace(state, "model.response", {"tool_calls": [call.name for call in turn.tool_calls]})

            if not turn.tool_calls:
                self._enforce_guardrails("output", agent.name, turn.content)
                if turn.content:
                    message = Message("assistant", turn.content, name=agent.name)
                    state.messages.append(message)
                    if persist:
                        persist.append(message)
                return RunResult(turn.content, agent.name, state.run_id, tuple(state.messages), state.turns)

            if turn.content:
                message = Message("assistant", turn.content, name=agent.name)
                state.messages.append(message)
                if persist:
                    persist.append(message)

            delegate_results: dict[str, str] = {}
            valid_delegate_calls: list[ToolCall] = []
            for call in turn.tool_calls:
                if call.name != "agent.delegate":
                    continue
                try:
                    self._enforce_guardrails("tool", agent.name, call)
                    target = str(call.arguments.get("agent", ""))
                    if target not in agent.delegates:
                        raise AgentRuntimeError(f"agent {agent.name} cannot delegate to {target}")
                    valid_delegate_calls.append(call)
                except AgentRuntimeError as exc:
                    delegate_results[call.id] = f"ERROR: {exc}"
            if valid_delegate_calls:
                delegate_results.update(self._execute_delegate_batch(state, agent, valid_delegate_calls))
            handoff_occurred = False

            for call in turn.tool_calls:
                self._trace(state, "tool.call", {"name": call.name, "id": call.id})
                try:
                    if call.name != "agent.delegate":
                        self._enforce_guardrails("tool", agent.name, call)
                    if call.name == "skills.load":
                        name = str(call.arguments.get("name", ""))
                        if not self._allowed_skill(agent, name):
                            raise SkillError(f"skill {name!r} is not enabled for agent {agent.name}")
                        self.skills.get(name)
                        state.activated_skills.setdefault(agent.name, set()).add(name)
                        result = f"Activated skill {name}. Its instructions are now in the agent context."
                    elif call.name == "skills.read":
                        name = str(call.arguments.get("name", ""))
                        path = str(call.arguments.get("path", ""))
                        if name not in state.activated_skills.get(agent.name, set()):
                            raise SkillError(f"skill {name!r} must be activated before reading resources")
                        result = self.skills.get(name).read_resource(path)
                    elif call.name == "agent.delegate":
                        result = delegate_results[call.id]
                    elif call.name == "agent.handoff":
                        target = str(call.arguments.get("agent", ""))
                        if target not in agent.handoffs:
                            raise AgentRuntimeError(f"agent {agent.name} cannot hand off to {target}")
                        result = f"Control transferred from {agent.name} to {target}."
                        state.current_agent = target
                        handoff_occurred = True
                    else:
                        result = self.tools.execute(
                            call,
                            approval=self.approval,
                            preapproved=self._preapproved_tools(state, agent),
                        )
                except (SkillError, ToolError, AgentRuntimeError) as exc:
                    result = f"ERROR: {exc}"
                    self._trace(state, "tool.error", {"name": call.name, "error": str(exc)})

                try:
                    self._enforce_guardrails("tool_result", agent.name, result, call=call)
                except AgentRuntimeError as exc:
                    result = f"ERROR: tool result blocked: {exc}"
                    self._trace(state, "tool_result.blocked", {"name": call.name, "error": str(exc)})

                tool_message = Message("tool", result, name=call.name, tool_call_id=call.id)
                state.messages.append(tool_message)
                if persist:
                    persist.append(tool_message)
                self._trace(state, "tool.result", {"name": call.name, "id": call.id})
                if handoff_occurred:
                    break

    def run(self, agent_name: str, user_input: str, *, session: Session | None = None) -> RunResult:
        if agent_name not in self.agents:
            raise AgentRuntimeError(f"unknown agent: {agent_name}")
        history = session.load() if session else []
        user_message = Message("user", user_input)
        history.append(user_message)
        self._enforce_guardrails("input", agent_name, tuple(history))
        if session:
            session.append(user_message)
        state = _RunState(
            run_id=str(uuid4()),
            parent_run_id=None,
            messages=history,
            current_agent=agent_name,
            depth=0,
        )
        self._trace(state, "run.start", {"input": user_input})
        result = self._drive(state, persist=session)
        self._trace(state, "run.end", {"last_agent": result.last_agent, "turns": result.turns})
        return result
