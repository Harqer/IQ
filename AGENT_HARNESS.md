# IQ Agent Harness

IQ's model architecture and its agent runtime are separate systems. The model produces language/tool decisions; the harness owns tools, skills, memory, delegation, handoffs, approvals, and traces.

## Supported primitives

- Agent Skills open format: `SKILL.md`, `scripts/`, `references/`, `assets/`
- Progressive disclosure: catalog metadata first; skill instructions only after activation
- Function tools with JSON-schema inputs
- External tool providers for MCP/connector/plugin adapters
- Manager-style specialist delegation (`agent.delegate`)
- Conversation handoffs (`agent.handoff`)
- Parallel execution of multiple subagent delegations from one model turn
- In-memory and JSONL sessions
- Approval gates for side-effecting tools
- Trace hooks for run/tool/delegation events
- Bounded turns and bounded delegation depth

## Runtime boundary

Implement `iq_harness.ModelBackend` for the IQ inference engine:

```python
class IQBackend:
    def generate(self, request: ModelRequest) -> ModelTurn:
        # Convert request.system_prompt/messages/tools to the IQ inference format.
        # Parse IQ's structured tool calls back into ModelTurn.
        ...
```

The harness intentionally does not embed OpenAI or Anthropic APIs. It can wrap IQ directly and keeps the orchestration layer model-provider independent.

## Skill layout

```text
skills/
└── repo-audit/
    ├── SKILL.md
    ├── scripts/
    ├── references/
    └── assets/
```

Minimum `SKILL.md`:

```markdown
---
name: repo-audit
description: Audit a repository for correctness and security. Use for code review and refactoring tasks.
---

Inspect the repository and tests before editing. Validate changes before reporting completion.
```

At session start only the skill name and description enter the model context. The full body is inserted after the model invokes `skills.load`.

## Agent patterns

Use delegation when the parent should keep control and combine specialist outputs. Use handoffs when a specialist should become the active agent. Do not spawn subagents for simple sequential operations.

```python
manager = AgentSpec(
    name="manager",
    instructions="Plan the task and integrate specialist results.",
    delegates=("researcher", "coder"),
)
```

## Tool safety

Mark tools with external side effects `requires_approval=True`. The runtime denies those calls unless an approval callback approves them, except for exact tools explicitly pre-approved by an activated skill's `allowed-tools` field.

Repository/file/web/MCP implementations should be registered as `ToolProvider`s; tool credentials stay outside prompts and source control.

## Tests

```bash
python -m unittest discover -s tests -p 'test_agent_harness.py'
```
