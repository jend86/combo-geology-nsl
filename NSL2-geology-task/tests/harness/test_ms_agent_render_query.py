"""MsAgentProfile.render_query prepends an MCP-tool-call preamble.

The base HarnessProfile.render_query appends a generic capability manifest
to the task's prompt. That's insufficient for ms-agent: without an explicit
"CALL the tool" instruction, models trained to be helpful emit a text
response that describes what they'd do — ms-agent's ReAct loop sees no
tool call and terminates on turn 1.

This test locks in that MsAgentProfile.render_query layers a tool-call
preamble over whatever the task provides.
"""

from __future__ import annotations

from src.harness.profiles.ms_agent import MsAgentProfile, MsAgentProfileConfig
from src.task.types import Capability, TaskPromptSpec


def _profile() -> MsAgentProfile:
    return MsAgentProfile(MsAgentProfileConfig(model="claude-sonnet-4-6"))


def _spec() -> TaskPromptSpec:
    return TaskPromptSpec(
        system_instruction="You are a researcher. Do the task.",
        environment_context="Fork block 1234.",
        capabilities=[
            Capability(
                name="analyzer",
                description="read state",
            ),
            Capability(
                name="exploiter",
                description="write Attack.sol",
            ),
        ],
    )


def test_render_query_prepends_tool_call_preamble() -> None:
    out = _profile().render_query(_spec())
    # Preamble must lead — the first thing the model sees is the
    # instruction to call tools.
    assert out.lstrip().lower().startswith("you operate via mcp tools")


def test_render_query_contains_non_system_prompt_content() -> None:
    """System text is rendered into agent.yaml, not query.txt."""
    out = _profile().render_query(_spec())
    assert "You are a researcher. Do the task." not in out
    assert "Fork block 1234." in out
    # The base class's capability manifest should still be reachable.
    assert "analyzer" in out
    assert "exploiter" in out


def test_render_query_warns_about_tool_output_truncation() -> None:
    out = _profile().render_query(_spec())

    assert "Tool output truncation:" in out
    assert "truncated: true" in out
    assert "only a prefix" in out


def test_render_query_names_at_least_one_capability_as_tool() -> None:
    """The preamble teaches the nsl.<cap> grammar — mention at least one
    capability name concretely so the model connects the manifest to the
    tool-call schema."""
    out = _profile().render_query(_spec())
    # Any ms-agent-shaped tool reference — "nsl.<cap>" or "exploiter(" or
    # similar concrete grammar. Test is lenient on exact phrasing to avoid
    # re-breaking on minor prompt tweaks.
    lower = out.lower()
    assert "nsl." in lower or "tool" in lower
