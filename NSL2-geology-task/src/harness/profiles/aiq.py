"""AiqProfile - adapter for NVIDIA NeMo Agent Toolkit (NAT).

The user-facing profile name remains ``aiq`` for continuity with the design
request, but the container uses the current NAT Python API. The profile renders
YAML consumed by ``nat.utils.run_workflow`` plus a small ``workflow.json``
manifest when the task declares multiple workflow steps.

HITL is intentionally unsupported: this profile defaults to built-in NAT agent
types that run non-interactively, and the wrapper does not register a human
input callback. A workflow that invokes HITL should fail loudly rather than
block a data-generation worker.
"""

from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.harness.tool_contract import ToolCallContractProbe
from src.harness.context import project_step_constraints
from src.harness.profiles.base import HarnessProfile
from src.task.types import (
    Capability,
    CapabilityInvocation,
    CapabilityResult,
    EpisodeArtifacts,
    EpisodeConstraints,
    TaskPromptSpec,
    Workflow,
    WorkflowStep,
)


class AiqProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    agent_type: str = "tool_calling_agent"
    max_iterations: int = 12
    tool_call_timeout_s: int = 90
    output_dir: str = "/work/output"
    transcript_filename: str = "trace.jsonl"
    function_group_name: str = "capabilities"
    force_tool_choice: bool = False
    extra_yaml: dict[str, Any] = Field(default_factory=dict)


_TOOL_CALL_USER_PREAMBLE = (
    "To take any action, CALL the appropriate tool from the schema you "
    "have been given - do not describe what you would do in plain text. "
    "Each capability listed below is exposed as one of those tools; "
    "pass your input as the tool arguments."
)


def _merge_dicts(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _safe_config_filename(step_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", step_name).strip("._")
    if not safe:
        raise ValueError("workflow step name cannot render to an empty filename")
    return f"{safe}.yaml"


def _json_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _transcript_filename_for_config(config_filename: str) -> str:
    return f"{Path(config_filename).stem}.jsonl"


def _contract_probe_tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Validate that the backend emits structured tool calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Set this to ok.",
                    }
                },
                "required": ["message"],
            },
        },
    }


def _relative_output_dir(output_dir: str) -> PurePosixPath | None:
    path = PurePosixPath(posixpath.normpath(output_dir))
    if path.is_absolute():
        try:
            path = path.relative_to("/work")
        except ValueError:
            return None
    if ".." in path.parts:
        return None
    return path


class AiqProfile(HarnessProfile):
    name = "aiq"
    profile_config_class = AiqProfileConfig

    config: AiqProfileConfig

    def __init__(self, config: AiqProfileConfig) -> None:
        super().__init__(config)

    # --- Query rendering ---

    def render_query(
        self,
        prompt_spec: TaskPromptSpec,
        *,
        constraints: EpisodeConstraints | None = None,
    ) -> str:
        parts: list[str] = []
        if prompt_spec.capabilities:
            parts.append(_TOOL_CALL_USER_PREAMBLE)
        body = self.render_query_without_system(prompt_spec, constraints=constraints)
        if body:
            parts.append(body)
        return "\n\n".join(parts)

    def render_query_without_system(
        self,
        prompt_spec: TaskPromptSpec,
        *,
        constraints: EpisodeConstraints | None = None,
    ) -> str:
        parts: list[str] = []
        if prompt_spec.environment_context:
            parts.append(prompt_spec.environment_context)
        if prompt_spec.capabilities:
            group = self.config.function_group_name
            caps = "\n".join(
                f"- {group}__{c.name}: {c.description}" for c in prompt_spec.capabilities
            )
            parts.append(
                "Capabilities (exposed as MCP tools under "
                f"function_groups.{group}; the LLM-visible names are prefixed):\n"
                + caps
            )
        constraints_block = self.render_constraints_block(constraints)
        if constraints_block:
            parts.append(constraints_block)
        return "\n\n".join(parts)

    # --- Container launch ---

    def default_args(self, scratch: Path) -> list[str]:
        return ["python", "/opt/nsl/run.py"]

    def env(self, scratch: Path) -> dict[str, str]:
        return {"TOOL_CALL_TIMEOUT": str(self.config.tool_call_timeout_s)}

    def supports_native_workflow(self, workflow: Workflow) -> bool:
        return True

    def tool_call_contract_probe(self) -> ToolCallContractProbe:
        name = f"{self.config.function_group_name}__contract_probe"
        headers = (
            {"X-NSL-Tool-Choice": "required"}
            if self.config.force_tool_choice
            else {}
        )
        return ToolCallContractProbe(
            tools=[_contract_probe_tool(name)],
            messages=[
                {
                    "role": "system",
                    "content": "You are a deterministic tool-call contract validator.",
                },
                {
                    "role": "user",
                    "content": (
                        "You are a contract-validation bot. You MUST call exactly "
                        f"the provided tool named {name} and set message to \"ok\". "
                        "Do not answer in plain text."
                    ),
                },
            ],
            expected_tool_name=name,
            expected_argument_keys={"message"},
            expected_arguments={"message": "ok"},
            tool_choice="auto",
            headers=headers,
        )

    # --- Config rendering ---

    def render_config(
        self,
        *,
        scratch: Path,
        query: str,
        capabilities: list[Capability],
        inference_url: str,
        mcp_url: str,
        token: str,
        prompt_spec: TaskPromptSpec | None = None,
        workflow: Workflow | None = None,
        constraints: EpisodeConstraints | None = None,
    ) -> None:
        output_rel = _relative_output_dir(self.config.output_dir)
        if output_rel is not None:
            scratch.joinpath(*output_rel.parts).mkdir(parents=True, exist_ok=True)

        if workflow is not None:
            self._render_workflow(
                scratch=scratch,
                workflow=workflow,
                prompt_spec=prompt_spec,
                capabilities=capabilities,
                inference_url=inference_url,
                mcp_url=mcp_url,
                token=token,
                constraints=constraints,
            )
            return

        agent_yaml = self._agent_yaml(
            inference_url=inference_url,
            mcp_url=mcp_url,
            token=token,
            prompt_spec=prompt_spec,
            capabilities=capabilities,
            step=None,
            transcript_filename=self.config.transcript_filename,
        )
        (scratch / "agent.yaml").write_text(yaml.safe_dump(agent_yaml))
        (scratch / "query.txt").write_text(query)

    def _render_workflow(
        self,
        *,
        scratch: Path,
        workflow: Workflow,
        prompt_spec: TaskPromptSpec | None,
        capabilities: list[Capability],
        inference_url: str,
        mcp_url: str,
        token: str,
        constraints: EpisodeConstraints | None = None,
    ) -> None:
        ordered = workflow.topological_order()
        filenames = {step.name: _safe_config_filename(step.name) for step in ordered}
        if len(set(filenames.values())) != len(filenames):
            raise ValueError("workflow step names collide after filename sanitization")

        manifest_steps: list[dict[str, Any]] = []
        for step in ordered:
            filename = filenames[step.name]
            agent_yaml = self._agent_yaml(
                inference_url=inference_url,
                mcp_url=mcp_url,
                token=token,
                prompt_spec=prompt_spec,
                capabilities=capabilities,
                step=step,
                transcript_filename=filename.rsplit(".", 1)[0] + ".jsonl",
            )
            (scratch / filename).write_text(yaml.safe_dump(agent_yaml))
            step_constraints = project_step_constraints(
                constraints,
                step.name,
                has_capabilities=self._step_has_tools(step, capabilities),
            )
            manifest_step = {
                "name": step.name,
                "config": filename,
                "prompt": self._render_step_prompt(
                    step,
                    capabilities=capabilities,
                    constraints=step_constraints,
                ),
                "inherit_context": step.context_mode == "inherit",
            }
            if self.config.force_tool_choice and self._step_has_tools(
                step, capabilities
            ):
                manifest_step["tool_choice"] = "required"
            manifest_steps.append(manifest_step)
        (scratch / "workflow.json").write_text(json.dumps({"steps": manifest_steps}))

    def _agent_yaml(
        self,
        *,
        inference_url: str,
        mcp_url: str,
        token: str,
        prompt_spec: TaskPromptSpec | None,
        capabilities: list[Capability],
        step: WorkflowStep | None,
        transcript_filename: str,
    ) -> dict[str, Any]:
        group = self.config.function_group_name
        prompt = prompt_spec.system_instruction if prompt_spec is not None else ""
        system_prompt = (
            f"{prompt}\n\n"
            "Tool naming: tools you will see in the function schema are "
            f"registered under the '{group}' function group with NAT's "
            f"'<group>__<tool>' naming, so the tool you would call to run "
            f"Python is named exactly {group}__run_python. Call the "
            "appropriate tool - do not describe what you would do in plain text."
        ).strip()
        custom_headers = {"Authorization": f"Bearer {token}"}
        if step is not None:
            custom_headers["X-NSL-Workflow-Step"] = step.name
        function_group_config: dict[str, Any] = {
            "_type": "mcp_client",
            "server": {
                "transport": "streamable-http",
                "url": mcp_url,
                "custom_headers": custom_headers,
            },
            "tool_call_timeout": self.config.tool_call_timeout_s,
        }
        # When a step declares an explicit capability list, NAT must register
        # those tools in the workflow builder's global function registry under
        # their prefixed names (e.g. capabilities__run_python). That only
        # happens when `include` is set on the function group config — without
        # it, individual `tool_names` lookups fail with
        # `Function ... not found in list of functions`. See
        # `nat.builder.workflow_builder._WorkflowBuilder.add_function_group`,
        # which calls `instance.get_included_functions()` (respects `include`).
        if step is not None and not step.inherit_all_capabilities and step.capabilities:
            function_group_config["include"] = list(step.capabilities)
        agent_yaml: dict[str, Any] = {
            "general": {
                "use_uvloop": True,
                "telemetry": {
                    "tracing": {
                        "local_file": {
                            "_type": "file",
                            "output_path": f"{self.config.output_dir.rstrip('/')}/{transcript_filename}",
                            "project": "nsl-aiq",
                        }
                    }
                },
            },
            "llms": {
                "shim_llm": {
                    "_type": "openai",
                    "base_url": inference_url,
                    "api_key": token,
                    "model_name": self.config.model,
                    "temperature": 0.0,
                    # NSL OpenAiShim rejects `stream=true`. NAT's
                    # tool_calling_agent unconditionally calls `astream` on the
                    # bound LangChain ChatOpenAI client, which would otherwise
                    # send a streaming request on every turn. Setting
                    # `disable_streaming=True` on the LangChain model makes
                    # `astream`/`stream` defer to `ainvoke`/`invoke`, keeping
                    # the wire request non-streaming. NAT's OpenAIModelConfig
                    # is `extra="allow"` and forwards unknown kwargs to
                    # `ChatOpenAI(**config_dict)` via model_dump, so the field
                    # is honoured end-to-end.
                    "disable_streaming": True,
                }
            },
            "function_groups": {group: function_group_config},
            "workflow": {
                "_type": self.config.agent_type,
                "llm_name": "shim_llm",
                "tool_names": self._tool_names(step, capabilities),
                "max_iterations": self.config.max_iterations,
                "system_prompt": system_prompt,
            },
        }
        if step is not None and step.terminator_capabilities:
            group = self.config.function_group_name
            agent_yaml["workflow"]["return_direct"] = [
                f"{group}__{cap}" for cap in step.terminator_capabilities
            ]
        if self.config.extra_yaml:
            agent_yaml = _merge_dicts(agent_yaml, self.config.extra_yaml)
        if step is not None:
            shim_llm = agent_yaml["llms"]["shim_llm"]
            headers = dict(shim_llm.get("default_headers") or {})
            headers["X-NSL-Workflow-Step"] = step.name
            shim_llm["default_headers"] = headers
        if self._force_tool_choice_for(capabilities=capabilities, step=step):
            shim_llm = agent_yaml["llms"]["shim_llm"]
            headers = dict(shim_llm.get("default_headers") or {})
            headers["X-NSL-Tool-Choice"] = "required"
            shim_llm["default_headers"] = headers
        return agent_yaml

    def _tool_names(
        self,
        step: WorkflowStep | None,
        capabilities: list[Capability],
    ) -> list[str]:
        group = self.config.function_group_name
        if step is None or step.inherit_all_capabilities:
            return [group] if capabilities else []
        return [f"{group}__{name}" for name in step.capabilities]

    @staticmethod
    def _step_has_tools(
        step: WorkflowStep,
        capabilities: list[Capability] | None = None,
    ) -> bool:
        if step.inherit_all_capabilities:
            return capabilities is None or bool(capabilities)
        return bool(step.capabilities)

    def _render_step_prompt(
        self,
        step: WorkflowStep,
        *,
        capabilities: list[Capability],
        constraints: EpisodeConstraints | None = None,
    ) -> str:
        parts: list[str] = []
        if not self._step_has_tools(step, capabilities):
            parts.append(step.prompt)
        else:
            parts.append(f"{_TOOL_CALL_USER_PREAMBLE}\n\n{step.prompt}")
        constraints_block = self.render_constraints_block(constraints)
        if constraints_block:
            parts.append(constraints_block)
        return "\n\n".join(parts)

    def _force_tool_choice_for(
        self,
        *,
        capabilities: list[Capability],
        step: WorkflowStep | None,
    ) -> bool:
        if not self.config.force_tool_choice:
            return False
        if step is not None:
            return self._step_has_tools(step, capabilities)
        return bool(capabilities)

    # --- Transcript recovery ---

    def read_transcript(self, scratch: Path) -> dict[str, Any] | None:
        output_rel = _relative_output_dir(self.config.output_dir)
        if output_rel is None:
            return None
        output_dir = scratch.joinpath(*output_rel.parts)
        messages: list[dict[str, Any]] = []
        workflow_steps = self._workflow_transcript_paths(scratch, output_dir)
        workflow_payload: dict[str, dict[str, Any]] = {}
        last_workflow_step: str | None = None
        if output_dir.is_dir():
            jsonl_paths: list[tuple[str | None, Path]]
            if workflow_steps is None:
                jsonl_paths = [
                    (None, path) for path in sorted(output_dir.glob("*.jsonl"))
                ]
            else:
                jsonl_paths = workflow_steps
            for step_name, jsonl in jsonl_paths:
                step_messages, tool_end_count, first_assistant = self._read_jsonl(jsonl)
                messages.extend(step_messages)
                if step_name is not None:
                    if step_messages or tool_end_count or first_assistant is not None:
                        workflow_payload[step_name] = {
                            "messages": step_messages,
                            "tool_end_count": tool_end_count,
                            "first_assistant_content": first_assistant,
                        }
                        last_workflow_step = step_name
        final_path = scratch / "final_answer.txt"
        final_response = final_path.read_text() if final_path.exists() else None
        if not messages and not final_response:
            return None
        transcript: dict[str, Any] = {
            "messages": messages,
            "final_response": final_response,
        }
        if workflow_payload:
            transcript["workflow"] = workflow_payload
        if last_workflow_step is not None:
            transcript["last_workflow_step"] = last_workflow_step
        return transcript

    def _workflow_transcript_paths(
        self,
        scratch: Path,
        output_dir: Path,
    ) -> list[tuple[str, Path]] | None:
        manifest_path = scratch / "workflow.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            return None
        steps = manifest.get("steps")
        if not isinstance(steps, list):
            return None

        paths: list[tuple[str, Path]] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            name = step.get("name")
            config = step.get("config")
            if not isinstance(name, str) or not isinstance(config, str):
                continue
            paths.append((name, output_dir / _transcript_filename_for_config(config)))
        return paths

    def _read_jsonl(self, jsonl: Path) -> tuple[list[dict[str, Any]], int, str | None]:
        messages: list[dict[str, Any]] = []
        tool_end_count = 0
        first_assistant: str | None = None
        if not jsonl.exists():
            return messages, tool_end_count, first_assistant

        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = record.get("payload") or {}
            event_type = payload.get("event_type")
            data = payload.get("data") or {}
            if event_type == "LLM_END":
                content = data.get("output") or ""
                if isinstance(content, str) and content.strip():
                    if first_assistant is None:
                        first_assistant = content
                    messages.append({"role": "assistant", "content": content})
            elif event_type == "TOOL_END":
                tool_end_count += 1
                output = data.get("output")
                if output is not None:
                    messages.append(
                        {
                            "role": "tool",
                            "name": payload.get("name"),
                            "content": _json_content(output),
                        }
                    )
        return messages, tool_end_count, first_assistant

    # --- Artifact reconstruction ---

    def to_artifacts(
        self,
        *,
        transcript: dict[str, Any] | None,
        capability_pairs: list[tuple[CapabilityInvocation, CapabilityResult]],
    ) -> EpisodeArtifacts:
        return EpisodeArtifacts(
            capability_invocations=[pair[0] for pair in capability_pairs],
            capability_results=[pair[1] for pair in capability_pairs],
            final_response=(transcript or {}).get("final_response"),
        )

    def count_llm_turns(self, transcript: dict[str, Any] | None) -> int:
        if not transcript:
            return 0
        return sum(
            1 for m in transcript.get("messages", []) if m.get("role") == "assistant"
        )


__all__ = ["AiqProfile", "AiqProfileConfig"]
