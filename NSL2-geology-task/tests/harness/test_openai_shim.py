"""``OpenAiShim`` exposes ``/v1/chat/completions`` backed by ``TracedGenner``.

Every call hitting the shim must land one entry in
``recorder.inference_records`` with the harness-supplied ``phase`` tag.
Streaming requests are explicitly rejected (v1). Auth is bearer-token on
the ``Authorization`` header.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi.testclient import TestClient
from result import Err, Ok

from src.genner.Base import (
    CONTEXT_OVERFLOW_PREFIX,
    Genner,
    INFERENCE_TIMEOUT_PREFIX,
    INFERENCE_UNAVAILABLE_PREFIX,
)
from src.harness.context_compaction import ContextCompactionSettings
from src.harness.openai_shim import OpenAiShim, _extract_pseudo_tool_calls
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.observability.types import InferenceResult, UsageInfo


class _DummyGenner(Genner):
    def __init__(self, content: str = "hello") -> None:
        super().__init__("dummy")
        self._content = content
        self.call_count = 0
        self.last_messages = None
        self.last_tools = None
        self.last_tool_choice = None
        self._tool_calls = None
        self._stop_reason = "stop"

    def set_tool_response(
        self,
        tool_calls,
        *,
        stop_reason: str = "tool_calls",
        content: str = "",
    ) -> None:
        self._tool_calls = tool_calls
        self._stop_reason = stop_reason
        self._content = content

    def plist_completion(self, messages, *, tools=None, tool_choice=None):
        self.call_count += 1
        self.last_messages = messages
        self.last_tools = tools
        self.last_tool_choice = tool_choice
        return Ok(
            InferenceResult(
                content=self._content,
                usage=UsageInfo(
                    prompt_tokens=len(messages),
                    completion_tokens=4,
                    total_tokens=len(messages) + 4,
                    stop_reason=self._stop_reason,
                ),
                tool_calls=self._tool_calls,
            )
        )

    @staticmethod
    def get_usage_info(response):
        return UsageInfo()


def _build_shim(
    tmp_path: Path,
    *,
    token: str,
    episode_id: str,
    context_compaction: ContextCompactionSettings | None = None,
) -> tuple[OpenAiShim, EventRecorder, _DummyGenner]:
    recorder = EventRecorder(
        episode_id=episode_id,
        output_path=tmp_path / f"{episode_id}.jsonl",
    )
    inner = _DummyGenner("ok")
    traced = TracedGenner(
        inner=inner,
        recorder=recorder,
        cancel_event=threading.Event(),
        episode_id=episode_id,
    )
    shim = OpenAiShim(
        traced,
        token=token,
        episode_id=episode_id,
        recorder=recorder,
        context_compaction=context_compaction,
    )
    return shim, recorder, inner


def _openai_body(stream: bool = False) -> dict:
    return {
        "model": "nsl-test",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
        "stream": stream,
    }


def test_shim_routes_call_to_genner_and_records_inference(tmp_path: Path) -> None:
    shim, recorder, inner = _build_shim(tmp_path, token="abc", episode_id="ep-1")
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer abc"},
    )
    assert response.status_code == 200
    body = response.json()
    # OpenAI-shape response
    assert body["model"] == "nsl-test"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "ok"
    # Inner genner was invoked once; one inference record captured.
    assert inner.call_count == 1
    assert len(recorder.inference_records) == 1
    assert recorder.snapshot_counters()["turns"] == 1


def test_shim_forwards_tools_and_tool_choice(tmp_path: Path) -> None:
    shim, _recorder, inner = _build_shim(tmp_path, token="abc", episode_id="ep-tools")
    client = TestClient(shim.app)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "nsl.analyzer",
                "description": "Analyze state",
                "parameters": {"type": "object"},
            },
        }
    ]
    response = client.post(
        "/v1/chat/completions",
        json={
            **_openai_body(),
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "nsl.analyzer"}},
        },
        headers={"Authorization": "Bearer abc"},
    )
    assert response.status_code == 200
    assert inner.last_tools == tools
    assert inner.last_tool_choice == {
        "type": "function",
        "function": {"name": "nsl.analyzer"},
    }


def test_shim_tool_choice_header_overrides_request(tmp_path: Path) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-tool-choice-header"
    )
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json={**_openai_body(), "tool_choice": "auto"},
        headers={
            "Authorization": "Bearer abc",
            "X-NSL-Tool-Choice": "required",
        },
    )

    assert response.status_code == 200
    assert inner.last_tool_choice == "required"


def test_shim_tool_choice_header_can_force_named_function(tmp_path: Path) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-tool-choice-function-header"
    )
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={
            "Authorization": "Bearer abc",
            "X-NSL-Tool-Choice": "function:nsl.analyzer",
        },
    )

    assert response.status_code == 200
    assert inner.last_tool_choice == {
        "type": "function",
        "function": {"name": "nsl.analyzer"},
    }


def test_shim_preserves_structured_messages_on_input(tmp_path: Path) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-messages"
    )
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "nsl-test",
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "nsl.analyzer",
                                "arguments": '{"command":"pwd"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "nsl.analyzer",
                    "content": "ok",
                },
            ],
        },
        headers={"Authorization": "Bearer abc"},
    )
    assert response.status_code == 200
    assert inner.last_messages is not None
    assistant = inner.last_messages[1]
    tool = inner.last_messages[2]
    tool_calls = assistant.get("tool_calls")
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "nsl.analyzer"
    assert tool.get("tool_call_id") == "call_1"
    assert tool.get("name") == "nsl.analyzer"


def test_shim_returns_tool_calls_in_openai_shape(tmp_path: Path) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-tool-response"
    )
    inner.set_tool_response(
        [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "nsl.analyzer",
                    "arguments": '{"command":"pwd"}',
                },
            }
        ]
    )
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer abc"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] is None
    assert body["choices"][0]["message"]["tool_calls"][0]["id"] == "call_1"
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert shim.last_tool_response_classification is not None
    assert shim.last_tool_response_classification.source == "structured"


def test_shim_rehydrates_pseudo_tool_call_markup(tmp_path: Path) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-pseudo-tool-response"
    )
    inner.set_tool_response(
        None,
        stop_reason="stop",
        content=(
            "I should inspect the contract first.\n"
            "</think>\n\n"
            "<tool_call>\n"
            "<function=nsl---analyzer>\n"
            "<parameter=arguments>\n"
            '{"command":"pwd"}\n'
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
    )
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer abc"},
    )
    assert response.status_code == 200
    body = response.json()
    message = body["choices"][0]["message"]
    assert message["content"] is None
    # ms-agent's ToolManager keys tools as `{server}---{tool}`, so the
    # model's `<function=nsl---analyzer>` is already the canonical
    # dispatch key. Preserve it verbatim — rewriting `---`→`.` here was
    # the cause of run 20260425-2p33ek's universal "Tool name
    # nsl.analyzer not found" failures.
    assert message["tool_calls"][0]["function"]["name"] == "nsl---analyzer"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"command":"pwd"}'
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert shim.last_tool_response_classification is not None
    assert shim.last_tool_response_classification.source == "synthesized"


def test_shim_filters_synthesized_tool_calls_to_declared_tool_names(
    tmp_path: Path,
) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-filtered-pseudo-tool"
    )
    inner.set_tool_response(
        None,
        stop_reason="stop",
        content=(
            "<tool_call>"
            "<name>evil---delete_everything</name>"
            "<arguments><message>ok</message></arguments>"
            "</tool_call>"
        ),
    )
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            **_openai_body(),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "nsl---contract_probe",
                        "description": "allowed",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
        headers={"Authorization": "Bearer abc"},
    )

    assert response.status_code == 200
    body = response.json()
    message = body["choices"][0]["message"]
    assert "tool_calls" not in message
    assert shim.last_tool_response_classification is not None
    assert shim.last_tool_response_classification.source == "missing"


def test_shim_counts_required_tool_choice_missing_response(tmp_path: Path) -> None:
    shim, recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-required-missing"
    )
    inner.set_tool_response(None, stop_reason="stop", content="I will answer directly.")
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            **_openai_body(),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "nsl---contract_probe",
                        "description": "allowed",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
        headers={
            "Authorization": "Bearer abc",
            "X-NSL-Tool-Choice": "required",
        },
    )

    assert response.status_code == 200
    counters = recorder.snapshot_counters()
    assert counters["tool_requests_total"] == 1
    assert counters["tool_responses_missing_total"] == 1


def test_shim_compacts_messages_before_traced_genner(tmp_path: Path) -> None:
    shim, recorder, inner = _build_shim(
        tmp_path,
        token="abc",
        episode_id="ep-compact",
        context_compaction=ContextCompactionSettings(
            enabled=True,
            trigger_tokens=1,
            target_tokens=100,
            keep_recent_tool_outputs=1,
        ),
    )
    client = TestClient(shim.app)
    old_tool_content = json.dumps(
        {
            "output": {"execution_id": "exec-1", "stdout": "x" * 2000},
            "success": True,
            "error": None,
        }
    )
    recent_tool_content = json.dumps({"output": "recent", "success": True})

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "nsl-test",
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "nsl---run_python", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "nsl---run_python",
                    "content": old_tool_content,
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "nsl---score", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "name": "nsl---score",
                    "content": recent_tool_content,
                },
            ],
        },
        headers={"Authorization": "Bearer abc"},
    )

    assert response.status_code == 200
    assert inner.last_messages is not None
    compacted_payload = json.loads(inner.last_messages[2]["content"])
    assert "context compaction" in compacted_payload["output"]["stdout"]
    assert compacted_payload["output"]["execution_id"] == "exec-1"
    assert inner.last_messages[4]["content"] == recent_tool_content
    assert recorder.inference_records[0].messages[2] == inner.last_messages[2]
    counters = recorder.snapshot_counters()
    assert counters["context_compactions_total"] == 1
    assert counters["context_compacted_tool_messages_total"] == 1


def test_shim_latches_inference_unavailable_flag(tmp_path: Path) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-inf-down"
    )

    def _down(messages, *, tools=None, tool_choice=None):
        return Err(f"{INFERENCE_UNAVAILABLE_PREFIX} Connection refused")

    inner.plist_completion = _down  # type: ignore[assignment]
    client = TestClient(shim.app)
    assert shim.inference_unavailable_detail is None

    response = client.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer abc"},
    )
    assert response.status_code == 502
    assert shim.inference_unavailable_detail is not None
    assert shim.inference_unavailable_detail.startswith(INFERENCE_UNAVAILABLE_PREFIX)


def test_shim_latches_timeout_into_separate_timeout_detail(tmp_path: Path) -> None:
    # A request timeout is the backend being slow, not an outage. The shim
    # latches it into a SEPARATE inference_timeout_detail (NOT
    # inference_unavailable_detail) so ContainerHarness categorises the episode
    # as inference_timeout — a benign, retryable abort that does NOT quarantine
    # the endpoint. (Latching it into inference_unavailable_detail would route
    # it to the endpoint_unavailable category, which the worker loop quarantines
    # on — with a single endpoint that breaches the capacity floor and aborts
    # the run.)
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-inf-timeout"
    )

    def _timeout(messages, *, tools=None, tool_choice=None):
        return Err(f"{INFERENCE_TIMEOUT_PREFIX} APITimeoutError: Request timed out.")

    inner.plist_completion = _timeout  # type: ignore[assignment]
    client = TestClient(shim.app)
    assert shim.inference_timeout_detail is None
    assert shim.inference_unavailable_detail is None

    response = client.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer abc"},
    )
    assert response.status_code == 502
    # Timeout latches into its OWN field, and NOT into the outage field.
    assert shim.inference_timeout_detail is not None
    assert shim.inference_timeout_detail.startswith(INFERENCE_TIMEOUT_PREFIX)
    assert shim.inference_unavailable_detail is None


def test_shim_latches_context_overflow_as_non_retryable_client_error(
    tmp_path: Path,
) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-context-overflow"
    )

    def _overflow(messages, *, tools=None, tool_choice=None):
        return Err(f"{CONTEXT_OVERFLOW_PREFIX} maximum context length exceeded")

    inner.plist_completion = _overflow  # type: ignore[assignment]
    client = TestClient(shim.app)
    assert shim.context_overflow_detail is None

    response = client.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer abc"},
    )

    assert response.status_code == 400
    assert shim.context_overflow_detail is not None
    assert shim.context_overflow_detail.startswith(CONTEXT_OVERFLOW_PREFIX)
    assert shim.inference_timeout_detail is None
    assert shim.inference_unavailable_detail is None


def test_extract_multi_parameter_blocks() -> None:
    tool_calls = _extract_pseudo_tool_calls(
        (
            "<tool_call>"
            "<function=foo>"
            "<parameter=a>1</parameter>"
            "<parameter=b>2</parameter>"
            "</function>"
            "</tool_call>"
        )
    )

    assert tool_calls is not None
    assert tool_calls[0]["function"]["arguments"] == '{"a": 1, "b": 2}'


def test_extract_name_arguments_single_param() -> None:
    tool_calls = _extract_pseudo_tool_calls(
        "<tool_call><name>nsl---run_python</name>"
        '<arguments><code>print("hello")</code></arguments>'
        "</tool_call>"
    )

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "nsl---run_python"
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["code"] == 'print("hello")'


def test_extract_name_arguments_multiple_params() -> None:
    tool_calls = _extract_pseudo_tool_calls(
        "<tool_call><name>nsl---run_python</name>"
        "<arguments><code>import os</code>"
        "<timeout_s>120</timeout_s></arguments>"
        "</tool_call>"
    )

    assert tool_calls is not None
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["code"] == "import os"
    assert args["timeout_s"] == 120


def test_extract_name_arguments_with_outer_wrapper() -> None:
    tool_calls = _extract_pseudo_tool_calls(
        "<tool_calls>\n"
        "  <tool_call>\n"
        "    <name>nsl---run_python</name>\n"
        "    <arguments><code>pass</code></arguments>\n"
        "  </tool_call>\n"
        "</tool_calls>"
    )

    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "nsl---run_python"


def test_extract_name_arguments_code_with_angle_brackets() -> None:
    code = "if x < 10:\n    print(x > 0)"
    tool_calls = _extract_pseudo_tool_calls(
        f"<tool_call><name>nsl---run_python</name>"
        f"<arguments><code>{code}</code></arguments>"
        f"</tool_call>"
    )

    assert tool_calls is not None
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert "if x < 10" in args["code"]


def test_extract_mixed_hermes_and_name_arguments() -> None:
    tool_calls = _extract_pseudo_tool_calls(
        "<tool_call><function=foo><parameter=a>1</parameter></function></tool_call>"
        "<tool_call><name>bar</name><arguments><x>2</x></arguments></tool_call>"
    )

    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "foo"
    assert tool_calls[1]["function"]["name"] == "bar"


def test_extract_multiple_name_arguments_calls() -> None:
    tool_calls = _extract_pseudo_tool_calls(
        "<tool_call><name>nsl---run_python</name>"
        "<arguments><code>print(1)</code></arguments></tool_call>"
        "<tool_call><name>nsl---run_python</name>"
        "<arguments><code>print(2)</code></arguments></tool_call>"
    )

    assert tool_calls is not None
    assert len(tool_calls) == 2
    args_0 = json.loads(tool_calls[0]["function"]["arguments"])
    args_1 = json.loads(tool_calls[1]["function"]["arguments"])
    assert args_0["code"] == "print(1)"
    assert args_1["code"] == "print(2)"


def test_extract_name_arguments_strips_code_fences() -> None:
    tool_calls = _extract_pseudo_tool_calls(
        "```xml\n<tool_call><name>nsl---run_python</name>"
        "<arguments><code>pass</code></arguments>"
        "</tool_call>\n```"
    )

    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "nsl---run_python"


def test_extract_multi_parameter_string_value_unquoted_stays_string() -> None:
    tool_calls = _extract_pseudo_tool_calls(
        (
            "<tool_call>"
            "<function=nsl---deploy_attack_sol>"
            "<parameter=attack_sol>contract A {}</parameter>"
            "</function>"
            "</tool_call>"
        )
    )

    assert tool_calls is not None
    assert tool_calls[0]["function"]["arguments"] == '{"attack_sol": "contract A {}"}'


def test_extract_duplicate_parameter_names_returns_structured_error() -> None:
    tool_calls = _extract_pseudo_tool_calls(
        (
            "<tool_call>"
            "<function=foo>"
            "<parameter=a>1</parameter>"
            "<parameter=a>2</parameter>"
            "</function>"
            "</tool_call>"
        )
    )

    assert tool_calls is not None
    assert (
        tool_calls[0]["function"]["arguments"]
        == '{"_error": "duplicate parameter names: [\'a\']"}'
    )


def test_extract_wrapped_arguments_unchanged() -> None:
    tool_calls = _extract_pseudo_tool_calls(
        (
            "<tool_call>"
            "<function=foo>"
            '<parameter=arguments>{"x":1}</parameter>'
            "</function>"
            "</tool_call>"
        )
    )

    assert tool_calls is not None
    assert tool_calls[0]["function"]["arguments"] == '{"x":1}'


def test_extract_function_attr_preserves_json_escaped_quotes() -> None:
    tool_calls = _extract_pseudo_tool_calls(
        r'''<function name="foo" arguments='{"msg": "She said \"hi\""}'/>'''
    )

    assert tool_calls is not None
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args == {"msg": 'She said "hi"'}


def test_extract_no_parameters_invalid_body_returns_structured_error() -> None:
    tool_calls = _extract_pseudo_tool_calls(
        "<tool_call><function=foo>not json</function></tool_call>"
    )

    assert tool_calls is not None
    assert '"_error":' in tool_calls[0]["function"]["arguments"]
    assert "not json" not in tool_calls[0]["function"]["arguments"]


def test_shim_rehydrates_name_arguments_dialect(tmp_path: Path) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-name-args"
    )
    inner.set_tool_response(
        None,
        stop_reason="stop",
        content=(
            "```xml\n"
            "<tool_calls>\n"
            "<tool_call>\n"
            "<name>nsl---run_python</name>\n"
            "<arguments>\n"
            '<code>import os; os.listdir("/")</code>\n'
            "</arguments>\n"
            "</tool_call>\n"
            "</tool_calls>\n"
            "```"
        ),
    )
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer abc"},
    )
    assert response.status_code == 200
    body = response.json()
    message = body["choices"][0]["message"]
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "nsl---run_python"
    assert body["choices"][0]["finish_reason"] == "tool_calls"


# --- Dialect 3: direct tool-name tags (<nsl---run_python>{json}) ---


def test_extract_direct_tool_tag_json_body() -> None:
    """<tool_name>{json}</tool_name> format with valid JSON body."""
    tool_calls = _extract_pseudo_tool_calls(
        '<nsl---run_python>{"code": "print(1)"}</nsl---run_python>'
    )
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "nsl---run_python"
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["code"] == "print(1)"


def test_extract_direct_tool_tag_triple_quoted_json() -> None:
    """Model emits Python triple-quoted strings inside JSON-like body."""
    content = (
        "<nsl---run_python>\n"
        "    {\n"
        '        "code": """\n'
        "import os\n"
        'for f in os.listdir("/tmp"):\n'
        '    os.remove(os.path.join("/tmp", f))\n'
        '"""\n'
        "    }\n"
        "</nsl---run_python>"
    )
    tool_calls = _extract_pseudo_tool_calls(content)
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "nsl---run_python"
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert "import os" in args["code"]
    assert "os.remove" in args["code"]


def test_extract_direct_tool_tag_in_code_fences() -> None:
    """Wrapped in ```xml code fences."""
    tool_calls = _extract_pseudo_tool_calls(
        '```xml\n<nsl---run_python>{"code": "pass"}</nsl---run_python>\n```'
    )
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "nsl---run_python"


def test_extract_direct_tool_tag_no_match_on_regular_xml() -> None:
    """Regular XML tags like <div> must not be extracted as tool calls."""
    tool_calls = _extract_pseudo_tool_calls("<div>some content</div>")
    assert tool_calls is None


def test_extract_direct_tool_tag_no_match_on_single_segment_tag() -> None:
    """Tags without --- separator are not tool calls."""
    tool_calls = _extract_pseudo_tool_calls('<run_python>{"code": "pass"}</run_python>')
    assert tool_calls is None


def test_extract_direct_tool_tag_with_whitespace() -> None:
    """Whitespace around body in direct tool tag."""
    tool_calls = _extract_pseudo_tool_calls(
        '<nsl---run_python>\n  {"code": "x = 1"}  \n</nsl---run_python>'
    )
    assert tool_calls is not None
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["code"] == "x = 1"


# --- Dialect 4: JSON-in-tag (<tool_call>{"name":..,"arguments":..}</tool_call>) ---
#
# This is the Hermes-2-Pro / Qwen2.5 / Qwen3-MoE chat-template emission
# contract. vLLM's hermes parser normally extracts these into structured
# tool_calls; the shim only sees them in raw text when the parser failed
# (intermittent on long thinking-model responses).


def test_extract_json_in_tag_basic() -> None:
    """Canonical Qwen2.5 / Hermes-2 JSON-in-tag emission."""
    tool_calls = _extract_pseudo_tool_calls(
        '<tool_call>\n{"name": "nsl---run_shell", "arguments": '
        '{"code": "ls /tmp", "timeout_s": 60}}\n</tool_call>'
    )
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "nsl---run_shell"
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["code"] == "ls /tmp"
    assert args["timeout_s"] == 60


def test_extract_json_in_tag_after_think_block() -> None:
    """Real-world Qwen3-MoE shape: <think>...</think> followed by tool call."""
    content = (
        "<think>\nLet me read the victim source first.\n</think>\n\n"
        '<tool_call>\n{"name": "nsl---run_shell", '
        '"arguments": {"code": "rg -n function /tmp/x.sol"}}\n</tool_call>'
    )
    tool_calls = _extract_pseudo_tool_calls(content)
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "nsl---run_shell"


def test_extract_json_in_tag_multiple_calls() -> None:
    """Two JSON-in-tag calls in one response."""
    content = (
        '<tool_call>{"name": "nsl---run_shell", "arguments": {"code": "pwd"}}</tool_call>'
        '<tool_call>{"name": "nsl---run_shell", "arguments": {"code": "ls"}}</tool_call>'
    )
    tool_calls = _extract_pseudo_tool_calls(content)
    assert tool_calls is not None
    assert len(tool_calls) == 2
    args_0 = json.loads(tool_calls[0]["function"]["arguments"])
    args_1 = json.loads(tool_calls[1]["function"]["arguments"])
    assert args_0["code"] == "pwd"
    assert args_1["code"] == "ls"


def test_extract_json_in_tag_with_xml_nested_dialect() -> None:
    """JSON-in-tag and Hermes XML-nested in one response — both extracted."""
    content = (
        '<tool_call>{"name": "foo", "arguments": {"a": 1}}</tool_call>'
        '<tool_call><function=bar><parameter=b>2</parameter></function></tool_call>'
    )
    tool_calls = _extract_pseudo_tool_calls(content)
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "foo"
    assert tool_calls[1]["function"]["name"] == "bar"


def test_extract_json_in_tag_arguments_is_string_json() -> None:
    """Some models emit `arguments` as a JSON-encoded string, not a nested object."""
    content = (
        '<tool_call>{"name": "nsl---run_shell", '
        '"arguments": "{\\"code\\": \\"ls\\"}"}</tool_call>'
    )
    tool_calls = _extract_pseudo_tool_calls(content)
    assert tool_calls is not None
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["code"] == "ls"


def test_extract_json_in_tag_falls_through_when_xml_nested_present() -> None:
    """If a tool_call body contains <function=...>, JSON path must not fire."""
    content = (
        "<tool_call><function=foo><parameter=a>1</parameter></function></tool_call>"
    )
    tool_calls = _extract_pseudo_tool_calls(content)
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "foo"


def test_extract_json_in_tag_invalid_json_returns_none() -> None:
    """Body looks like JSON but isn't parseable — skip rather than spuriously match."""
    content = '<tool_call>{not valid json}</tool_call>'
    tool_calls = _extract_pseudo_tool_calls(content)
    assert tool_calls is None


def test_extract_json_in_tag_missing_name_returns_none() -> None:
    """Valid JSON but no `name` field — not a tool call."""
    content = '<tool_call>{"arguments": {"a": 1}}</tool_call>'
    tool_calls = _extract_pseudo_tool_calls(content)
    assert tool_calls is None


# --- Dialect variant: <call> instead of <tool_call> ---


def test_extract_call_tag_with_name_arguments() -> None:
    """<call> used as synonym for <tool_call>."""
    tool_calls = _extract_pseudo_tool_calls(
        "<tool_calls>\n"
        "  <call>\n"
        "    <name>nsl---run_python</name>\n"
        "    <arguments><code>pass</code></arguments>\n"
        "  </call>\n"
        "</tool_calls>"
    )
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "nsl---run_python"


def test_extract_mixed_all_three_dialects() -> None:
    """All three dialects in one response — all extracted."""
    content = (
        # Dialect 1: Hermes
        "<tool_call><function=foo><parameter=a>1</parameter></function></tool_call>"
        # Dialect 2: <name>/<arguments>
        "<tool_call><name>bar</name><arguments><x>2</x></arguments></tool_call>"
        # Dialect 3: direct tool tag
        '<baz---tool>{"y": 3}</baz---tool>'
    )
    tool_calls = _extract_pseudo_tool_calls(content)
    assert tool_calls is not None
    assert len(tool_calls) == 3
    names = [tc["function"]["name"] for tc in tool_calls]
    assert "foo" in names
    assert "bar" in names
    assert "baz---tool" in names


def test_extract_skipped_pass1_calls_do_not_collide_with_direct_tag_ids() -> None:
    content = (
        "<tool_call>missing name and function</tool_call>"
        "<tool_call><name>bar</name><arguments>{\"x\":2}</arguments></tool_call>"
        '<baz---tool>{"y": 3}</baz---tool>'
    )

    tool_calls = _extract_pseudo_tool_calls(content)

    assert tool_calls is not None
    assert [tc["id"] for tc in tool_calls] == ["call_1", "call_2"]
    assert len({tc["id"] for tc in tool_calls}) == len(tool_calls)


def test_shim_rehydrates_direct_tool_tag_dialect(tmp_path: Path) -> None:
    """End-to-end: shim converts <tool_name>{json} to OpenAI tool_calls."""
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-direct-tag"
    )
    inner.set_tool_response(
        None,
        stop_reason="stop",
        content=(
            "```xml\n"
            "<nsl---run_python>\n"
            '{"code": "import os; os.listdir(\\"/\\")"}\n'
            "</nsl---run_python>\n"
            "```"
        ),
    )
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer abc"},
    )
    assert response.status_code == 200
    body = response.json()
    message = body["choices"][0]["message"]
    assert message["tool_calls"][0]["function"]["name"] == "nsl---run_python"
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_shim_rejects_missing_or_wrong_token(tmp_path: Path) -> None:
    shim, recorder, _inner = _build_shim(tmp_path, token="secret", episode_id="ep-2")
    client = TestClient(shim.app)
    missing = client.post("/v1/chat/completions", json=_openai_body())
    assert missing.status_code == 401
    wrong = client.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer not-the-token"},
    )
    assert wrong.status_code == 401
    # No inference should have been recorded.
    assert len(recorder.inference_records) == 0


def test_shim_rejects_streaming_v1(tmp_path: Path) -> None:
    shim, recorder, _inner = _build_shim(tmp_path, token="abc", episode_id="ep-3")
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json=_openai_body(stream=True),
        headers={"Authorization": "Bearer abc"},
    )
    assert response.status_code == 400
    assert len(recorder.inference_records) == 0


def test_shim_uses_explicit_x_nsl_phase_header(tmp_path: Path) -> None:
    """A header-specified phase tag overrides the auto-generated one."""
    shim, recorder, _inner = _build_shim(tmp_path, token="abc", episode_id="ep-4")
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={
            "Authorization": "Bearer abc",
            "X-NSL-Phase": "my_custom_phase",
        },
    )
    assert response.status_code == 200
    assert recorder.inference_records[0].phase == "my_custom_phase"


def test_shim_per_episode_phase_counters_are_isolated(tmp_path: Path) -> None:
    """Two shim instances keep independent monotonic counters —
    concurrent episodes must not collide on ``step_N`` numbering."""
    shim_a, rec_a, _ = _build_shim(tmp_path, token="t", episode_id="ep-a")
    shim_b, rec_b, _ = _build_shim(tmp_path, token="t", episode_id="ep-b")
    client_a = TestClient(shim_a.app)
    client_b = TestClient(shim_b.app)

    for _ in range(3):
        client_a.post(
            "/v1/chat/completions",
            json=_openai_body(),
            headers={"Authorization": "Bearer t"},
        )
    client_b.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer t"},
    )

    phases_a = [r.phase for r in rec_a.inference_records]
    phases_b = [r.phase for r in rec_b.inference_records]
    # Each shim independently produces ``step_1``, ``step_2`` ... — the two
    # episodes share no counter state, so ep-b's first call is ``step_1``
    # even after ep-a has advanced to step_3.
    assert len(phases_a) == 3
    assert len(phases_b) == 1
    assert phases_a[0].endswith("step_1")
    assert phases_a[1].endswith("step_2")
    assert phases_a[2].endswith("step_3")
    assert phases_b[0].endswith("step_1")
