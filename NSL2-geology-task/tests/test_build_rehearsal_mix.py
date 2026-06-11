"""Unit tests for the rehearsal-mix builder's pure format adapters.

These exercise the *pure* adapter functions with synthetic source rows, so they
need none of the heavy ``datasets``/tokenizer machinery — the script keeps all
non-stdlib imports lazy inside its I/O helpers. Each adapter must emit the
uniform ``{prompt, raw_response, success, source}`` schema that
``_load_self_generated_sft_rows`` consumes (so rehearsal rows ride the identical
completion-mask path as task rows when passed as extra ``--training-data``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_rehearsal_mix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_rehearsal_mix", _SCRIPT)
    assert spec and spec.loader, f"cannot load {_SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # must not import datasets/torch at module top
    return module


mod = _load_module()


def _assert_uniform(row, *, source):
    assert isinstance(row, dict)
    assert isinstance(row["prompt"], str) and row["prompt"].strip()
    assert isinstance(row["raw_response"], str) and row["raw_response"].strip()
    assert row["success"] is True
    assert row["source"] == source


# ---- chat (UltraChat: {messages: [{role, content}, ...]}) ----

def test_adapt_chat_picks_first_user_then_assistant():
    row = {
        "messages": [
            {"role": "user", "content": "What landmarks should I see in London?"},
            {"role": "assistant", "content": "1. Leadenhall Market ..."},
            {"role": "user", "content": "and food?"},
            {"role": "assistant", "content": "Borough Market ..."},
        ]
    }
    out = mod.adapt_chat(row)
    _assert_uniform(out, source="chat")
    assert out["prompt"] == "What landmarks should I see in London?"
    assert out["raw_response"] == "1. Leadenhall Market ..."


def test_adapt_chat_returns_none_when_no_assistant():
    assert mod.adapt_chat({"messages": [{"role": "user", "content": "hi"}]}) is None
    assert mod.adapt_chat({"messages": []}) is None
    assert mod.adapt_chat({}) is None


# ---- instruction (Magicoder: instruction/response; MetaMathQA: query/response) ----

def test_adapt_instruction_maps_named_fields():
    magic = {"instruction": "Fix this Python loop", "response": "```python\nwhile ...\n```"}
    out = mod.adapt_instruction(magic, query_field="instruction",
                                response_field="response", source="code")
    _assert_uniform(out, source="code")
    assert out["prompt"] == "Fix this Python loop"
    assert out["raw_response"].startswith("```python")

    math = {"query": "How far apart are the points?", "response": "The distance is sqrt(5)."}
    out2 = mod.adapt_instruction(math, query_field="query",
                                 response_field="response", source="math")
    _assert_uniform(out2, source="math")
    assert out2["prompt"] == "How far apart are the points?"


def test_adapt_instruction_returns_none_on_empty():
    assert mod.adapt_instruction({"instruction": "", "response": "x"},
                                 query_field="instruction", response_field="response",
                                 source="code") is None
    assert mod.adapt_instruction({"instruction": "q", "response": "  "},
                                 query_field="instruction", response_field="response",
                                 source="code") is None


# ---- ToolACE ({system, conversations:[{from, value}]}) self-consistent format ----

def test_adapt_toolace_keeps_system_scaffold_and_first_call():
    row = {
        "system": "You are an expert in composing functions. [func defs ...]",
        "conversations": [
            {"from": "user", "value": "Get me the top US market trends."},
            {"from": "assistant", "value": "[Market Trends API(trend_type=\"INDEXES\")]"},
            {"from": "tool", "value": "[{...}]"},
            {"from": "assistant", "value": "Here are the trends ..."},
        ],
    }
    out = mod.adapt_toolace(row)
    _assert_uniform(out, source="tool")
    # The function-listing system prompt must be carried into the prompt so the
    # [Func(args)] response surface form is self-justified (not format drift).
    assert "composing functions" in out["prompt"]
    assert "Get me the top US market trends." in out["prompt"]
    # response is the FIRST assistant turn (the tool call), not a later prose turn.
    assert out["raw_response"] == "[Market Trends API(trend_type=\"INDEXES\")]"


def test_adapt_toolace_returns_none_without_assistant():
    assert mod.adapt_toolace({"system": "s", "conversations":
                              [{"from": "user", "value": "hi"}]}) is None


# ---- geology continuation (raw text) ----

def test_adapt_geology_continuation_wrapping():
    text = "  Ozone   forecasting  service developed by the TEMIS consortium using assimilation.  "
    out = mod.adapt_geology(text, prompt_chars=20, max_chars=200)
    _assert_uniform(out, source="geology")
    assert out["prompt"].startswith("Continue the following geoscience passage")
    # whitespace normalized + excerpt embedded
    assert "Ozone forecasting" in out["raw_response"]
    assert "  " not in out["raw_response"]  # collapsed whitespace


def test_adapt_geology_respects_max_chars():
    out = mod.adapt_geology("word " * 500, prompt_chars=10, max_chars=50)
    assert len(out["raw_response"]) <= 50


def test_adapt_geology_returns_none_on_blank():
    assert mod.adapt_geology("   ", prompt_chars=10, max_chars=50) is None
    assert mod.adapt_geology("", prompt_chars=10, max_chars=50) is None
