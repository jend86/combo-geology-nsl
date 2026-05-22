"""config-forked-exploit.toml's [harness.orchestrator_modes.modes.<name>]
prompts must carry enough fenced-block grammar on their own.

After `tasks/forked_exploit.py` was made harness-neutral, the task prompt
no longer teaches "emit a ```solidity fenced block." That grammar now
lives only in the orchestrator-modes config, not the task. If someone
trims these prompts down, OrchestratorModeHarness silently regresses:
the extractor (`extract_code_block`) needs a fenced block and finds
none.

This is a cheap config-level regression safeguard.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config-forked-exploit.toml"


def _load_mode_prompts() -> dict[str, str]:
    data = tomllib.loads(_CONFIG.read_text())
    return {
        name: m["prompt"]
        for name, m in data["harness"]["orchestrator_modes"]["modes"].items()
    }


def test_exploiter_prompt_teaches_solidity_fence() -> None:
    prompts = _load_mode_prompts()
    assert "```solidity" in prompts["exploiter"]
    assert "fenced" in prompts["exploiter"].lower()


def test_debugger_prompt_teaches_solidity_fence() -> None:
    prompts = _load_mode_prompts()
    assert "```solidity" in prompts["debugger"]


def test_analyzer_prompt_teaches_code_fence() -> None:
    prompts = _load_mode_prompts()
    assert "fenced" in prompts["analyzer"].lower() or "```" in prompts["analyzer"]
