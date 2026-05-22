"""Regression guard: the source tree must not carry residual references to
``TaskPromptSpec.extras`` (the field was deleted in Phase 2)."""

from __future__ import annotations

from pathlib import Path


def test_no_prompt_spec_extras_references_in_src_or_tasks() -> None:
    """``TaskPromptSpec.extras`` was deleted in Phase 2 — surface any
    residual references. Narrow match to ``prompt_spec.extras`` /
    ``TaskPromptSpec.extras`` so ``HarnessTranscript.extra`` and other
    ``.extra`` dict uses are not flagged."""
    repo_root = Path(__file__).resolve().parents[1]
    offenders: list[tuple[Path, int, str]] = []
    for root in (repo_root / "src", repo_root / "tasks"):
        for py in root.rglob("*.py"):
            for lineno, line in enumerate(py.read_text().splitlines(), start=1):
                if "prompt_spec.extras" in line or "TaskPromptSpec.extras" in line:
                    offenders.append((py, lineno, line.strip()))
    assert offenders == [], (
        "Residual TaskPromptSpec.extras references:\n"
        + "\n".join(f"  {p}:{ln}: {s}" for p, ln, s in offenders)
    )
