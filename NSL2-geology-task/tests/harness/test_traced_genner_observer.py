"""TracedGenner.on_inference hook fires after every inference boundary.

The framework uses this hook to propagate prompt-token and harness-owned
telemetry updates to the progress display without requiring harnesses to opt
into a second callback surface.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

from result import Err, Ok

from src.harness.recorder import EventRecorder
from src.harness.budget import BudgetLedger
from src.harness.traced_genner import TracedGenner
from src.observability.types import InferenceResult, UsageInfo
from src.task.types import BudgetConstraints


def _make_recorder(tmp_path: Path) -> EventRecorder:
    return EventRecorder(episode_id="ep-1", output_path=tmp_path / "events.jsonl")


def _ok_inner(prompt_tokens: int) -> MagicMock:
    inner = MagicMock()
    inner.plist_completion.return_value = Ok(
        InferenceResult(
            content="hi",
            usage=UsageInfo(prompt_tokens=prompt_tokens, completion_tokens=1),
        )
    )
    return inner


def test_on_inference_fires_with_usage_on_success(tmp_path: Path) -> None:
    observed: list[UsageInfo | None] = []
    traced = TracedGenner(
        inner=_ok_inner(prompt_tokens=42),
        recorder=_make_recorder(tmp_path),
        cancel_event=threading.Event(),
        episode_id="ep-1",
        on_inference=observed.append,
    )

    traced.plist_completion([{"role": "user", "content": "hi"}], phase="p")
    assert len(observed) == 1
    assert observed[0] is not None
    assert observed[0].prompt_tokens == 42


def test_budget_ledger_records_successful_llm_turn(tmp_path: Path) -> None:
    recorder = _make_recorder(tmp_path)
    ledger = BudgetLedger(BudgetConstraints(max_llm_turns=1))
    traced = TracedGenner(
        inner=_ok_inner(prompt_tokens=42),
        recorder=recorder,
        cancel_event=threading.Event(),
        episode_id="ep-1",
        budget_ledger=ledger,
    )

    traced.plist_completion([{"role": "user", "content": "hi"}], phase="p")

    assert ledger.snapshot().llm_turns_used == 1
    assert recorder.snapshot_counters()["llm_turns"] == 1


def test_on_inference_fires_with_none_on_error(tmp_path: Path) -> None:
    observed: list[UsageInfo | None] = []
    inner = MagicMock()
    inner.plist_completion.return_value = Err("boom")
    traced = TracedGenner(
        inner=inner,
        recorder=_make_recorder(tmp_path),
        cancel_event=threading.Event(),
        episode_id="ep-1",
        on_inference=observed.append,
    )

    traced.plist_completion([{"role": "user", "content": "x"}], phase="p")
    assert observed == [None]


def test_budget_ledger_does_not_record_failed_llm_turn(tmp_path: Path) -> None:
    ledger = BudgetLedger(BudgetConstraints(max_llm_turns=1))
    inner = MagicMock()
    inner.plist_completion.return_value = Err("boom")
    traced = TracedGenner(
        inner=inner,
        recorder=_make_recorder(tmp_path),
        cancel_event=threading.Event(),
        episode_id="ep-1",
        budget_ledger=ledger,
    )

    traced.plist_completion([{"role": "user", "content": "x"}], phase="p")

    assert ledger.snapshot().llm_turns_used == 0


def test_observer_exception_does_not_derail_episode(tmp_path: Path) -> None:
    """UI hooks must be advisory — a raising observer must not propagate."""

    def _bad(_usage: UsageInfo | None) -> None:
        raise RuntimeError("display crash")

    traced = TracedGenner(
        inner=_ok_inner(prompt_tokens=7),
        recorder=_make_recorder(tmp_path),
        cancel_event=threading.Event(),
        episode_id="ep-1",
        on_inference=_bad,
    )
    # Must not raise — completion still succeeds.
    traced.plist_completion([{"role": "user", "content": "x"}], phase="p")


def test_on_inference_fires_on_pre_call_cancel(tmp_path: Path) -> None:
    """Cancellation path records an inference and fires the observer."""
    observed: list[UsageInfo | None] = []
    cancel = threading.Event()
    cancel.set()
    traced = TracedGenner(
        inner=_ok_inner(prompt_tokens=1),
        recorder=_make_recorder(tmp_path),
        cancel_event=cancel,
        episode_id="ep-1",
        on_inference=observed.append,
    )

    traced.plist_completion([{"role": "user", "content": "x"}], phase="p")
    assert observed == [None]
