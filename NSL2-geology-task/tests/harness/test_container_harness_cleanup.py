"""``ContainerHarness`` teardown invariants — container + servers always stop.

Each acquired resource registers with ``ExitStack`` before the next
acquisition. Exceptions at any stage (startup, wait, read_transcript,
to_artifacts) still produce reverse-order teardown: container removed →
bridge stopped → shim stopped.

These tests fake the Docker client and the server handles so we exercise
the cleanup path without a real daemon or real listeners.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.harness.container import ContainerHarness
from src.harness.context import HarnessConfigView, HarnessContext
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.task.types import Capability, TaskPromptSpec, Variation


@dataclass
class _Spy:
    stops: list[str] = field(default_factory=list)

    def make_shim_handle(self) -> MagicMock:
        handle = MagicMock()
        handle.port = 9001
        handle.stop = lambda: self.stops.append("shim")
        return handle

    def make_bridge_handle(self) -> MagicMock:
        handle = MagicMock()
        handle.port = 9002
        handle.stop = lambda: self.stops.append("bridge")
        return handle


class _FakeContainer:
    def __init__(self, spy: _Spy, *, exit_code: int = 0) -> None:
        self._spy = spy
        self.exit_code = exit_code
        self.removed = False
        self.id = "abc123"

    def wait(self, **_kwargs):
        return {"StatusCode": self.exit_code}

    def reload(self) -> None: ...

    @property
    def status(self) -> str:
        return "exited"

    def logs(self, **_kwargs) -> bytes:
        return b""

    def kill(self) -> None: ...

    def remove(self, **_kwargs) -> None:
        self.removed = True
        self._spy.stops.append("container")


class _StubTask:
    metric_name = "dummy"
    metric_unit = ""
    name = "stub"
    description = "stub"

    def parse_response(self, raw_response, *, invoked_capability=None):
        return []

    def execute_capability(self, invocation, containers, variation, ctx):
        from src.task.types import CapabilityResult

        return CapabilityResult(name=invocation.name, output={}, success=True)


def _ctx(tmp_path: Path, docker_client: Any) -> HarnessContext:
    recorder = EventRecorder(episode_id="ep-1", output_path=tmp_path / "events.jsonl")
    traced = TracedGenner(
        inner=MagicMock(),
        recorder=recorder,
        cancel_event=threading.Event(),
        episode_id="ep-1",
    )
    prompt_spec = TaskPromptSpec(
        system_instruction="sys",
        capabilities=[Capability(name="analyzer", description="r")],
    )
    return HarnessContext(
        episode_id="ep-1",
        genner=traced,
        task=_StubTask(),  # type: ignore[arg-type]
        variation=Variation(name="v", description="d"),
        prompt_spec=prompt_spec,
        episode_context={},
        containers=[],
        agent_container=None,  # type: ignore[arg-type]
        host_cache_folder=tmp_path,
        config=HarnessConfigView(
            harness_settings={},
            model_name="test",
            train_data_save_folder=str(tmp_path),
            code_host_cache_path=str(tmp_path),
        ),
        metrics=None,  # type: ignore[arg-type]
        recorder=recorder,
        cancel_event=threading.Event(),
        docker_client=docker_client,
    )


def _patched_harness(
    monkeypatch: pytest.MonkeyPatch,
    spy: _Spy,
    *,
    profile_mock: MagicMock,
    on_wait: Any = None,
    on_read: Any = None,
) -> ContainerHarness:
    import src.harness.container as container_mod

    monkeypatch.setattr(
        container_mod,
        "_serve_on_loopback",
        lambda app: spy.make_shim_handle(),
    )
    # The bridge exposes ``serve_on_loopback`` as an instance method — patch
    # the class so the ContainerHarness driver uses our spy handles.
    monkeypatch.setattr(
        container_mod.CapabilityMcpBridge,
        "serve_on_loopback",
        lambda self: spy.make_bridge_handle(),
    )
    monkeypatch.setattr(
        container_mod,
        "resolve_profile",
        lambda name, cfg: profile_mock,
    )
    monkeypatch.setattr(
        container_mod,
        "_docker_host_gateway",
        lambda client, network_mode=None: "172.17.0.1",
    )
    if on_wait is not None:
        monkeypatch.setattr(container_mod, "_wait_with_cancel", on_wait)
    if on_read is None:
        profile_mock.read_transcript.return_value = {
            "messages": [{"role": "assistant", "content": "done"}]
        }
    harness = ContainerHarness(
        harness_config={
            "profile": "ms_agent",
            "image": "img",
            "max_wall_seconds": 10,
        }
    )
    return harness


def _build_profile_mock() -> MagicMock:
    profile = MagicMock()
    profile.render_query.return_value = "q"
    profile.default_args.return_value = ["python", "/opt/nsl/run.py"]
    profile.env.return_value = {}
    profile.count_llm_turns.return_value = 1
    profile.to_artifacts.return_value = MagicMock(
        capability_invocations=[], capability_results=[], final_response="done"
    )
    return profile


def test_clean_path_stops_in_reverse_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _Spy()
    profile_mock = _build_profile_mock()
    docker = MagicMock()
    fake_container = _FakeContainer(spy)
    docker.containers.run.return_value = fake_container

    harness = _patched_harness(
        monkeypatch,
        spy,
        profile_mock=profile_mock,
        on_wait=lambda c, ev, max_wall: MagicMock(reason="exited", category="success"),
    )
    ctx = _ctx(tmp_path, docker)
    harness.run_episode(ctx=ctx)

    # Reverse-order teardown: container → bridge → shim.
    assert spy.stops == ["container", "bridge", "shim"]
    assert fake_container.removed is True


def test_exception_during_wait_still_tears_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _Spy()
    profile_mock = _build_profile_mock()
    docker = MagicMock()
    fake_container = _FakeContainer(spy)
    docker.containers.run.return_value = fake_container

    def _boom(*_a, **_k):
        raise RuntimeError("wait blew up")

    harness = _patched_harness(
        monkeypatch,
        spy,
        profile_mock=profile_mock,
        on_wait=_boom,
    )
    ctx = _ctx(tmp_path, docker)
    with pytest.raises(RuntimeError, match="wait blew up"):
        harness.run_episode(ctx=ctx)

    assert "container" in spy.stops
    assert "bridge" in spy.stops
    assert "shim" in spy.stops
    # Reverse-order preserved even under failure.
    assert spy.stops == ["container", "bridge", "shim"]


def test_exception_during_read_transcript_still_tears_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _Spy()
    profile_mock = _build_profile_mock()
    profile_mock.read_transcript.side_effect = RuntimeError("bad transcript")
    docker = MagicMock()
    fake_container = _FakeContainer(spy)
    docker.containers.run.return_value = fake_container

    harness = _patched_harness(
        monkeypatch,
        spy,
        profile_mock=profile_mock,
        on_wait=lambda c, ev, max_wall: MagicMock(reason="exited", category="success"),
    )
    ctx = _ctx(tmp_path, docker)
    with pytest.raises(RuntimeError, match="bad transcript"):
        harness.run_episode(ctx=ctx)

    # All three must still stop — container removed first, bridge, then shim.
    assert spy.stops == ["container", "bridge", "shim"]


def test_container_remove_failure_does_not_block_server_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_force_remove_quietly`` logs but does not raise — servers must
    still stop when the Docker daemon is unreachable."""
    spy = _Spy()
    profile_mock = _build_profile_mock()
    docker = MagicMock()
    fake_container = _FakeContainer(spy)

    def _broken_remove(**_kwargs):
        raise RuntimeError("daemon unreachable")

    fake_container.remove = _broken_remove  # type: ignore[assignment]
    docker.containers.run.return_value = fake_container

    harness = _patched_harness(
        monkeypatch,
        spy,
        profile_mock=profile_mock,
        on_wait=lambda c, ev, max_wall: MagicMock(reason="exited", category="success"),
    )
    ctx = _ctx(tmp_path, docker)
    # Clean exit — even with container remove failing.
    harness.run_episode(ctx=ctx)
    assert "bridge" in spy.stops
    assert "shim" in spy.stops
