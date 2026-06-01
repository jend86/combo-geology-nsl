"""Regression test: the dual-endpoint pool smoke test must not self-recurse.

`_build_pool_smoke_test` returns a `run_smoke_test` that the caller assigns onto the
PRIMARY session's `.smoke_test`. The primary session is also a value in
`sessions_by_endpoint`, so if the pool smoke test resolves `session.smoke_test` at call
time it points back at itself for the primary endpoint -> infinite recursion
("maximum recursion depth exceeded"), which quarantined BOTH endpoints and aborted the
run at startup (2026-06-01). The fix snapshots each endpoint's own smoke test at build
time, before that reassignment.
"""

from src.backend.vllm import _build_pool_smoke_test


class _FakeSession:
    def __init__(self, name: str):
        self.name = name
        self.smoke_test = lambda: f"ok-{name}"


class _FakePool:
    def __init__(self, ids: list[str]):
        self._ids = ids
        self.unhealthy: list[str] = []

    def endpoint_ids(self) -> list[str]:
        return list(self._ids)

    def is_healthy(self, endpoint_id: str) -> bool:
        return endpoint_id not in self.unhealthy

    def mark_unhealthy(self, endpoint_id: str, _reason: str) -> None:
        self.unhealthy.append(endpoint_id)


def test_pool_smoke_test_does_not_recurse_on_primary():
    sessions = {
        "local-4090": _FakeSession("local-4090"),
        "runpod-a40-1": _FakeSession("runpod-a40-1"),
    }
    pool = _FakePool(["local-4090", "runpod-a40-1"])
    run = _build_pool_smoke_test(pool, sessions)  # type: ignore[arg-type]  # duck-typed fakes
    # Reproduce the real wiring that triggered the bug: the primary session's
    # smoke_test is reassigned to the pool smoke test itself. Old code recursed here.
    sessions["local-4090"].smoke_test = run
    assert run() == "local-4090: ok-local-4090"


def test_pool_smoke_test_falls_through_to_next_healthy_endpoint():
    def boom():
        raise RuntimeError("down")

    sessions = {
        "local-4090": _FakeSession("local-4090"),
        "runpod-a40-1": _FakeSession("runpod-a40-1"),
    }
    sessions["local-4090"].smoke_test = boom  # captured at build time below
    pool = _FakePool(["local-4090", "runpod-a40-1"])
    run = _build_pool_smoke_test(pool, sessions)  # type: ignore[arg-type]  # duck-typed fakes
    sessions["local-4090"].smoke_test = run  # reassign primary, as the real code does
    # local-4090 raises -> marked unhealthy + recorded -> fall through to runpod-a40-1.
    assert run() == "runpod-a40-1: ok-runpod-a40-1"
    assert pool.unhealthy == ["local-4090"]
