from __future__ import annotations

from typing import Any

import pytest
from result import Err, Ok

from src.backend.endpoint_pool import EndpointPool, EndpointPoolUnavailable, EndpointState
from src.genner.Base import (
    Genner,
    INFERENCE_TIMEOUT_PREFIX,
    INFERENCE_UNAVAILABLE_PREFIX,
)
from src.observability.types import InferenceResult
from src.typing.message import Message


class _FakeGenner(Genner):
    def __init__(self, name: str, result: Any | None = None) -> None:
        super().__init__("vllm")
        self.name = name
        self.result = result if result is not None else Ok(InferenceResult(content=name))
        self.calls: list[list[Message]] = []

    def plist_completion(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ):
        del tools, tool_choice
        self.calls.append(messages)
        return self.result

    @staticmethod
    def get_usage_info(response: object):
        return None


def _state(endpoint_id: str, capacity: int, result: Any | None = None) -> EndpointState:
    return EndpointState(
        endpoint_id=endpoint_id,
        base_url=f"http://{endpoint_id}:8000/v1",
        models_url=f"http://{endpoint_id}:8000/v1/models",
        metrics_url=f"http://{endpoint_id}:8000/metrics",
        capacity=capacity,
        genner=_FakeGenner(endpoint_id, result),
    )


def test_capacity_proportional_home_assignment_is_deterministic() -> None:
    pool = EndpointPool([_state("ep0", 8), _state("ep1", 16)])

    homes = [pool.home_endpoint_id(slot_id) for slot_id in range(24)]

    assert homes.count("ep0") == 8
    assert homes.count("ep1") == 16
    assert homes == [pool.home_endpoint_id(slot_id) for slot_id in range(24)]
    assert homes == [pool.home_endpoint_id(slot_id + 24) for slot_id in range(24)]


def test_capacity_cap_never_exceeds_endpoint_limit() -> None:
    pool = EndpointPool([_state("ep0", 1)])

    lease = pool.lease(home_key=0, timeout=0.01)
    try:
        with pytest.raises(EndpointPoolUnavailable):
            pool.lease(home_key=0, timeout=0.01)
        assert pool.in_flight("ep0") == 1
    finally:
        pool.release(lease)

    assert pool.in_flight("ep0") == 0


def test_reactive_quarantine_skips_unhealthy_home_until_probe_marks_healthy() -> None:
    pool = EndpointPool([_state("ep0", 1), _state("ep1", 1)])
    home = pool.home_endpoint_id(0)
    fallback = "ep1" if home == "ep0" else "ep0"

    pool.mark_unhealthy(home, "connection refused")
    lease = pool.lease(home_key=0, timeout=0.01)
    try:
        assert lease.endpoint_id == fallback
    finally:
        pool.release(lease)

    pool.mark_healthy(home)
    lease = pool.lease(home_key=0, timeout=0.01)
    try:
        assert lease.endpoint_id == home
    finally:
        pool.release(lease)


def test_endpoint_unavailable_result_marks_endpoint_unhealthy_immediately() -> None:
    # A genuine endpoint outage (connection refused / unreachable) carries the
    # inference_unavailable prefix and DOES quarantine the endpoint, so the
    # pool can fail over to healthy endpoints.
    pool = EndpointPool(
        [
            _state(
                "ep0",
                1,
                Err(f"{INFERENCE_UNAVAILABLE_PREFIX} APIConnectionError: connection refused"),
            )
        ]
    )

    lease = pool.lease(home_key=0, timeout=0.01)
    try:
        result = lease.genner.plist_completion([{"role": "user", "content": "hi"}])
    finally:
        pool.release(lease)

    assert isinstance(result, Err)
    assert not pool.is_healthy("ep0")


def test_inference_timeout_result_does_not_quarantine_endpoint() -> None:
    # A request timeout is a retryable episode failure, not an endpoint outage.
    # With a single endpoint, quarantining on a timeout would drop healthy
    # capacity below the floor and abort the whole run. The endpoint must stay
    # healthy: quarantine is keyed on the inference_unavailable prefix, which
    # the timeout prefix deliberately does not match.
    pool = EndpointPool(
        [
            _state(
                "ep0",
                1,
                Err(f"{INFERENCE_TIMEOUT_PREFIX} APITimeoutError: Request timed out."),
            )
        ]
    )

    lease = pool.lease(home_key=0, timeout=0.01)
    try:
        result = lease.genner.plist_completion([{"role": "user", "content": "hi"}])
    finally:
        pool.release(lease)

    assert isinstance(result, Err)
    assert pool.is_healthy("ep0")
    assert not pool.below_capacity_floor()
