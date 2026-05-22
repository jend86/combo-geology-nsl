"""Tests for task-owned episode constraint types.

TDD: these tests are written before the implementation and should fail until
EpisodeConstraints, BudgetConstraints, NoToolReplyPolicy, SuccessConstraints,
StepConstraints are added to src/task/types.py.
"""

from types import MappingProxyType

import pytest

from src.task.types import (
    BudgetConstraints,
    EpisodeConstraints,
    NoToolReplyPolicy,
    StepConstraints,
    SuccessConstraints,
)


def test_budget_constraints_defaults() -> None:
    bc = BudgetConstraints()
    assert bc.max_task_tool_calls is None
    assert bc.max_llm_turns is None
    assert bc.max_task_tool_calls_by_name == {}


def test_budget_constraints_with_values() -> None:
    bc = BudgetConstraints(
        max_task_tool_calls=20,
        max_llm_turns=30,
        max_task_tool_calls_by_name={"deploy": 5},
    )
    assert bc.max_task_tool_calls == 20
    assert bc.max_llm_turns == 30
    assert bc.max_task_tool_calls_by_name["deploy"] == 5


def test_budget_constraints_frozen() -> None:
    bc = BudgetConstraints(max_task_tool_calls=10)
    with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
        bc.max_task_tool_calls = 20  # type: ignore[misc]


def test_budget_constraints_mapping_is_immutable() -> None:
    bc = BudgetConstraints(max_task_tool_calls_by_name={"cap": 3})
    assert isinstance(bc.max_task_tool_calls_by_name, MappingProxyType)


def test_no_tool_reply_policy_defaults() -> None:
    p = NoToolReplyPolicy()
    assert p.retry is False
    assert p.max_retries == 0
    assert p.retry_instruction is None


def test_no_tool_reply_policy_frozen() -> None:
    p = NoToolReplyPolicy(retry=True, max_retries=2)
    with pytest.raises(Exception):
        p.retry = False  # type: ignore[misc]


def test_success_constraints_defaults() -> None:
    sc = SuccessConstraints()
    assert sc.min_task_tool_calls_for_success == 1
    assert sc.terminal_capability_for_success is None


def test_success_constraints_zero_min() -> None:
    sc = SuccessConstraints(min_task_tool_calls_for_success=0)
    assert sc.min_task_tool_calls_for_success == 0


def test_success_constraints_frozen() -> None:
    sc = SuccessConstraints()
    with pytest.raises(Exception):
        sc.min_task_tool_calls_for_success = 0  # type: ignore[misc]


def test_step_constraints_defaults() -> None:
    sc = StepConstraints()
    assert isinstance(sc.budgets, BudgetConstraints)
    assert isinstance(sc.no_tool_reply, NoToolReplyPolicy)
    assert isinstance(sc.success, SuccessConstraints)


def test_episode_constraints_defaults() -> None:
    ec = EpisodeConstraints()
    assert isinstance(ec.budgets, BudgetConstraints)
    assert isinstance(ec.no_tool_reply, NoToolReplyPolicy)
    assert isinstance(ec.success, SuccessConstraints)
    assert ec.step_overrides == {}


def test_episode_constraints_step_overrides_mapping_is_immutable() -> None:
    sc = StepConstraints(success=SuccessConstraints(min_task_tool_calls_for_success=0))
    ec = EpisodeConstraints(step_overrides={"plan_cleanup": sc})
    assert isinstance(ec.step_overrides, MappingProxyType)
    assert ec.step_overrides["plan_cleanup"].success.min_task_tool_calls_for_success == 0


def test_episode_constraints_frozen() -> None:
    ec = EpisodeConstraints()
    with pytest.raises(Exception):
        ec.budgets = BudgetConstraints(max_task_tool_calls=5)  # type: ignore[misc]


def test_episode_constraints_full_example() -> None:
    ec = EpisodeConstraints(
        budgets=BudgetConstraints(
            max_task_tool_calls=30,
            max_task_tool_calls_by_name={"deploy_attack_sol": 8},
            max_llm_turns=40,
        ),
        success=SuccessConstraints(
            min_task_tool_calls_for_success=1,
            terminal_capability_for_success=None,
        ),
    )
    assert ec.budgets.max_task_tool_calls == 30
    assert ec.budgets.max_task_tool_calls_by_name["deploy_attack_sol"] == 8
    assert ec.budgets.max_llm_turns == 40
    assert ec.success.min_task_tool_calls_for_success == 1
