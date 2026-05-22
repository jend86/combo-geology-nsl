from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
RUN_PY = ROOT / "docker" / "ms-agent" / "run.py"


def _load_run_module():
    spec = importlib.util.spec_from_file_location("nsl_ms_agent_run", RUN_PY)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_topological_chain_returns_next_chain_from_single_root() -> None:
    run = _load_run_module()
    spec = {
        "plan": {"agent_config": "plan.yaml", "next": ["act"]},
        "act": {"agent_config": "act.yaml"},
    }

    run._validate_spec(spec)

    assert run._topological_chain(spec) == ["plan", "act"]


def test_topological_chain_counts_on_error_target_as_non_root() -> None:
    run = _load_run_module()
    spec = {
        "try": {"agent_config": "try.yaml", "on_error": "recover"},
        "recover": {"agent_config": "recover.yaml"},
    }

    run._validate_spec(spec)

    assert run._topological_chain(spec) == ["try"]


def test_validate_spec_rejects_missing_next_step() -> None:
    run = _load_run_module()
    spec = {"plan": {"agent_config": "plan.yaml", "next": ["missing"]}}

    with pytest.raises(RuntimeError, match="next='missing' not found"):
        run._validate_spec(spec)


def test_validate_spec_rejects_missing_on_error_step() -> None:
    run = _load_run_module()
    spec = {"plan": {"agent_config": "plan.yaml", "on_error": "missing"}}

    with pytest.raises(RuntimeError, match="on_error='missing' not found"):
        run._validate_spec(spec)


def test_topological_chain_rejects_multiple_roots() -> None:
    run = _load_run_module()
    spec = {
        "plan": {"agent_config": "plan.yaml"},
        "act": {"agent_config": "act.yaml"},
    }

    run._validate_spec(spec)

    with pytest.raises(RuntimeError, match="exactly one root"):
        run._topological_chain(spec)


def test_topological_chain_rejects_cycles() -> None:
    run = _load_run_module()
    spec = {
        "root": {"agent_config": "root.yaml", "next": ["a"]},
        "a": {"agent_config": "a.yaml", "next": ["b"]},
        "b": {"agent_config": "b.yaml", "next": ["a"]},
    }

    run._validate_spec(spec)
    with pytest.raises(RuntimeError, match="cycle"):
        run._topological_chain(spec)
