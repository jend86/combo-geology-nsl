"""``OrchestratorModesConfig`` is a typed Pydantic model — not an untyped dict.

Phase 2 replaces ``HarnessConfig.orchestrator_modes: Dict[str, Any]`` with
``OrchestratorModesConfig | None``. ``extra="forbid"`` surfaces typos at
config load rather than silently ignoring them. Mode declarations live
under a single typed ``modes: dict[str, ModeConfig]`` block.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.typing.config import HarnessConfig, ModeConfig, OrchestratorModesConfig


def _min_app_payload() -> dict:
    """Smallest AppConfig payload that validates — avoids collateral errors."""
    return {
        "model_name": "test",
        "code_host_cache_path": "./code/",
        "container_ids": ["svc-1"],
        "train_data_save_folder": "./data/",
    }


def test_orchestrator_modes_requires_orchestrator_prompt() -> None:
    """``orchestrator_prompt`` is required — no silent default."""
    with pytest.raises(ValidationError):
        OrchestratorModesConfig()  # type: ignore[call-arg]


def test_mode_config_runs_code_requires_capability() -> None:
    """``runs_code=True`` without ``code_capability`` is a config error."""
    with pytest.raises(ValidationError):
        ModeConfig(prompt="x", runs_code=True)


def test_mode_config_runs_code_with_capability() -> None:
    cfg = ModeConfig(prompt="x", runs_code=True, code_capability="run_python")
    assert cfg.runs_code is True
    assert cfg.code_capability == "run_python"


def test_orchestrator_modes_requires_a_scratchpad_writer() -> None:
    """When modes are populated, at least one must opt into the scratchpad."""
    with pytest.raises(ValidationError, match="writes_scratchpad"):
        OrchestratorModesConfig(
            orchestrator_prompt="hi",
            modes={"a": ModeConfig(prompt="x", writes_scratchpad=False)},
        )


def test_orchestrator_modes_forbids_extra_keys() -> None:
    """A typo like ``action_budgt`` must raise, not be silently dropped."""
    with pytest.raises(ValidationError):
        OrchestratorModesConfig(
            orchestrator_prompt="hi",
            action_budgt=5,  # type: ignore[call-arg]
        )


def test_mode_config_forbids_extra_keys() -> None:
    """A typo on a mode field must raise."""
    with pytest.raises(ValidationError):
        ModeConfig(prompt="x", scratchpad_lable="y")  # type: ignore[call-arg]


def test_app_config_load_surfaces_typo_under_orchestrator_modes() -> None:
    """End-to-end: typo under ``[harness.orchestrator_modes]`` trips
    validation at ``AppConfig`` load, not at episode run-time."""
    from src.typing.config import AppConfig

    payload = _min_app_payload()
    payload["harness"] = {
        "name": "orchestrator_modes",
        "orchestrator_modes": {
            "orchestrator_prompt": "hi",
            "action_budgt": 5,  # typo
        },
    }
    with pytest.raises(ValidationError):
        AppConfig.model_validate(payload)


def test_harness_config_enforces_exactly_one_populated_section() -> None:
    """``name`` must match exactly one populated section. Two sections
    populated is a config smell — reject it."""
    with pytest.raises(ValidationError):
        HarnessConfig(
            name="orchestrator_modes",
            orchestrator_modes=OrchestratorModesConfig(orchestrator_prompt="a"),
            container={  # type: ignore[arg-type]
                "profile": "ms_agent",
                "image": "x",
            },
        )


def test_harness_config_section_must_match_name() -> None:
    """``name="container"`` without a populated ``container`` section must fail."""
    with pytest.raises(ValidationError):
        HarnessConfig(
            name="container",
            orchestrator_modes=OrchestratorModesConfig(orchestrator_prompt="a"),
        )
