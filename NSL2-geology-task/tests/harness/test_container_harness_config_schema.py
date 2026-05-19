"""``ContainerHarnessConfig`` is typed + per-profile config is late-validated.

Phase 2 introduces ``HarnessConfig.container: ContainerHarnessConfig | None``
with ``extra="forbid"``. The nested ``profile_config: dict[str, Any]`` stays
generic to keep the discriminated-union footprint small, but the loader
validates it against ``profile.profile_config_class`` at ``AppConfig`` load
so typos surface at the same layer as the outer pydantic model.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.typing.config import (
    AppConfig,
    ContainerHarnessConfig,
    HarnessConfig,
    OrchestratorModesConfig,
)


def _min_app_payload() -> dict:
    return {
        "model_name": "test",
        "code_host_cache_path": "./code/",
        "container_ids": ["svc-1"],
        "train_data_save_folder": "./data/",
    }


def test_container_defaults_and_minimum_fields() -> None:
    cfg = ContainerHarnessConfig(
        profile="ms_agent",
        image="nsl/ms-agent:0.1.0",
    )
    assert cfg.max_wall_seconds == 600
    assert cfg.mem_limit == "2g"
    assert cfg.network_mode == "bridge"
    assert cfg.inference_transport == "tcp"
    assert cfg.env == {}
    assert cfg.profile_config == {}


def test_container_rejects_unknown_network_mode() -> None:
    """Network mode is ``Literal["bridge", "none", "host"]``."""
    with pytest.raises(ValidationError):
        ContainerHarnessConfig(
            profile="ms_agent",
            image="x",
            network_mode="overlay",  # type: ignore[arg-type]
        )


def test_container_rejects_non_tcp_transport() -> None:
    """v1 narrows to TCP."""
    with pytest.raises(ValidationError):
        ContainerHarnessConfig(
            profile="ms_agent",
            image="x",
            inference_transport="uds",  # type: ignore[arg-type]
        )


def test_container_forbids_extra_keys() -> None:
    with pytest.raises(ValidationError):
        ContainerHarnessConfig(
            profile="ms_agent",
            image="x",
            max_wall_secnds=900,  # type: ignore[call-arg]
        )


def test_container_config_valid_loads_through_app_config() -> None:
    payload = _min_app_payload()
    payload["harness"] = {
        "name": "container",
        "container": {
            "profile": "ms_agent",
            "image": "nsl/ms-agent:0.1.0",
            "profile_config": {
                "model": "claude-sonnet-4-6",
                "max_chat_round": 60,
                "tool_call_timeout": 90,
                "transcript_tag": "episode",
            },
        },
    }
    app = AppConfig.model_validate(payload)
    assert app.harness.name == "container"
    assert app.harness.container is not None
    assert app.harness.container.profile == "ms_agent"


def test_app_config_surfaces_profile_config_typo_at_load() -> None:
    """Typo in ``profile_config`` trips the profile's model validation at
    ``AppConfig`` load — not at run_episode time."""
    payload = _min_app_payload()
    payload["harness"] = {
        "name": "container",
        "container": {
            "profile": "ms_agent",
            "image": "nsl/ms-agent:0.1.0",
            "profile_config": {
                "model": "claude-sonnet-4-6",
                "max_chat_roud": 60,  # typo
            },
        },
    }
    with pytest.raises(ValidationError):
        AppConfig.model_validate(payload)


def test_app_config_rejects_unknown_profile_name() -> None:
    payload = _min_app_payload()
    payload["harness"] = {
        "name": "container",
        "container": {
            "profile": "does_not_exist",
            "image": "x",
            "profile_config": {},
        },
    }
    with pytest.raises(ValidationError):
        AppConfig.model_validate(payload)


def test_harness_config_exactly_one_section_populated() -> None:
    """Both orchestrator_modes and container populated → reject."""
    with pytest.raises(ValidationError):
        HarnessConfig(
            name="container",
            orchestrator_modes=OrchestratorModesConfig(orchestrator_prompt="hi"),
            container=ContainerHarnessConfig(profile="ms_agent", image="x"),
        )


def test_harness_config_name_must_match_populated_section() -> None:
    with pytest.raises(ValidationError):
        HarnessConfig(
            name="container",
            # container section missing — name doesn't match any populated section
        )
