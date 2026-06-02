from __future__ import annotations

import json
import tomllib
from pathlib import Path

from src.task.types import CapabilityExecutionContext
from tasks.feature_hypothesis_kazakhstan import FeatureHypothesisKazakhstanTask


def _task(tmp_path: Path) -> FeatureHypothesisKazakhstanTask:
    return FeatureHypothesisKazakhstanTask(
        {
            "dataset_dir": str(tmp_path / "dataset"),
            "store_dir": str(tmp_path / "store"),
            "kg_dir": str(tmp_path / "knowledge"),
            "artifact_dir": str(tmp_path / "artifacts"),
        }
    )


def _ctx(tmp_path: Path, *, phase_records: dict) -> CapabilityExecutionContext:
    return CapabilityExecutionContext(
        episode_id="framework_ep_1",
        workflow_step="rewrite",
        episode_context={
            "episode_id": "task_ep_1",
            "framework_episode_id": "framework_ep_1",
            "store_dir": str(tmp_path / "store" / "teniz_basin"),
            "kg_dir": str(tmp_path / "knowledge" / "teniz_basin"),
            "phase_records": phase_records,
        },
    )


def test_prompt_names_true_two_stage_gate_and_support_matching(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = task.list_variations()[0]

    spec = task.prompt_spec(
        variation,
        {"episode_id": "ep_1", "workflow_kind": "survey", "n_features": 3},
    )

    prompt = spec.system_instruction
    assert "A feature layer is admitted if bic_delta < 0." not in prompt
    assert "Stage 1" in prompt
    assert "MAE" in prompt
    assert "Stage 2" in prompt
    assert "bic_delta < 0" in prompt
    assert "Prefer a dense prospectivity surface" not in prompt
    assert "match spatial support to the geological claim" in prompt
    assert "Dense basin-scale surfaces" in prompt
    assert "sparse localized layers are acceptable" in prompt


def test_episode_constraints_use_overflow_recovery_caps(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = task.list_variations()[0]

    constraints = task.episode_constraints(variation, {"workflow_kind": "survey"})

    assert constraints.budgets.max_task_tool_calls == 60
    assert constraints.budgets.max_llm_turns == 45


def test_external_aiq_config_uses_shorter_iteration_cap() -> None:
    repo = Path(__file__).resolve().parents[1]
    config_path = repo / "config" / "config-feature-hypothesis-kazakhstan-external.toml"

    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert config["harness"]["container"]["tool_output_max_chars"] == 2500
    assert config["harness"]["container"]["context_compaction_enabled"] is True
    assert config["harness"]["container"]["context_compaction_trigger_tokens"] == 52000
    assert config["harness"]["container"]["context_compaction_target_tokens"] == 45000
    assert config["harness"]["container"]["profile_config"]["max_iterations"] == 28


def test_stage1_rejected_candidate_is_quarantined_without_kg_admit(tmp_path: Path) -> None:
    task = _task(tmp_path)
    layer_name = "rejected_redox_surface_123"
    phase_records = {
        "hypothesise": {
            "hypothesis": "Reduced-facies redox surface predicts copper prospectivity.",
            "data_spec": {"features": ["redox"]},
        },
        "code": {"result_summary": "built candidate layer"},
        "translate": {"feature_layer_name": layer_name},
        "evaluate": {
            "layer_name": layer_name,
            "bic_delta": -2.5,
            "bic_delta_raw": -250.0,
            "n_effective_samples_before": 100,
            "n_effective_samples_after": 140,
            "n_effective_samples_delta": 40,
            "candidate_nonzero_voxels": 224,
            "candidate_fill_fraction": 0.0007,
            "admitted": False,
            "masking_test_passed": False,
            "masking_test_improvement": -0.0012,
            "masking_test_direction": "mae_delta",
            "stage_completed": "mae_bic_completed",
        },
    }
    ctx = _ctx(tmp_path, phase_records=phase_records)
    scratch_layers = tmp_path / "store" / "teniz_basin" / "scratch" / "task_ep_1" / "layers"
    scratch_layers.mkdir(parents=True)
    original_bytes = b"not-a-real-npy-but-preserved-verbatim"
    (scratch_layers / f"{layer_name}.npy").write_bytes(original_bytes)

    result = task._exec_submit_rewrite(
        [],
        {"prompt": "Why this layer?", "response": "Because redox matters."},
        ctx,
    )

    assert result.success is True
    assert result.output["knowledge_saved"] is False
    assert result.output["rejected_candidate_quarantined"] is True
    assert not (tmp_path / "knowledge" / "teniz_basin" / "experiments.jsonl").exists()

    quarantine_dir = tmp_path / "store" / "teniz_basin" / "rejected" / "task_ep_1"
    quarantined_layer = quarantine_dir / f"{layer_name}.npy"
    metadata_path = quarantine_dir / "metadata.json"

    assert quarantined_layer.read_bytes() == original_bytes
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["episode_id"] == "task_ep_1"
    assert metadata["layer_name"] == layer_name
    assert metadata["rejection_stage"] == "stage_1"
    assert metadata["evaluate"]["n_effective_samples_delta"] == 40
