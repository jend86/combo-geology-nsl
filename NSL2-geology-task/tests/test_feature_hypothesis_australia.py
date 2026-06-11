"""Australia (Coe Fairbairn) feature-hypothesis task — parametrization invariants.

These pin the region/dataset-specific surface that distinguishes the Australia
task from its Kazakhstan twin. Written TDD-first: every assertion here fails on
a verbatim clone of ``feature_hypothesis_kazakhstan.py`` (Kazakh grid at
66–71°E, 16 Kazakh source files, name ``feature-hypothesis-kazakhstan``, Kazakh
geology tokens in source) and passes only after the Australia parametrization.

Run from ``NSL2-geology-task/`` with the nix-ld ``ln`` shim (numpy native libs)::

    ln -c uv run pytest tests/test_feature_hypothesis_australia.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import tasks.feature_hypothesis_australia as m
from tasks.feature_hypothesis_australia import (
    FeatureHypothesisAustraliaProposerRows,
    FeatureHypothesisAustraliaState,
    FeatureHypothesisAustraliaTask,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent          # NSL2-geology-task/
_DATASET_DIR = _REPO_ROOT.parent / "Australian_data"         # sibling raw dataset
_AMALG = _DATASET_DIR / "amalgamated_csvs"

_GRID = m._AUSTRALIA_COE_GRID
_SOURCES = m._AUSTRALIA_SOURCE_FILES


def _iter_lonlat(csv_path: Path):
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                yield float(row["longitude"]), float(row["latitude"])
            except (KeyError, ValueError, TypeError):
                continue


class TestGridEnclosesData:
    def test_crs_and_shape_unchanged(self) -> None:
        assert _GRID["crs"] == "EPSG:4326"
        assert _GRID["shape"] == [200, 200, 8]
        # Depth kept 0–80 m (drillhole depth is agent-synthesised; see design doc).
        assert _GRID["origin"][2] == 0.0 and _GRID["maximum"][2] == 80.0

    def test_grid_is_western_australia_not_kazakhstan(self) -> None:
        # Coe Fairbairn sits at ~117.9°E / −27.3° (WA). The Kazakh grid was
        # 66–71°E / 49–52°N — a clone that forgot to re-grid fails here.
        lon0, lat0, _ = _GRID["origin"]
        lon1, lat1, _ = _GRID["maximum"]
        assert 117.0 < lon0 < lon1 < 119.0
        assert -28.0 < lat0 < lat1 < -27.0

    @pytest.mark.parametrize("name", ["geochemDrillhole.csv", "geochemSurface.csv"])
    def test_all_sample_points_inside_grid(self, name: str) -> None:
        path = _AMALG / name
        if not path.exists():
            pytest.skip(f"dataset not present at {path}")
        lon0, lat0, _ = _GRID["origin"]
        lon1, lat1, _ = _GRID["maximum"]
        n = 0
        for lon, lat in _iter_lonlat(path):
            n += 1
            assert lon0 <= lon <= lon1, f"{name}: lon {lon} outside [{lon0}, {lon1}]"
            assert lat0 <= lat <= lat1, f"{name}: lat {lat} outside [{lat0}, {lat1}]"
        assert n > 0, f"{name} had no readable longitude/latitude rows"


class TestRotationAnchors:
    def test_exactly_five_single_file_agent_guides(self) -> None:
        assert len(_SOURCES) == 5
        keys = [s["key"] for s in _SOURCES]
        assert len(set(keys)) == 5, "rotation keys must be unique (state is keyed on them)"
        for s in _SOURCES:
            assert s["path"].endswith(".md")
            assert "AGENT_GUIDE" in s["path"]
            assert not s.get("glob_pattern"), "guides are single-file anchors, no glob"

    def test_guide_paths_resolve_under_dataset_dir(self) -> None:
        if not _DATASET_DIR.exists():
            pytest.skip("dataset not present")
        for s in _SOURCES:
            assert (_DATASET_DIR / s["path"]).is_file(), s["path"]

    def test_round_robin_cycles_all_before_repeat(self, tmp_path) -> None:
        task = FeatureHypothesisAustraliaTask.__new__(FeatureHypothesisAustraliaTask)
        kg = str(tmp_path)
        picks = [
            task._pick_assigned_source(kg, _SOURCES)["source"]["key"] for _ in range(5)
        ]
        assert sorted(picks) == sorted(s["key"] for s in _SOURCES), (
            "first 5 picks must cover all 5 guides (least-explored round-robin)"
        )
        sixth = task._pick_assigned_source(kg, _SOURCES)["source"]["key"]
        assert sixth == picks[0], "6th pick wraps to the first (ties broken by order)"

    def test_assigned_guide_content_is_injected(self) -> None:
        if not _DATASET_DIR.exists():
            pytest.skip("dataset not present")
        task = FeatureHypothesisAustraliaTask.__new__(FeatureHypothesisAustraliaTask)
        sample = task._read_source_sample(_SOURCES[0], str(_DATASET_DIR))
        assert sample, "expected a non-empty content sample for the assigned guide"
        assert "WAMEX" in sample, "the AGENT_GUIDE header should be injected verbatim"


class TestRegistration:
    def test_load_task_builds_and_validates(self, tmp_path) -> None:
        from src.task.loader import load_task

        cfg = {
            "dataset_dir": str(_DATASET_DIR),
            "store_dir": str(tmp_path / "store"),
            "kg_dir": str(tmp_path / "kg"),
            "docker_compose_dir": "docker/feature-hypothesis-australia-compose",
        }
        task = load_task(
            "tasks.feature_hypothesis_australia.FeatureHypothesisAustraliaTask", cfg
        )
        assert task.name == "feature-hypothesis-australia"
        variations = task.list_variations()
        assert len(variations) == 1
        assert variations[0].name == "coe_fairbairn"
        assert variations[0].grid_spec["crs"] == "EPSG:4326"
        assert variations[0].grid_spec["origin"] == _GRID["origin"]


class TestNoRegionBleed:
    def test_module_source_has_no_kazakh_geology_tokens(self) -> None:
        src = Path(m.__file__).read_text()
        for tok in (
            "Teniz", "Smolianova", "USGS", "copper", "Cu_pct", "chalcopyrite",
            "copper_prospects", "vladimirov", "Soviet", "Kazkhstan",
            "sandstone copper",
        ):
            assert tok not in src, f"residual Kazakh token in module source: {tok!r}"

    def test_only_intentional_kazakhstan_reference_is_the_sft_twin(self) -> None:
        # The one allowed mention is the ProposerRows docstring naming the
        # sibling Kazakhstan class (recipe-hash provenance). Anything more is bleed.
        src = Path(m.__file__).read_text()
        assert src.count("Kazakhstan") <= 1


class TestPromptParity:
    """Australia prompts should match Kazakhstan's neutral source-catalog style."""

    def test_system_prompt_is_not_target_recipe(self) -> None:
        text = m._SYSTEM_PROMPT
        lowered = text.lower()
        assert "gold project" not in lowered
        assert "calcrete gold" not in lowered
        assert "supergene blankets" not in lowered
        assert "shears, lodes" not in lowered
        assert "western australian mineral exploration data" in lowered
        assert "project-scale" in lowered

    def test_dataset_overview_avoids_directive_targeting(self) -> None:
        text = m._DATASET_OVERVIEW
        for phrase in (
            "START HERE",
            "deposit-scale GOLD",
            "ground-truth",
            "geologically validated",
            "Commodity & setting",
            "Pathfinders:",
        ):
            assert phrase not in text
        assert "multi-element" in text
        assert "recorded mineral occurrences" in text

    def test_rotation_source_descriptions_are_neutral(self) -> None:
        text = "\n".join(str(source["description"]) for source in _SOURCES)
        for phrase in (
            "Au assays",
            "known occurrences",
            "known Au",
            "Hosts most",
            "Strong drillhole + surface coverage",
        ):
            assert phrase not in text
        assert "WAMEX knowledge base" in text
        assert "report context" in text

    def test_workflow_prompts_use_project_not_basin_language(self) -> None:
        task = FeatureHypothesisAustraliaTask.__new__(FeatureHypothesisAustraliaTask)
        fallback = task._generate_explore_prompt({})
        assert "project-scale feature opportunity" in fallback
        assert "regional feature opportunity" not in fallback

        variation = m.FeatureHypothesisAustraliaVariation(name="coe_fairbairn", description="test")
        workflow = task._survey_workflow(variation, {})
        all_prompts = "\n".join(step.prompt for step in workflow.steps)
        assert "Australia basin analysis" not in all_prompts
        assert "over the basin" not in all_prompts
        assert "over the project grid" in all_prompts
        assert "Coe Fairbairn project area" in all_prompts

    def test_enhanced_data_spec_removes_target_recipe_language(self) -> None:
        task = FeatureHypothesisAustraliaTask.__new__(FeatureHypothesisAustraliaTask)
        out = task._enhance_data_spec({"files": []})
        text = json.dumps(out, sort_keys=True)
        for phrase in (
            "PRIMARY",
            "ground-truth",
            "geologically validated",
            "Known gold occurrences",
            "commodity",
            "pathfinder_elements",
        ):
            assert phrase not in text
        assert "au_ppm" in text
        assert "multi-element assay" in text


class TestSftIdentity:
    def test_proposer_rows_tag_is_australia_and_distinct(self) -> None:
        tag = FeatureHypothesisAustraliaProposerRows().name
        assert tag == "FeatureHypothesisAustraliaProposerRows[v1]"
        # Must differ from the Kazakhstan twin so export_recipe.json hashes are distinct.
        from tasks.feature_hypothesis_kazakhstan import (
            FeatureHypothesisKazakhstanProposerRows,
        )

        assert tag != FeatureHypothesisKazakhstanProposerRows().name


class TestRewardAndKgGate:
    """Region-agnostic reward + KG-gate behaviour, homed on the Australia task
    (ported from the retired base task's first_layer_none test). The shared
    logic is identical to the Kazakhstan twin; this pins it for Australia."""

    def test_first_layer_auto_reward_uses_none_bic_delta(self) -> None:
        task = FeatureHypothesisAustraliaTask.__new__(FeatureHypothesisAustraliaTask)
        final = FeatureHypothesisAustraliaState(
            bic_delta=None,
            admitted=True,
            masking_test_passed=True,
            masking_test_improvement=0.0,
            masking_test_direction="first_layer_auto",
            stage_completed="first_layer_auto",
            admission_path="first_layer_auto",
        )
        reward = task.compute_reward(FeatureHypothesisAustraliaState(), final, None)
        assert reward.success is True
        assert reward.value == 1.0
        assert reward.breakdown["first_layer_auto"] is True
        assert reward.breakdown["bic_delta"] is None

    def test_kg_gate_allows_first_layer_auto_without_bic_delta(self) -> None:
        assert FeatureHypothesisAustraliaTask._should_persist_to_kg(
            masking_test_passed=True,
            admitted=True,
            bic_delta=None,
            stage_completed="first_layer_auto",
            admission_path="first_layer_auto",
        )

    def test_kg_gate_allows_current_mae_bic_stage(self) -> None:
        assert FeatureHypothesisAustraliaTask._should_persist_to_kg(
            masking_test_passed=True,
            admitted=True,
            bic_delta=-1.0,
            stage_completed="mae_bic_completed",
        )

    def test_kg_gate_blocks_none_bic_for_normal_path(self) -> None:
        assert not FeatureHypothesisAustraliaTask._should_persist_to_kg(
            masking_test_passed=True,
            admitted=True,
            bic_delta=None,
            stage_completed="mae_bic_completed",
        )


class TestDegenerateFillGate:
    """Survey-phase rejection of the degenerate full-grid CONSTANT layer.

    The smoke test admitted a layer that filled all 320,000 voxels with the
    single value 1.0 (``fill_fraction=1.0``, ``uniform_nonzero_value=True``) via
    ``first_layer_auto`` — a data-starved fallback blob. The gate lives in the
    survey-only ``_seed_phase_admission_ok`` (a no-op in crossbreed, where MAE/BIC
    governs) and rejects ONLY full-fill + uniform, so it does not penalise
    (a) the agent's normal sparse-uniform seeds or (b) full continuous fields
    from ``spatial_set_layer_array``. Written TDD-first: fails before the gate.
    """

    @staticmethod
    def _kg_record_for(values):
        # Multi-op, artifact-backed provenance so the OTHER seed-gate reasons
        # (all_creative_fallback, single_spatial_operation) cannot fire — this
        # isolates the new degenerate-fill rule.
        kg = {
            "spatial_operation_provenance_count": 3,
            "coordinate_source_counts": {"artifact": 3},
            "geometry_kind_counts": {},
        }
        FeatureHypothesisAustraliaTask._stamp_candidate_triviality(kg, values=values)
        return kg

    def test_full_grid_constant_fill_rejected_in_survey(self) -> None:
        import numpy as np

        full_const = np.ones((200, 200, 8), dtype=float)  # 100% fill, single value
        kg = self._kg_record_for(full_const)
        assert kg["candidate_fill_fraction"] == 1.0
        assert kg["uniform_nonzero_value"] is True
        ok = FeatureHypothesisAustraliaTask._seed_phase_admission_ok(kg, seed_phase=True)
        assert ok is False
        assert kg["first_root_rejection_reason"] == "degenerate_fill"

    def test_sparse_uniform_seed_admitted_in_survey(self) -> None:
        import numpy as np

        sparse = np.zeros((200, 200, 8), dtype=float)
        sparse[10:30, 10:30, 0:2] = 1.0  # ~0.25% fill, uniform value — a fine seed
        kg = self._kg_record_for(sparse)
        assert kg["candidate_fill_fraction"] < 0.95
        ok = FeatureHypothesisAustraliaTask._seed_phase_admission_ok(kg, seed_phase=True)
        assert ok is True
        assert kg["first_root_rejection_reason"] == "none"

    def test_full_continuous_field_admitted_in_survey(self) -> None:
        import numpy as np

        # Full-fill but VARYING (a continuous prospectivity surface from
        # spatial_set_layer_array) — must NOT be rejected.
        field = np.linspace(1.0, 100.0, 200 * 200 * 8).reshape((200, 200, 8))
        kg = self._kg_record_for(field)
        assert kg["candidate_fill_fraction"] == 1.0
        assert kg["uniform_nonzero_value"] is False
        ok = FeatureHypothesisAustraliaTask._seed_phase_admission_ok(kg, seed_phase=True)
        assert ok is True

    def test_full_constant_fill_is_telemetry_only_in_crossbreed(self) -> None:
        import numpy as np

        full_const = np.ones((200, 200, 8), dtype=float)
        kg = self._kg_record_for(full_const)
        # Crossbreed (seed_phase=False): gate is a no-op; MAE/BIC handles it.
        ok = FeatureHypothesisAustraliaTask._seed_phase_admission_ok(kg, seed_phase=False)
        assert ok is True
        assert kg["first_root_rejection_reason"] == "none"

    def test_empty_array_op_seed_rejected_in_survey(self) -> None:
        import numpy as np

        # All-zero layer with op_count>0 (e.g. a set_layer_array of a grid the
        # agent's code never populated) — slips declared_nothing (op_count!=0)
        # but carries no signal. The 2026-06-11 re-smoke admitted exactly this.
        empty = np.zeros((200, 200, 8), dtype=float)
        kg = self._kg_record_for(empty)
        assert kg["candidate_fill_fraction"] == 0.0
        assert kg["uniform_nonzero_value"] is False
        ok = FeatureHypothesisAustraliaTask._seed_phase_admission_ok(kg, seed_phase=True)
        assert ok is False
        assert kg["first_root_rejection_reason"] == "degenerate_empty"

    def test_empty_seed_is_telemetry_only_in_crossbreed(self) -> None:
        import numpy as np

        empty = np.zeros((200, 200, 8), dtype=float)
        kg = self._kg_record_for(empty)
        ok = FeatureHypothesisAustraliaTask._seed_phase_admission_ok(kg, seed_phase=False)
        assert ok is True
