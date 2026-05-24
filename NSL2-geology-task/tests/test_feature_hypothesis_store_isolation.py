"""Store-isolation tests for the feature-hypothesis pipeline.

Spec: ``docs/design/feature_hypothesis_voxel_store_isolation.md``.

These tests cover the per-episode scratch + admitted overlay layout that
replaces the previously-shared ``store_dir``. The failure modes they pin
were observed in a 35-episode parallel run (``20260524-pdgqar``):

- ``Spatial layer 'structural_proximity' not found in store`` after Slot A's
  ``scoring_create_feature_layer`` rename-in-place deleted a base-name
  layer Slot B was about to score.
- Joint scoring's ``layer_names`` returning the union of every slot's
  in-flight layers, polluting the BIC of any candidate.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

# Make sibling voxel-features-mcp imports work in this test suite.
_VFM = Path(__file__).resolve().parent.parent.parent / "voxel-features-mcp"
if str(_VFM) not in sys.path:
    sys.path.insert(0, str(_VFM))

from voxel_features.spatial import SpatialVoxelStore  # noqa: E402
from voxel_features.store import GridSpec  # noqa: E402
from voxel_features.mcp.tools.scoring_tools import (  # noqa: E402
    scoring_create_feature_layer,
)


GRID = GridSpec(
    origin=(0.0, 0.0, 0.0),
    maximum=(1.0, 1.0, 1.0),
    shape=(4, 4, 2),
)


def _store(scratch: Path, admitted: Path | None = None) -> SpatialVoxelStore:
    return SpatialVoxelStore(scratch, GRID, read_only_overlay=admitted)


def _values(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(GRID.shape).astype(float)


class TestConcurrentAddSameName:
    """Two slots adding a same-named layer to disjoint scratch dirs must
    each see only their own value — the symptom of the original bug was
    Slot B's add silently being clobbered by Slot A's index.json write."""

    def test_disjoint_scratch_dirs_isolate_writes(self, tmp_path: Path) -> None:
        scratch_a = tmp_path / "scratch" / "ep_A"
        scratch_b = tmp_path / "scratch" / "ep_B"
        values_a = _values(1)
        values_b = _values(2)

        errors: list[BaseException] = []

        def add(scratch: Path, values: np.ndarray) -> None:
            try:
                store = _store(scratch)
                store.add_layer(name="X", values=values, dtype="float")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t_a = threading.Thread(target=add, args=(scratch_a, values_a))
        t_b = threading.Thread(target=add, args=(scratch_b, values_b))
        t_a.start()
        t_b.start()
        t_a.join()
        t_b.join()

        assert not errors, f"unexpected errors: {errors!r}"

        store_a = _store(scratch_a)
        store_b = _store(scratch_b)
        np.testing.assert_array_equal(store_a.get_layer_values("X"), values_a)
        np.testing.assert_array_equal(store_b.get_layer_values("X"), values_b)


class TestScoringSeesOnlyAdmittedPlusScratch:
    """Joint scoring must only see (this episode's scratch) ∪ (admitted),
    never the union of every slot's scratch."""

    def test_layer_names_union(self, tmp_path: Path) -> None:
        admitted = tmp_path / "admitted"
        scratch = tmp_path / "scratch" / "ep_1"
        other_scratch = tmp_path / "scratch" / "ep_2"

        # Seed admitted with Y; seed *other* scratch with Z.
        admitted_store = _store(admitted)
        admitted_store.add_layer(name="Y", values=_values(10), dtype="float")
        other_store = _store(other_scratch)
        other_store.add_layer(name="Z_other", values=_values(20), dtype="float")

        # The episode-1 store sees its own scratch + admitted, NOT
        # ep_2's scratch.
        store = _store(scratch, admitted=admitted)
        store.add_layer(name="X", values=_values(30), dtype="float")

        assert set(store.layer_names) == {"X", "Y"}, (
            f"layer_names polluted by sibling scratch: {store.layer_names}"
        )

    def test_get_layer_values_falls_back_to_overlay(self, tmp_path: Path) -> None:
        admitted = tmp_path / "admitted"
        scratch = tmp_path / "scratch" / "ep_1"
        admitted_vals = _values(11)
        scratch_vals = _values(22)

        _store(admitted).add_layer(name="Y", values=admitted_vals, dtype="float")

        store = _store(scratch, admitted=admitted)
        store.add_layer(name="X", values=scratch_vals, dtype="float")

        np.testing.assert_array_equal(store.get_layer_values("X"), scratch_vals)
        np.testing.assert_array_equal(store.get_layer_values("Y"), admitted_vals)

    def test_scratch_shadows_overlay_on_collision(self, tmp_path: Path) -> None:
        # If the episode happens to spatial_add a layer named identically
        # to an admitted layer, the scratch values must win — admitted is
        # read-only from the episode's perspective.
        admitted = tmp_path / "admitted"
        scratch = tmp_path / "scratch" / "ep_1"
        admitted_vals = _values(11)
        scratch_vals = _values(22)

        _store(admitted).add_layer(
            name="structural_proximity", values=admitted_vals, dtype="float"
        )

        store = _store(scratch, admitted=admitted)
        store.add_layer(
            name="structural_proximity", values=scratch_vals, dtype="float"
        )

        np.testing.assert_array_equal(
            store.get_layer_values("structural_proximity"), scratch_vals
        )


class TestRemoveLayerIsScratchOnly:
    """``remove_layer`` must never delete from the overlay — that's how
    Slot A's rename-in-place obliterated the admitted layer Slot B needed."""

    def test_remove_overlay_only_layer_raises(self, tmp_path: Path) -> None:
        admitted = tmp_path / "admitted"
        scratch = tmp_path / "scratch" / "ep_1"
        _store(admitted).add_layer(name="Y", values=_values(1), dtype="float")

        store = _store(scratch, admitted=admitted)
        with pytest.raises(KeyError):
            store.remove_layer("Y")

        # Overlay layer must still be intact on a fresh read.
        assert "Y" in _store(scratch, admitted=admitted).layer_names

    def test_remove_scratch_layer_does_not_touch_overlay(
        self, tmp_path: Path
    ) -> None:
        admitted = tmp_path / "admitted"
        scratch = tmp_path / "scratch" / "ep_1"
        _store(admitted).add_layer(name="Y", values=_values(1), dtype="float")

        store = _store(scratch, admitted=admitted)
        store.add_layer(name="X", values=_values(2), dtype="float")
        store.remove_layer("X")

        assert "X" not in store.layer_names
        assert "Y" in store.layer_names


class TestScoringRenameIsLocalToScratch:
    """``scoring_create_feature_layer`` does a remove+add rename. Each
    slot's rename must stay inside its own scratch — Slot A's call
    must not delete the same-named candidate in Slot B's scratch."""

    def test_two_scratch_dirs_same_base_name(self, tmp_path: Path) -> None:
        scratch_a = tmp_path / "scratch" / "ep_A"
        scratch_b = tmp_path / "scratch" / "ep_B"

        store_a = _store(scratch_a)
        store_b = _store(scratch_b)
        store_a.add_layer(name="structural_proximity", values=_values(1), dtype="float")
        store_b.add_layer(name="structural_proximity", values=_values(2), dtype="float")

        # Slot A scores its candidate (renames base → base_<ts>); Slot B's
        # candidate must still be intact afterward.
        result_a = scoring_create_feature_layer(
            store_a, name="structural_proximity", dtype="float"
        )
        assert result_a["success"] is True

        # Slot B opens a fresh handle and the original name is still there.
        store_b_fresh = _store(scratch_b)
        assert "structural_proximity" in store_b_fresh.layer_names


class TestCleanupRemovesScratch:
    """The post-finalize cleanup hook must delete the episode's scratch
    dir even if the episode crashed mid-way."""

    def test_cleanup_removes_scratch_dir(self, tmp_path: Path) -> None:
        # The episode_id already carries an "ep_" prefix from
        # ``populate``, so the scratch dir name == episode_id.
        scratch = tmp_path / "scratch" / "ep_42"
        scratch.mkdir(parents=True, exist_ok=True)
        (scratch / "index.json").write_text("{}")
        (scratch / "layers").mkdir()
        (scratch / "layers" / "X.npy").write_bytes(b"\x00")

        from tasks.feature_hypothesis_kazakhstan import (
            FeatureHypothesisKazakhstanTask,
        )

        task = FeatureHypothesisKazakhstanTask({
            "store_dir": str(tmp_path / "store_root"),
            "kg_dir": str(tmp_path / "kg_root"),
        })
        task.cleanup_episode_resources({
            "store_dir": str(tmp_path),
            "episode_id": "ep_42",
        })

        assert not scratch.exists(), "scratch dir should be removed"

    def test_cleanup_is_idempotent(self, tmp_path: Path) -> None:
        from tasks.feature_hypothesis_kazakhstan import (
            FeatureHypothesisKazakhstanTask,
        )

        task = FeatureHypothesisKazakhstanTask({
            "store_dir": str(tmp_path / "store_root"),
            "kg_dir": str(tmp_path / "kg_root"),
        })
        # No scratch dir exists; cleanup must not raise.
        task.cleanup_episode_resources({
            "store_dir": str(tmp_path / "nope"),
            "episode_id": "ep_404",
        })


class TestAdmitPromotesScratchToAdmitted:
    """After ``_admit_with_dedup`` admits a (parents, hypothesis), the
    candidate's ``.npy`` must live under ``admitted/layers/``, not the
    episode's scratch."""

    def test_promotion_moves_npy(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "store" / "teniz_basin"
        scratch = store_dir / "scratch" / "ep_1"
        admitted = store_dir / "admitted"
        kg_dir = tmp_path / "kg" / "teniz_basin"

        store = _store(scratch, admitted=admitted)
        store.add_layer(name="candidate", values=_values(1), dtype="float")

        from tasks.feature_hypothesis_kazakhstan import (
            FeatureHypothesisKazakhstanTask,
        )

        task = FeatureHypothesisKazakhstanTask({
            "store_dir": str(tmp_path / "store_root"),
            "kg_dir": str(tmp_path / "kg_root"),
        })

        kg_record = {
            "node_id": "exp_1",
            "hypothesis": "h",
            "parent_node_1": None,
            "parent_node_2": None,
            "bic_delta": -10.0,
            "layer_name": "candidate",
            "artifact_links": {
                "layer_file": "store/teniz_basin/admitted/layers/candidate.npy"
            },
        }

        result = task._admit_with_dedup(
            kg_dir,
            kg_record,
            parents=[],
            hypothesis="h",
            scratch_dir=scratch,
            admitted_dir=admitted,
            layer_name="candidate",
        )
        assert result is True

        assert (admitted / "layers" / "candidate.npy").exists(), (
            "admitted .npy should exist after promotion"
        )
        assert not (scratch / "layers" / "candidate.npy").exists(), (
            "scratch .npy should be gone after promotion"
        )

        # Admitted index has the layer entry; a fresh store_dir overlay sees it.
        verify = _store(tmp_path / "verify_scratch", admitted=admitted)
        assert "candidate" in verify.layer_names
