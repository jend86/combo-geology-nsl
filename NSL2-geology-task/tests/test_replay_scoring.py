"""Tests for ``scripts/replay_scoring.py``.

Covers the three contracts the replay tool must satisfy:

1. Replay reproduces seeded scoring (within numerical tolerance) for a
   fabricated run dir built from a known fixed code path.
2. Missing layer .npy → row is recorded as ``status=skipped_missing_layer``,
   no crash.
3. ``experiments.jsonl`` and the input store are byte-identical after replay.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from voxel_features.scoring import evaluate_new_layer
from voxel_features.store import GridSpec, VoxelStore


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "replay_scoring.py"

_GRID = GridSpec(
    origin=(0.0, 0.0, 0.0),
    maximum=(0.02, 0.02, 20.0),
    shape=(20, 20, 4),
)


def _seed_for_node(node_id: str) -> int:
    return int(hashlib.sha256(node_id.encode()).hexdigest()[:8], 16)


def _build_fake_run(tmp: Path, n_layers: int = 3) -> tuple[Path, Path, list[dict]]:
    """Build a minimal store + experiments.jsonl with ``n_layers`` admits.

    Returns ``(store_dir, kg_dir, rows)``. Each row in ``rows`` includes the
    ``node_id``, ``layer_name``, and (newly scored) ``bic_delta`` so tests can
    compare against the replayed values.
    """
    store_dir = tmp / "store" / "test_region"
    kg_dir = tmp / "kg" / "test_region"
    store_dir.mkdir(parents=True, exist_ok=True)
    kg_dir.mkdir(parents=True, exist_ok=True)
    layers_dir = store_dir / "admitted" / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    # Drive evaluate_new_layer to compute the canonical bic_delta we'll
    # compare against on replay.
    score_store = VoxelStore(tmp / "score_tmp", _GRID)
    rng = np.random.default_rng(0)
    rows: list[dict] = []

    for i in range(n_layers):
        node_id = f"node_{i}"
        layer_name = f"layer_{i}"
        # Each layer mildly correlated with the previous one.
        values = (rng.random((20, 20, 4)).astype(np.float32) + 0.5 * (i + 1) * 0.1)
        np.save(layers_dir / f"{layer_name}.npy", values)
        seed = _seed_for_node(node_id)
        result = evaluate_new_layer(score_store, layer_name, values, "float", seed=seed)
        bic_delta = result["bic_delta"]
        rows.append({
            "node_id": node_id,
            "layer_name": layer_name,
            "bic_delta": float(bic_delta) if bic_delta is not None else None,
            "masking_test_passed": bool(result["masking_test_passed"]),
            "masking_test_improvement": float(result["masking_test_improvement"]),
            "masking_test_direction": result["masking_test_direction"],
            "stage_completed": result["stage_completed"],
            "admission_path": result.get("admission_path", "normal"),
            "scoring_version": "two_stage_v2",
            "timestamp": "2026-05-25T00:00:00",
        })

    # Write experiments.jsonl
    with (kg_dir / "experiments.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    # Write admitted/index.json so replay can reconstruct the grid
    admitted_index = {
        "grid": {
            "origin": list(_GRID.origin),
            "maximum": list(_GRID.maximum),
            "shape": list(_GRID.shape),
            "crs": _GRID.crs,
        },
        "layers": {row["layer_name"]: {"name": row["layer_name"]} for row in rows},
    }
    (store_dir / "admitted" / "index.json").write_text(json.dumps(admitted_index))

    return store_dir, kg_dir, rows


def _hash_tree(path: Path) -> dict[str, str]:
    """Hash every file under ``path`` so we can detect any mutation."""
    out: dict[str, str] = {}
    for p in sorted(path.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(path))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _run_replay(store_dir: Path, kg_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "replay_report.jsonl"
    summary = out_dir / "replay_summary.json"
    proc = subprocess.run(
        [
            sys.executable, str(_SCRIPT),
            "--run-id", "test_run",
            "--store-dir", str(store_dir),
            "--kg-dir", str(kg_dir),
            "--out", str(out),
            "--summary-out", str(summary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert summary.exists(), f"replay did not write summary; stdout={proc.stdout!r}"
    return json.loads(summary.read_text())


# ---------------------------------------------------------------------------
# Test 9 — replay reproduces seeded scoring
# ---------------------------------------------------------------------------
def test_replay_reproduces_seeded_scoring(tmp_path: Path) -> None:
    store_dir, kg_dir, rows = _build_fake_run(tmp_path, n_layers=3)
    out_dir = tmp_path / "replay_out"
    summary = _run_replay(store_dir, kg_dir, out_dir)

    assert summary["error"] == 0, summary
    report = [json.loads(line) for line in (out_dir / "replay_report.jsonl").open()]
    assert len(report) == len(rows)

    for original, replayed in zip(rows, report):
        # Replay should reproduce the exact bic_delta (same seed, same code path)
        assert replayed["status"] == "ok", replayed
        assert replayed["replay"]["admission_path"] == original["admission_path"]
        if original["bic_delta"] is None:
            assert replayed["replay"]["bic_delta"] is None
        else:
            assert replayed["replay"]["bic_delta"] == pytest.approx(
                original["bic_delta"], abs=1e-9
            ), f"replay diverged from original for {original['node_id']}"


# ---------------------------------------------------------------------------
# Test 10 — missing layer .npy is reported, not raised
# ---------------------------------------------------------------------------
def test_replay_handles_missing_layer_file(tmp_path: Path) -> None:
    store_dir, kg_dir, rows = _build_fake_run(tmp_path, n_layers=2)
    # Delete one .npy
    target = rows[1]["layer_name"]
    (store_dir / "admitted" / "layers" / f"{target}.npy").unlink()

    out_dir = tmp_path / "replay_out"
    summary = _run_replay(store_dir, kg_dir, out_dir)
    assert summary["skipped_missing_layer"] == 1, summary
    assert summary["error"] == 0, summary


# ---------------------------------------------------------------------------
# Test 11 — inputs are byte-identical after replay
# ---------------------------------------------------------------------------
def test_replay_does_not_mutate_inputs(tmp_path: Path) -> None:
    store_dir, kg_dir, _rows = _build_fake_run(tmp_path, n_layers=2)
    store_before = _hash_tree(store_dir)
    kg_before = _hash_tree(kg_dir)

    out_dir = tmp_path / "replay_out"
    _run_replay(store_dir, kg_dir, out_dir)

    store_after = _hash_tree(store_dir)
    kg_after = _hash_tree(kg_dir)
    assert store_before == store_after, "replay mutated store dir"
    assert kg_before == kg_after, "replay mutated kg dir"
