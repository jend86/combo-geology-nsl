#!/usr/bin/env python3
"""Replay scoring over a recorded run's experiments.jsonl with the post-fix code.

Walks ``experiments.jsonl`` in append order, reconstructs the VoxelStore state
at each admit step (loading prior layers' ``.npy`` files from
``store/<region>/admitted/layers/``), and re-runs ``evaluate_new_layer`` under
the current (post-fix, post-calibration, post-RNG-seeding) scoring code with a
stable per-episode seed derived from the row's ``node_id``.

What this produces
------------------
- ``<out>`` (default ``replay_report.jsonl``): one row per replay attempt with
  ``node_id``, ``layer_name``, original ``bic_delta``, replayed ``bic_delta``,
  replayed ``stage1_passed`` / ``stage2_passed``, and a ``status`` field.
- ``<out_summary>``: aggregate counts (``ok`` / ``skipped_missing_layer`` /
  ``error``) plus the fraction of historical admits that would survive the
  fixed gate.

What this does NOT do
---------------------
- Mutate any input file. ``experiments.jsonl`` and per-episode JSONs are
  untouched.
- Simulate the agent. Replay only re-scores layers that were already admitted
  — rejections are gone (no rejected ``.npy`` files survive the original
  workflow), so we cannot tell from this data alone how many *rejections*
  would change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _seed_for_node(node_id: str) -> int:
    """Stable seed in [0, 2**32) derived from the row's node_id.

    Uses sha256 rather than Python's hash() (which is salted per-process).
    """
    return int(hashlib.sha256(node_id.encode()).hexdigest()[:8], 16)


@dataclass
class ReplayRow:
    node_id: str
    layer_name: str
    original_bic_delta: float | None
    original_admitted: bool
    replay_bic_delta: float | None
    replay_stage1_passed: bool | None
    replay_admitted: bool | None
    replay_admission_path: str | None
    status: str  # ok | skipped_missing_layer | error
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "layer_name": self.layer_name,
            "original": {
                "bic_delta": self.original_bic_delta,
                "admitted": self.original_admitted,
            },
            "replay": {
                "bic_delta": self.replay_bic_delta,
                "stage1_passed": self.replay_stage1_passed,
                "admitted": self.replay_admitted,
                "admission_path": self.replay_admission_path,
            },
            "status": self.status,
            "error": self.error,
        }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"warn: malformed jsonl row: {e}", file=sys.stderr)
    return rows


def _resolve_layer_npy(
    store_dir: Path,
    layer_name: str,
) -> Path | None:
    """Find the .npy for an admitted layer; tolerate the two known layouts."""
    for candidate in (
        store_dir / "admitted" / "layers" / f"{layer_name}.npy",
        store_dir / "layers" / f"{layer_name}.npy",
    ):
        if candidate.exists():
            return candidate
    return None


def replay(
    run_id: str,
    store_dir: Path,
    kg_dir: Path,
    out_path: Path,
    summary_path: Path,
) -> None:
    # Imports are deferred so the script can show --help without numpy/scipy.
    import numpy as np
    from voxel_features.store import GridSpec, VoxelStore
    from voxel_features.scoring import evaluate_new_layer

    rows = _load_jsonl(kg_dir / "experiments.jsonl")
    if not rows:
        print(f"no rows in {kg_dir/'experiments.jsonl'}; nothing to replay")
        return

    # Reconstruct the grid spec from the store's admitted/index.json so the
    # replayed coherence calculation uses the real coordinate system.
    admitted_index_path = store_dir / "admitted" / "index.json"
    if not admitted_index_path.exists():
        print(f"missing {admitted_index_path}; cannot reconstruct grid")
        return
    admitted_index = json.loads(admitted_index_path.read_text())
    g = admitted_index["grid"]
    grid = GridSpec(
        origin=tuple(g["origin"]),
        maximum=tuple(g["maximum"]),
        shape=tuple(g["shape"]),
        crs=g.get("crs"),
    )

    replay_rows: list[ReplayRow] = []

    # We replay into a fresh tmp store so the original spatial.db / admitted
    # index is never touched.
    with tempfile.TemporaryDirectory(prefix=f"replay-{run_id}-") as tmp:
        replay_store = VoxelStore(Path(tmp), grid)

        for row in rows:
            node_id = str(row.get("node_id") or "")
            layer_name = str(row.get("layer_name") or "")
            original_bic = row.get("bic_delta")
            original_admission_path = row.get("admission_path")
            original_admitted = (
                original_admission_path == "first_layer_auto"
                or (
                    bool(row.get("masking_test_passed", False))
                    and original_bic is not None
                    and original_bic < 0.0
                )
            )

            if not layer_name:
                replay_rows.append(ReplayRow(
                    node_id=node_id,
                    layer_name=layer_name,
                    original_bic_delta=original_bic,
                    original_admitted=original_admitted,
                    replay_bic_delta=None,
                    replay_stage1_passed=None,
                    replay_admitted=None,
                    replay_admission_path=None,
                    status="error",
                    error="missing layer_name in row",
                ))
                continue

            npy_path = _resolve_layer_npy(store_dir, layer_name)
            if npy_path is None:
                replay_rows.append(ReplayRow(
                    node_id=node_id,
                    layer_name=layer_name,
                    original_bic_delta=original_bic,
                    original_admitted=original_admitted,
                    replay_bic_delta=None,
                    replay_stage1_passed=None,
                    replay_admitted=None,
                    replay_admission_path=None,
                    status="skipped_missing_layer",
                    error=f"no .npy for {layer_name}",
                ))
                continue

            try:
                values = np.load(npy_path)
                if values.shape != tuple(grid.shape):
                    # Some legacy artefacts were saved flattened.
                    values = values.reshape(grid.shape)
                seed = _seed_for_node(node_id or layer_name)
                result = evaluate_new_layer(
                    replay_store,
                    layer_name=layer_name,
                    layer_values=values.astype(np.float32),
                    layer_dtype="float",
                    seed=seed,
                )
                replay_bic = result.get("bic_delta")
                replay_rows.append(ReplayRow(
                    node_id=node_id,
                    layer_name=layer_name,
                    original_bic_delta=original_bic,
                    original_admitted=original_admitted,
                    replay_bic_delta=float(replay_bic) if replay_bic is not None else None,
                    replay_stage1_passed=bool(result["masking_test_passed"]),
                    replay_admitted=bool(result["admitted"]),
                    replay_admission_path=result.get("admission_path"),
                    status="ok",
                ))
            except Exception as e:  # noqa: BLE001 — replay should never crash
                replay_rows.append(ReplayRow(
                    node_id=node_id,
                    layer_name=layer_name,
                    original_bic_delta=original_bic,
                    original_admitted=original_admitted,
                    replay_bic_delta=None,
                    replay_stage1_passed=None,
                    replay_admitted=None,
                    replay_admission_path=None,
                    status="error",
                    error=str(e),
                ))

    # Write report.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in replay_rows:
            fh.write(json.dumps(r.to_dict()) + "\n")

    n_total = len(replay_rows)
    n_ok = sum(1 for r in replay_rows if r.status == "ok")
    n_skipped = sum(1 for r in replay_rows if r.status == "skipped_missing_layer")
    n_error = sum(1 for r in replay_rows if r.status == "error")
    n_survives = sum(
        1 for r in replay_rows
        if r.status == "ok" and r.replay_admitted
    )

    summary = {
        "run_id": run_id,
        "total_rows": n_total,
        "ok": n_ok,
        "skipped_missing_layer": n_skipped,
        "error": n_error,
        "replay_admit_count": n_survives,
        "replay_admit_rate_among_ok": (n_survives / n_ok) if n_ok else 0.0,
        "out_path": str(out_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--store-dir", required=True, type=Path)
    parser.add_argument("--kg-dir", required=True, type=Path)
    parser.add_argument("--out", default=Path("replay_report.jsonl"), type=Path)
    parser.add_argument("--summary-out", default=Path("replay_summary.json"), type=Path)
    args = parser.parse_args()

    replay(
        run_id=args.run_id,
        store_dir=args.store_dir.resolve(),
        kg_dir=args.kg_dir.resolve(),
        out_path=args.out.resolve(),
        summary_path=args.summary_out.resolve(),
    )


if __name__ == "__main__":
    main()
