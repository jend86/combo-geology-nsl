#!/usr/bin/env python3
"""Apply the post-fix scoring's verdicts to an existing run's persisted state.

Given a replay report from ``scripts/replay_scoring.py``, mutate the run dir
so a resumed run sees only the experiments that survive the fixed gate.

What gets rewritten
-------------------
For each row in ``experiments.jsonl`` we look up the replay verdict:

  - **ok + replay_admitted**: kept; the scoring-related fields
    (``bic_delta``, ``masking_test_passed``, ``masking_test_improvement``,
    ``masking_test_direction``, ``scoring_version``) are overwritten with
    the replay values, so a resumed run treats the row as if it had been
    scored under the new code.
  - **anything else** (replay rejected; replay error; missing .npy): dropped.
    These rows cannot be defended under the fixed gate and dropping them
    avoids polluting the parent pool / crossbreed lineage.

Cascade
-------
Once the surviving set ``S`` is fixed, the following are rewritten:

  - ``experiments.jsonl``     — keep only rows whose ``node_id`` ∈ S; null
                                out ``parent_node_1`` / ``parent_node_2``
                                refs that point outside S (dangling).
  - ``admitted_index.json``   — recompute fingerprints
                                (``_fingerprint(parents, hypothesis)`` from
                                ``feature_hypothesis_kazakhstan.py``) over
                                survivors; replace ``fingerprints``.
  - ``crossbreed_index.jsonl``— drop pairs where ``node_1`` or ``node_2``
                                ∉ S.
  - ``crossbreed_queue.jsonl``— drop queued pairs where any parent ∉ S.
  - ``store/<region>/admitted/index.json`` — keep only ``layers`` entries
                                whose name ∈ {survivor layer_names}.
  - Orphan ``.npy`` files in ``store/<region>/admitted/layers/`` (i.e. not
    referenced by any survivor) are deleted.

Optional add-ons (off by default)
--------------------------------
``--training-pickle PATH`` — filter ``training_pairs.pkl`` in place. Rows
whose ``episode_id`` corresponds to a *dropped* KG node_id are removed.
Rows whose episode_id never appeared in the KG ledger (successful-but-not-
admitted episodes) are kept untouched.

``--checkpoint PATH`` — overwrite the generation checkpoint's success
counters (``total_successful``, ``training_row_count``,
``raw_successful_row_count``, ``total_score``, ``success_rate``) by
scaling them by ``kept / before`` so the resumed run sees post-fix-clean
progress (and continues toward ``target_training_rows``). Requires
``--training-pickle`` to compute the keep ratio.

What this does NOT mutate
-------------------------
- ``store/<region>/admitted/spatial.db`` (sqlite log of spatial ops).
- ``store/<region>/layers/`` (the legacy path).
- ``generations/<run_id>/<gen>/successful/`` ``failed/`` ``all_episodes.jsonl``
  — those are per-episode artefacts that aren't read on resume. They stay
  as historical record.

Always make a backup of the run dir first; this script overwrites in place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Mirror of ``feature_hypothesis_kazakhstan._fingerprint`` — kept local to
# avoid pulling the task module (heavy imports). If that algorithm ever
# changes upstream, update this mirror.
def _fingerprint(parents: list[str] | None, hypothesis: str) -> str:
    payload = "|".join(parents or []) + "::" + re.sub(r"\s+", " ", (hypothesis or "")).strip()
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Survivor:
    node_id: str
    layer_name: str
    replay: dict[str, Any]


def _load_replay(path: Path) -> dict[str, Survivor]:
    out: dict[str, Survivor] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("replay", {}).get("admitted"):
                out[row["node_id"]] = Survivor(
                    node_id=row["node_id"],
                    layer_name=row["layer_name"],
                    replay=row["replay"],
                )
    return out


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    tmp.replace(path)


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def filter_experiments(
    rows: list[dict[str, Any]],
    survivors: dict[str, Survivor],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep only survivor rows; overlay replayed scoring fields; null
    out dangling parent references.

    Returns (filtered_rows, stats).
    """
    kept: list[dict[str, Any]] = []
    survivor_ids = set(survivors)
    dangling_nulled = 0

    for row in rows:
        node_id = row.get("node_id")
        if node_id not in survivor_ids:
            continue
        replay = survivors[node_id].replay
        new_row = dict(row)
        # Overlay replayed scoring so downstream consumers see the
        # post-fix verdict, not the pre-fix one.
        new_row["bic_delta"] = replay.get("bic_delta")
        new_row["masking_test_passed"] = replay.get("stage1_passed")
        new_row["masking_test_improvement"] = replay.get("stage1_improvement", 0.0) if "stage1_improvement" in replay else new_row.get("masking_test_improvement")
        new_row["masking_test_direction"] = "mae_delta"  # replay always uses MAE-delta gate
        new_row["scoring_version"] = "two_stage_v2"

        # Null out dangling parent refs.
        for pkey in ("parent_node_1", "parent_node_2"):
            pid = new_row.get(pkey)
            if pid and pid not in survivor_ids:
                new_row[pkey] = None
                dangling_nulled += 1

        kept.append(new_row)

    return kept, {"kept": len(kept), "dangling_parent_nulled": dangling_nulled}


def filter_crossbreed_index(
    rows: list[dict[str, Any]],
    survivor_ids: set[str],
) -> list[dict[str, Any]]:
    return [
        r for r in rows
        if r.get("node_1") in survivor_ids and r.get("node_2") in survivor_ids
    ]


def filter_crossbreed_queue(
    rows: list[dict[str, Any]],
    survivor_ids: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        parents = r.get("parents") or []
        if all(p in survivor_ids for p in parents):
            out.append(r)
    return out


def recompute_admitted_index_fingerprints(
    surviving_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute the dedup ledger so only survivor fingerprints remain.

    For each surviving row, we recompute its fingerprint from
    ``parent_node_1`` + ``parent_node_2`` + ``hypothesis`` — matching
    ``feature_hypothesis_kazakhstan._admit_with_dedup``. Order matters and
    only non-null parents are included (mirroring the task code).
    """
    fps: list[str] = []
    for row in surviving_rows:
        parents = [p for p in (row.get("parent_node_1"), row.get("parent_node_2")) if p]
        fp = _fingerprint(parents, row.get("hypothesis", ""))
        fps.append(fp)
    return {"fingerprints": fps}


def filter_training_pickle(
    pickle_path: Path,
    survivor_episode_ids: set[str],
    all_kg_episode_ids: set[str],
    dry_run: bool,
) -> dict[str, Any]:
    """Drop pickle rows whose ``episode_id`` is in the KG ledger but is *not*
    a survivor (i.e. the row came from an admit that has now been retroactively
    rejected). Rows whose episode_id isn't in the KG at all are kept — those
    are workflow-successful-but-non-admitting episodes whose training pairs
    don't depend on the admit verdict.
    """
    import pickle  # deferred so --help doesn't require pickle env

    with pickle_path.open("rb") as fh:
        data = pickle.load(fh)
    before = len(data)
    kept = []
    dropped_eids: set[str] = set()
    for row in data:
        eid = row.get("episode_id")
        if eid in all_kg_episode_ids and eid not in survivor_episode_ids:
            dropped_eids.add(eid)
            continue
        kept.append(row)
    if not dry_run:
        tmp = pickle_path.with_suffix(pickle_path.suffix + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(kept, fh)
        tmp.replace(pickle_path)
    return {
        "before": before,
        "after": len(kept),
        "dropped": before - len(kept),
        "dropped_episode_ids": sorted(dropped_eids),
    }


def rescale_checkpoint(
    checkpoint_path: Path,
    kept_pickle_rows: int,
    before_pickle_rows: int,
    dropped_kg_admits: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Scale success counters by ``kept / before`` and reduce
    ``total_successful`` by the number of dropped admits.

    This is a heuristic — the exact mapping between pickle rows and the
    transform-derived ``training_row_count`` is many-to-many. Scaling
    proportionally is the closest available approximation without re-running
    the export pipeline.
    """
    if before_pickle_rows == 0:
        return {"scaled": False, "reason": "pickle empty"}
    ratio = kept_pickle_rows / before_pickle_rows
    cp = json.loads(checkpoint_path.read_text())
    before = {
        "training_row_count": cp.get("training_row_count"),
        "raw_successful_row_count": cp.get("raw_successful_row_count"),
        "total_successful": cp.get("total_successful"),
        "total_score": cp.get("total_score"),
        "success_rate": cp.get("success_rate"),
    }
    cp["training_row_count"] = int(round((cp.get("training_row_count") or 0) * ratio))
    cp["raw_successful_row_count"] = int(round((cp.get("raw_successful_row_count") or 0) * ratio))
    cp["total_score"] = (cp.get("total_score") or 0.0) * ratio
    new_total_successful = max(0, (cp.get("total_successful") or 0) - dropped_kg_admits)
    cp["total_successful"] = new_total_successful
    total_run = cp.get("total_episodes_run") or 0
    cp["success_rate"] = (new_total_successful / total_run) if total_run else 0.0
    cp.setdefault("_retroactive_filter_meta", {})
    cp["_retroactive_filter_meta"] = {
        "applied_at": __import__("datetime").datetime.now().isoformat(),
        "pickle_keep_ratio": ratio,
        "dropped_kg_admits": dropped_kg_admits,
        "before": before,
    }
    if not dry_run:
        _atomic_write_json(checkpoint_path, cp)
    return {"scaled": True, "ratio": ratio, "before": before, "after": {
        "training_row_count": cp["training_row_count"],
        "raw_successful_row_count": cp["raw_successful_row_count"],
        "total_successful": cp["total_successful"],
        "total_score": cp["total_score"],
        "success_rate": cp["success_rate"],
    }}


def prune_store_index_and_npys(
    admitted_dir: Path,
    survivor_layer_names: set[str],
    dry_run: bool,
) -> dict[str, Any]:
    """Drop layers from store admitted/index.json not in ``survivor_layer_names``
    and delete the corresponding .npy files. Returns a stats dict.
    """
    stats: dict[str, Any] = {"layers_removed": [], "npys_removed": []}
    idx_path = admitted_dir / "index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        layers = idx.get("layers", {})
        removed = [name for name in layers if name not in survivor_layer_names]
        for name in removed:
            del layers[name]
        idx["layers"] = layers
        stats["layers_removed"] = removed
        if not dry_run:
            _atomic_write_json(idx_path, idx)

    layers_dir = admitted_dir / "layers"
    if layers_dir.exists():
        for npy in layers_dir.glob("*.npy"):
            if npy.stem not in survivor_layer_names:
                stats["npys_removed"].append(npy.name)
                if not dry_run:
                    npy.unlink()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--replay-report", required=True, type=Path)
    parser.add_argument("--kg-dir", required=True, type=Path)
    parser.add_argument("--store-admitted-dir", required=True, type=Path,
                        help="e.g. .../store/teniz_basin/admitted")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without mutating any file.")
    parser.add_argument("--training-pickle", type=Path, default=None,
                        help="Optional: also filter this training_pairs.pkl.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Optional: also rescale this checkpoint.json.")
    parser.add_argument("--backup-experiments", type=Path, default=None,
                        help="Pre-filter experiments.jsonl path (used to derive "
                             "the all-KG-episode-id set for pickle filtering).")
    args = parser.parse_args()

    survivors = _load_replay(args.replay_report)
    survivor_ids = set(survivors)
    survivor_layer_names = {s.layer_name for s in survivors.values()}

    kg = args.kg_dir
    experiments = _load_jsonl(kg / "experiments.jsonl")
    crossbreed_index = _load_jsonl(kg / "crossbreed_index.jsonl")
    crossbreed_queue = _load_jsonl(kg / "crossbreed_queue.jsonl")

    filtered_exps, exp_stats = filter_experiments(experiments, survivors)
    filtered_cb_idx = filter_crossbreed_index(crossbreed_index, survivor_ids)
    filtered_cb_q = filter_crossbreed_queue(crossbreed_queue, survivor_ids)
    new_admitted_index = recompute_admitted_index_fingerprints(filtered_exps)

    summary: dict[str, Any] = {
        "survivors": {
            "count": len(survivors),
            "layer_names": sorted(survivor_layer_names),
        },
        "experiments_jsonl": {
            "before": len(experiments),
            "after": len(filtered_exps),
            "dropped": len(experiments) - len(filtered_exps),
            "dangling_parent_refs_nulled": exp_stats["dangling_parent_nulled"],
        },
        "admitted_index_json": {
            "before_fingerprints": None,  # filled below
            "after_fingerprints": len(new_admitted_index["fingerprints"]),
        },
        "crossbreed_index_jsonl": {
            "before": len(crossbreed_index),
            "after": len(filtered_cb_idx),
            "dropped": len(crossbreed_index) - len(filtered_cb_idx),
        },
        "crossbreed_queue_jsonl": {
            "before": len(crossbreed_queue),
            "after": len(filtered_cb_q),
            "dropped": len(crossbreed_queue) - len(filtered_cb_q),
        },
    }

    admitted_index_path = kg / "admitted_index.json"
    if admitted_index_path.exists():
        summary["admitted_index_json"]["before_fingerprints"] = len(
            json.loads(admitted_index_path.read_text()).get("fingerprints", [])
        )

    store_stats = prune_store_index_and_npys(
        args.store_admitted_dir, survivor_layer_names, dry_run=args.dry_run,
    )
    summary["store_admitted"] = {
        "layers_removed_from_index": store_stats["layers_removed"],
        "layers_removed_count": len(store_stats["layers_removed"]),
        "npys_deleted": store_stats["npys_removed"],
        "npys_deleted_count": len(store_stats["npys_removed"]),
    }

    if not args.dry_run:
        _write_jsonl(kg / "experiments.jsonl", filtered_exps)
        _write_jsonl(kg / "crossbreed_index.jsonl", filtered_cb_idx)
        _write_jsonl(kg / "crossbreed_queue.jsonl", filtered_cb_q)
        _atomic_write_json(kg / "admitted_index.json", new_admitted_index)

    # Optional add-ons.
    if args.training_pickle is not None:
        survivor_eids = {
            (nid[len("exp_"):] if nid.startswith("exp_") else nid)
            for nid in survivor_ids
        }
        # Build the full KG episode-id set (all admits, pre-filter).
        backup_path = args.backup_experiments
        if backup_path is None:
            # Pick the freshest pre-filter backup automatically.
            candidates = sorted(
                args.kg_dir.parent.parent.glob("_pre_retrofilter_backup_*/knowledge/*/experiments.jsonl")
            )
            backup_path = candidates[-1] if candidates else None
        all_kg_eids: set[str] = set()
        if backup_path and backup_path.exists():
            for row in _load_jsonl(backup_path):
                nid = row.get("node_id", "")
                all_kg_eids.add(nid[len("exp_"):] if nid.startswith("exp_") else nid)
        else:
            print("warn: no pre-filter experiments.jsonl backup found; "
                  "pickle filter will drop only currently-surviving-rows' opposites",
                  flush=True)
            all_kg_eids = survivor_eids  # degenerate: keep everything not in survivor set
        pickle_stats = filter_training_pickle(
            args.training_pickle, survivor_eids, all_kg_eids, dry_run=args.dry_run,
        )
        summary["training_pickle"] = pickle_stats

        if args.checkpoint is not None:
            # Source-of-truth for dropped count is the replay report (independent
            # of whether experiments.jsonl has already been filtered).
            replay_rows = []
            with args.replay_report.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        replay_rows.append(json.loads(line))
            dropped_kg_admits = sum(
                1 for r in replay_rows
                if not (r.get("status") == "ok" and r.get("replay", {}).get("admitted"))
            )
            cp_stats = rescale_checkpoint(
                args.checkpoint,
                kept_pickle_rows=pickle_stats["after"],
                before_pickle_rows=pickle_stats["before"],
                dropped_kg_admits=dropped_kg_admits,
                dry_run=args.dry_run,
            )
            summary["checkpoint"] = cp_stats

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
