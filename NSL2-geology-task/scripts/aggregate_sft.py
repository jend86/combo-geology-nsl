"""Aggregate CURRENT-format SFT rows across multiple runs into one pooled dataset.

Re-applies the task's current ``training_data_transforms()`` to each run's
``all_episodes.jsonl`` (best-effort evidence backfill: on-disk dataset chunk →
captured ``source_excerpt`` [2026-05-31+ runs only] → transcript SAMPLE block),
tags every row with its ``source_run_id``, concatenates with cross-run exact
prompt/response dedup, and writes a pooled ``sft_training_rows.jsonl`` plus a
``manifest.json``.

Non-mutating: builds in-memory and writes only under the chosen ``--out`` dir.
It NEVER publishes into a run's ``exports/sft/`` — safe to run against the live
generation (read-only on ``all_episodes.jsonl``).

The pooled file is consumable directly by ``src.train.qlora`` via
``--training-data <pooled>`` (that loader concatenates files, skips rows where
``success`` is falsy, and reads only ``prompt`` + ``raw_response``).

Usage:
    LD_LIBRARY_PATH="$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH" \
      .venv/bin/python scripts/aggregate_sft.py [--out <dir>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

TASK_CLASS = "tasks.feature_hypothesis_kazakhstan.FeatureHypothesisKazakhstanTask"
BASE = Path("data/kazakhstan/feature-hypothesis")

# (run_id, generation_dir, note). Order = oldest → newest.
RUNS: list[tuple[str, Path, str]] = [
    ("20260524-rg26xw", BASE / "generations/20260524-rg26xw/generation_0", "old workflow (survey+hypothesise); no data_spec.files / source_excerpt"),
    ("20260529-archive-gen0", BASE / "_archive_2026-05-31/generation_0", "05-29 archived run; old workflow"),
    ("20260529-h7x7ix", BASE / "generations/20260529-h7x7ix/generation_0", "05-29 short run; old workflow"),
    ("20260529-r2ligp", BASE / "generations/20260529-r2ligp/generation_0", "05-29→31 main run; old workflow"),
    ("20260531-f2jcpm", BASE / "generations/20260531-f2jcpm/generation_0", "05-31 reshape run; native source_excerpt"),
    ("20260531-uz1hbx-aborted", BASE / "_archive_2026-05-31/generation_0_run1_aborted", "05-31 aborted; native source_excerpt"),
    ("LIVE-generation_0", BASE / "generations/generation_0", "ACTIVE run snapshot (read-only); native source_excerpt"),
]


def _load_episodes(gen_dir: Path):
    """Robustly load episodes; skip unparseable lines (e.g. a partial trailing
    line on the live run). Returns (GenerationData|None, n_lines, n_skipped)."""
    from src.typing.trajectory import EpisodeTrajectory, GenerationData

    path = gen_dir / "all_episodes.jsonl"
    gd: GenerationData | None = None
    n = skipped = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                ep = EpisodeTrajectory.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                skipped += 1
                continue
            if gd is None:
                gd = GenerationData(generation_id=ep.generation_id)
            gd.add_episode(ep)
    return gd, n, skipped


def _build_rows(task, gen_dir: Path, max_per_family: int) -> tuple[list[dict], int, int, str | None]:
    from src.training_data.transforms import (
        build_export_recipe,
        build_training_export,
        TrainingDataExportContext,
    )
    from tasks.feature_hypothesis_kazakhstan import ExperimentReasoningRows

    gd, n_lines, skipped = _load_episodes(gen_dir)
    if gd is None:
        return [], n_lines, skipped, None
    meta_path = gen_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    # Mirror task.training_data_transforms() but override only the per-family
    # diversity cap (default 5). All other params keep the canonical defaults;
    # exact prompt/response dedup in _curate still applies.
    transforms = (
        ExperimentReasoningRows(
            max_per_family=max_per_family,
            dataset_dir=str(getattr(task, "_dataset_dir", "") or ""),
        ),
    )
    recipe = build_export_recipe(transforms)
    ctx = TrainingDataExportContext(
        generation_id=gd.generation_id,
        run_id=metadata.get("run_id"),
        task_name=str(getattr(task, "name", type(task).__name__)),
        source_generation_dir=gen_dir,
        source_all_episodes_path=gen_dir / "all_episodes.jsonl",
        export_recipe_hash=recipe.recipe_hash,
    )
    export = build_training_export(gd, transforms, ctx)
    return export.rows, n_lines, skipped, recipe.recipe_hash


def _pair_hash(row: dict) -> str:
    p = re.sub(r"\s+", " ", str(row.get("prompt", ""))).strip().lower()
    r = re.sub(r"\s+", " ", str(row.get("raw_response", ""))).strip().lower()
    return hashlib.sha256(f"{p}\n---\n{r}".encode("utf-8")).hexdigest()


def _has_evidence(row: dict) -> bool:
    return "Source examined" in (row.get("prompt") or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE / "aggregated_sft" / "20260601-pooled"))
    ap.add_argument(
        "--max-per-family",
        type=int,
        default=10**9,
        help="per-family diversity cap in _curate (canonical default 5; 10**9 ≈ lift fully). "
        "Exact prompt/response dedup always applies.",
    )
    args = ap.parse_args()

    from src.task.loader import load_task

    task = load_task(TASK_CLASS)
    current_hash = None

    pooled: list[dict] = []
    seen: set[str] = set()
    per_run: list[dict] = []
    dropped_dupes = 0

    for run_id, gen_dir, note in RUNS:
        if not (gen_dir / "all_episodes.jsonl").is_file():
            per_run.append({"run_id": run_id, "status": "MISSING", "note": note})
            continue
        rows, n_lines, skipped, rhash = _build_rows(task, gen_dir, args.max_per_family)
        current_hash = current_hash or rhash
        kinds = Counter((r.get("record_meta") or {}).get("task_kind", "?") for r in rows)
        n_success = sum(1 for r in rows if r.get("success"))
        n_evid = sum(1 for r in rows if _has_evidence(r))
        # evidence-bearing kinds only
        ev_kinds = {"dataset_hypothesis", "analysis_plan"}
        n_evid_eligible = sum(
            1 for r in rows if (r.get("record_meta") or {}).get("task_kind") in ev_kinds
        )
        kept = 0
        for r in rows:
            h = _pair_hash(r)
            if h in seen:
                dropped_dupes += 1
                continue
            seen.add(h)
            r["source_run_id"] = run_id  # safe: trainer ignores unknown fields
            pooled.append(r)
            kept += 1
        per_run.append({
            "run_id": run_id,
            "generation_dir": str(gen_dir),
            "note": note,
            "episode_lines": n_lines,
            "episode_lines_skipped": skipped,
            "rows_built": len(rows),
            "rows_kept_after_dedup": kept,
            "rows_success_true": n_success,
            "rows_with_evidence_block": n_evid,
            "evidence_eligible_rows": n_evid_eligible,
            "task_kind_dist": dict(kinds),
            "recipe_hash": rhash,
        })

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "sft_training_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as fh:
        for r in pooled:
            fh.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")

    total_success = sum(1 for r in pooled if r.get("success"))
    total_evid = sum(1 for r in pooled if _has_evidence(r))
    manifest = {
        "schema_version": 1,
        "created_for": "pooled SFT finetune across recent Kazakhstan runs",
        "max_per_family": args.max_per_family,
        "recipe_hash": current_hash,
        "recipe_note": (
            "max_per_family overridden from canonical 5; recipe_hash therefore "
            "differs from the run-time ee073501 (row SCHEMA is identical, only "
            "the diversity cap differs). Exact prompt/response dedup still applied."
        ),
        "pooled_rows": len(pooled),
        "pooled_rows_success_true": total_success,
        "pooled_rows_with_evidence_block": total_evid,
        "cross_run_exact_dupes_dropped": dropped_dupes,
        "trainer_note": "src.train.qlora skips rows where success is falsy; uses prompt+raw_response only",
        "runs": per_run,
        "sft_training_rows_path": rows_path.name,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )

    # ---- self-checks / summary ----
    assert all(isinstance(r.get("prompt"), str) and isinstance(r.get("raw_response"), str) for r in pooled), \
        "pooled rows must all carry str prompt + raw_response"
    print(f"\n=== pooled → {rows_path} ===")
    print(f"  total rows ................. {len(pooled)}")
    print(f"  success=True (trainable) ... {total_success}")
    print(f"  with 'Source examined' ..... {total_evid}")
    print(f"  cross-run dupes dropped .... {dropped_dupes}")
    print(f"  current recipe hash ........ {current_hash}")
    print(f"\n  {'run':26} {'lines':>6} {'skip':>4} {'built':>6} {'kept':>5} {'succ':>5} {'evid':>5}")
    for pr in per_run:
        if pr.get("status") == "MISSING":
            print(f"  {pr['run_id']:26} MISSING")
            continue
        print(f"  {pr['run_id']:26} {pr['episode_lines']:6} {pr['episode_lines_skipped']:4} "
              f"{pr['rows_built']:6} {pr['rows_kept_after_dedup']:5} {pr['rows_success_true']:5} "
              f"{pr['rows_with_evidence_block']:5}")
    print(f"\n  manifest → {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
