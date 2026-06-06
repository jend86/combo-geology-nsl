"""Aggregate CURRENT-format SFT rows across all recoverable Kazakhstan runs.

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


def _run_id_for_generation_dir(gen_dir: Path, base: Path) -> str:
    try:
        rel = gen_dir.relative_to(base)
    except ValueError:
        rel = gen_dir
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "generations":
        if parts[1].startswith("generation_"):
            return f"LIVE-{parts[1]}"
        if len(parts) >= 3:
            return f"{parts[1]}-{parts[2]}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(rel)).strip("-") or "unknown-run"


def _discover_runs(base: Path) -> list[tuple[str, Path, str]]:
    runs: list[tuple[str, Path, str]] = []
    seen_paths: set[Path] = set()
    for all_episodes_path in sorted(base.rglob("all_episodes.jsonl")):
        if "aggregated_sft" in all_episodes_path.parts:
            continue
        gen_dir = all_episodes_path.parent
        resolved = gen_dir.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        runs.append(
            (
                _run_id_for_generation_dir(gen_dir, base),
                gen_dir,
                "auto-discovered all_episodes.jsonl; transformed with current ExperimentReasoningRows",
            )
        )
    return runs


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


def _build_rows(
    task,
    gen_dir: Path,
    max_per_family: int,
    max_coordinate_provenance_rows: int,
) -> tuple[list[dict], int, int, str | None, dict]:
    from src.training_data.transforms import (
        build_export_recipe,
        build_training_export,
        TrainingDataExportContext,
    )
    from tasks.feature_hypothesis_kazakhstan import ExperimentReasoningRows

    gd, n_lines, skipped = _load_episodes(gen_dir)
    if gd is None:
        return [], n_lines, skipped, None, {}
    meta_path = gen_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    # Mirror task.training_data_transforms() but override only the per-family
    # diversity cap (default 5). All other params keep the canonical defaults;
    # exact prompt/response dedup in _curate still applies.
    transforms = (
        ExperimentReasoningRows(
            max_per_family=max_per_family,
            max_coordinate_provenance_rows=max_coordinate_provenance_rows,
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
    return export.rows, n_lines, skipped, recipe.recipe_hash, export.report


def _pair_hash(row: dict) -> str:
    p = re.sub(r"\s+", " ", str(row.get("prompt", ""))).strip().lower()
    r = re.sub(r"\s+", " ", str(row.get("raw_response", ""))).strip().lower()
    return hashlib.sha256(f"{p}\n---\n{r}".encode("utf-8")).hexdigest()


def _has_evidence(row: dict) -> bool:
    return "Source examined" in (row.get("prompt") or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE / "aggregated_sft" / "20260606-v2-full-refresh"))
    ap.add_argument("--discover-root", default=str(BASE))
    ap.add_argument(
        "--max-per-family",
        type=int,
        default=10**9,
        help="per-family diversity cap in _curate (canonical default 5; 10**9 ≈ lift fully). "
        "Exact prompt/response dedup always applies.",
    )
    ap.add_argument(
        "--max-coordinate-provenance-rows",
        type=int,
        default=3,
        help="cap for per-episode coordinate_provenance rows; exact dedup still applies",
    )
    ap.add_argument(
        "--exclude-pair-kind",
        action="append",
        default=["code_synthesis"],
        help=(
            "pair_kind/task_kind to exclude from the pooled aggregate after rebuilding rows; "
            "repeat to exclude more. Defaults to code_synthesis so this aggregate omits "
            "code-writing targets."
        ),
    )
    args = ap.parse_args()

    from src.task.loader import load_task

    task = load_task(TASK_CLASS)
    recipe_hashes: set[str] = set()

    pooled: list[dict] = []
    seen: set[str] = set()
    per_run: list[dict] = []
    dropped_dupes = 0
    excluded_by_kind = Counter()
    excluded_pair_kinds = {str(kind) for kind in (args.exclude_pair_kind or []) if str(kind)}
    runs = _discover_runs(Path(args.discover_root))

    for run_id, gen_dir, note in runs:
        if not (gen_dir / "all_episodes.jsonl").is_file():
            per_run.append({"run_id": run_id, "status": "MISSING", "note": note})
            continue
        rows, n_lines, skipped, rhash, report = _build_rows(
            task,
            gen_dir,
            args.max_per_family,
            args.max_coordinate_provenance_rows,
        )
        if rhash:
            recipe_hashes.add(rhash)
        kinds = Counter((r.get("record_meta") or {}).get("task_kind", "?") for r in rows)
        routes = Counter((r.get("record_meta") or {}).get("artifact_route", "?") for r in rows)
        n_success = sum(1 for r in rows if r.get("success"))
        n_evid = sum(1 for r in rows if _has_evidence(r))
        # evidence-bearing kinds only
        ev_kinds = {"dataset_hypothesis", "analysis_plan"}
        n_evid_eligible = sum(
            1 for r in rows if (r.get("record_meta") or {}).get("task_kind") in ev_kinds
        )
        kept = 0
        run_excluded_by_kind = Counter()
        for r in rows:
            meta = r.get("record_meta") if isinstance(r.get("record_meta"), dict) else {}
            pair_kind = str(meta.get("pair_kind") or meta.get("task_kind") or "")
            if pair_kind in excluded_pair_kinds:
                excluded_by_kind[pair_kind] += 1
                run_excluded_by_kind[pair_kind] += 1
                continue
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
            "rows_excluded_by_pair_kind": dict(run_excluded_by_kind),
            "rows_success_true": n_success,
            "rows_with_evidence_block": n_evid,
            "evidence_eligible_rows": n_evid_eligible,
            "task_kind_dist": dict(kinds),
            "artifact_route_dist": dict(routes),
            "recipe_hash": rhash,
            "export_report": {
                key: report.get(key)
                for key in (
                    "training_row_count",
                    "rows_by_pair_kind",
                    "rows_by_artifact_route",
                    "episodes_with_value_grid",
                    "episodes_with_feature_geometry",
                    "creative_fallback_rows_method_framed",
                )
                if key in report
            },
        })

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "sft_training_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as fh:
        for r in pooled:
            fh.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")

    total_success = sum(1 for r in pooled if r.get("success"))
    total_evid = sum(1 for r in pooled if _has_evidence(r))
    pooled_kinds = Counter((r.get("record_meta") or {}).get("task_kind", "?") for r in pooled)
    pooled_routes = Counter((r.get("record_meta") or {}).get("artifact_route", "?") for r in pooled)
    manifest = {
        "schema_version": 2,
        "created_for": "pooled v2 SFT finetune across all recoverable Kazakhstan runs",
        "max_per_family": args.max_per_family,
        "max_coordinate_provenance_rows": args.max_coordinate_provenance_rows,
        "excluded_pair_kinds": sorted(excluded_pair_kinds),
        "rows_excluded_by_pair_kind": dict(excluded_by_kind),
        "discover_root": str(Path(args.discover_root)),
        "recipe_hashes": sorted(recipe_hashes),
        "recipe_note": (
            "Rows are rebuilt from all discovered all_episodes.jsonl files using the current "
            "ExperimentReasoningRows[v2] transform. Exact prompt/response dedup is applied "
            "across runs after per-run curation."
        ),
        "pooled_rows": len(pooled),
        "pooled_rows_success_true": total_success,
        "pooled_rows_with_evidence_block": total_evid,
        "pooled_rows_by_pair_kind": dict(pooled_kinds),
        "pooled_rows_by_artifact_route": dict(pooled_routes),
        "cross_run_exact_dupes_dropped": dropped_dupes,
        "trainer_note": "src.train.qlora skips rows where success is falsy; uses prompt+raw_response only",
        "discovered_run_count": len(runs),
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
    print(f"  recipe hashes .............. {', '.join(sorted(recipe_hashes)) or 'none'}")
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
