"""Preview SFT training rows — read an existing export OR build them in-memory.

Two modes:

  export  <path-to-sft_training_rows.jsonl>
          Read rows already written to disk (authentic shape at the time the
          export was produced — does NOT re-run the current transform).

  build   <generation_dir>
          Reconstruct GenerationData from all_episodes.jsonl, apply the task's
          CURRENT training_data_transforms(), and preview the resulting rows
          WITHOUT publishing anything to disk (no export dir, no latest.json).
          Safe to run against an in-progress generation.

Prints a per-shape summary (rows, rows/episode, task-kind / workflow-step
distribution, length stats) and a few truncated sample rows.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

TASK_CLASS = "tasks.feature_hypothesis_kazakhstan.FeatureHypothesisKazakhstanTask"
TRUNC = 700


def _truncate(text: str, limit: int = TRUNC) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n        … [+{len(text) - limit} chars truncated]"


def _kind(row: dict) -> str:
    meta = row.get("record_meta") or {}
    return (
        meta.get("task_kind")
        or meta.get("task")
        or row.get("workflow_step")
        or "?"
    )


def _load_export_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_rows(generation_dir: Path) -> tuple[list[dict], dict]:
    from src.task.loader import load_task
    from src.training_data.transforms import (
        build_training_export,
        build_export_recipe,
        TrainingDataExportContext,
    )
    from src.typing.trajectory import EpisodeTrajectory, GenerationData

    task = load_task(TASK_CLASS)
    all_episodes_path = generation_dir / "all_episodes.jsonl"

    gen_data: GenerationData | None = None
    with all_episodes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            episode = EpisodeTrajectory.from_dict(json.loads(line))
            if gen_data is None:
                gen_data = GenerationData(generation_id=episode.generation_id)
            gen_data.add_episode(episode)
    if gen_data is None:
        raise SystemExit(f"No episodes found in {all_episodes_path}")

    meta_path = generation_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    transforms = tuple(task.training_data_transforms())
    recipe = build_export_recipe(transforms)
    context = TrainingDataExportContext(
        generation_id=gen_data.generation_id,
        run_id=metadata.get("run_id"),
        task_name=str(getattr(task, "name", type(task).__name__)),
        source_generation_dir=generation_dir,
        source_all_episodes_path=all_episodes_path,
        export_recipe_hash=recipe.recipe_hash,
    )
    export = build_training_export(gen_data, transforms, context)
    return export.rows, export.report


def _summarize(rows: list[dict], report: dict | None, n_samples: int) -> None:
    episodes = {r.get("episode_id") for r in rows}
    n_ep = len([e for e in episodes if e])
    kinds = Counter(_kind(r) for r in rows)
    prompt_chars = [len(r.get("prompt") or "") for r in rows]
    resp_chars = [len(r.get("raw_response") or "") for r in rows]
    faith = Counter(
        (r.get("record_meta") or {}).get("faithfulness", "—") for r in rows
    )

    print(f"  rows total ............. {len(rows)}")
    print(f"  distinct episodes ...... {n_ep}")
    if n_ep:
        print(f"  rows / episode ......... {len(rows) / n_ep:.2f}")
    if report:
        print(f"  raw_successful_rows .... {report.get('raw_successful_row_count')}")
        print(f"  training_row_count ..... {report.get('training_row_count')}")
    print(f"  task-kind / step dist .. {dict(kinds)}")
    if any(k != '—' for k in faith):
        print(f"  faithfulness tags ...... {dict(faith)}")
    if prompt_chars:
        print(
            f"  prompt chars ........... "
            f"median={int(median(prompt_chars))} max={max(prompt_chars)}"
        )
    if resp_chars:
        print(
            f"  response chars ......... "
            f"median={int(median(resp_chars))} max={max(resp_chars)}"
        )
    # Per-kind response-length medians — surfaces thin/degraded targets.
    per_kind: dict[str, list[int]] = {}
    for r in rows:
        per_kind.setdefault(_kind(r), []).append(len(r.get("raw_response") or ""))
    print("  per-kind response chars (median / min / max):")
    for k in sorted(per_kind):
        v = per_kind[k]
        print(f"      {k:<20} n={len(v):<4} median={int(median(v)):<5} min={min(v):<5} max={max(v)}")

    print()
    print(f"  --- {min(n_samples, len(rows))} sample rows ---")
    # Spread samples across distinct task kinds where possible.
    by_kind: dict[str, list[dict]] = {}
    for r in rows:
        by_kind.setdefault(_kind(r), []).append(r)
    picked: list[dict] = []
    for kind_rows in by_kind.values():
        picked.append(kind_rows[0])
    for r in rows:
        if len(picked) >= n_samples:
            break
        if r not in picked:
            picked.append(r)
    for i, r in enumerate(picked[:n_samples]):
        meta = r.get("record_meta") or {}
        print()
        print(f"  [sample {i}] kind={_kind(r)!r} workflow_step={r.get('workflow_step')!r}")
        prov = {
            k: meta.get(k)
            for k in ("task_kind", "faithfulness", "novelty", "provenance", "outcome_appended")
            if k in meta
        }
        if prov:
            print(f"     record_meta: {prov}")
        print(f"     PROMPT ({len(r.get('prompt') or '')} ch):")
        print("       " + _truncate(r.get("prompt") or "").replace("\n", "\n       "))
        print(f"     RAW_RESPONSE ({len(r.get('raw_response') or '')} ch):")
        print("       " + _truncate(r.get("raw_response") or "").replace("\n", "\n       "))


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage:\n"
            "  preview_sft_rows.py export <sft_training_rows.jsonl> [n_samples]\n"
            "  preview_sft_rows.py build  <generation_dir>          [n_samples]"
        )
    mode, target = sys.argv[1], Path(sys.argv[2])
    n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    dump_path = Path(sys.argv[4]) if len(sys.argv) > 4 else None

    print("=" * 78)
    if mode == "export":
        print(f"EXPORT (on-disk, original shape): {target}")
        print("=" * 78)
        rows = _load_export_rows(target)
        report = None
    elif mode == "build":
        print(f"BUILD (in-memory, CURRENT transform, no publish): {target}")
        print("=" * 78)
        rows, report = _build_rows(target)
    else:
        raise SystemExit(f"unknown mode: {mode!r}")

    _summarize(rows, report, n_samples)

    if dump_path is not None:
        with dump_path.open("w", encoding="utf-8") as handle:
            for r in rows:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n  [dumped {len(rows)} rows → {dump_path}]")


if __name__ == "__main__":
    main()
