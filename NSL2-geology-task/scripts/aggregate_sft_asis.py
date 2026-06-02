"""Pool SFT rows ACROSS runs in their NATURAL (as-exported) form — no re-transform.

Unlike ``aggregate_sft.py`` (which re-runs the task's current
``training_data_transforms()`` over each run's ``all_episodes.jsonl`` and thereby
homogenises everything to the current reshape schema), this script keeps every row
in the exact form it was emitted — we only fill in the per-turn completion the old
exporter dropped, and tag provenance.

EMPTY-RESPONSE FIX — RECOVER, DON'T DROP
----------------------------------------
The raw-turn rows (``interaction_type=external::external::step_N``) were emitted with
``raw_response=""`` — the old exporter never wrote the completion. It is NOT lost:
each turn's prompt is a role-tagged transcript (``[system]/[user]/[assistant]/[tool]``)
and the NEXT same-phase prompt is that transcript grown by exactly that turn's
``[assistant]`` block + tool result. So a turn's completion is the ``[assistant]``
block newly present in the following same-phase prompt. We backfill it verbatim
(prefix-role-match guarded so a tool result can never leak in). The only unrecoverable
row is each phase's FINAL turn (the next phase resets the transcript) — dropped.

SOURCES (one logical export per run, but a run may draw from two row-streams):
  20260524-rg26xw    export 89eeb998   recover per-turn completions
  20260529-r2ligp    export 89eeb998   recover per-turn completions
  20260531-f2jcpm    export ee073501   58 decomposed rows kept as-is
                   + all_episodes      per-turn completions recovered from the raw
                                       turns of its SUCCESS episodes (the ee073501
                                       export collapsed them away; recovery pads it)
  20260531-uz1hbx-aborted  export ee073501   3 decomposed rows kept as-is

``all_episodes`` recovery is success-FILTERED (expert iteration trains on successful
trajectories; the on-disk exports were already success-only). It reads the same raw
turns + applies the same next-prompt diff — NOT the reshape transform.

A row is kept iff ``success`` truthy AND ``prompt`` is a ``str`` AND ``raw_response``
(recovered or native) is a non-blank ``str`` — the trainer's contract plus the
blank-response guard it lacks. Cross-run/-source exact prompt+response dedup.

Out of scope: 20260529-h7x7ix, 20260529-archive-gen0, the LIVE generation_0 — no
on-disk export and (h7x7ix/archive) old runs we did not expand.

Non-mutating: reads read-only, writes only under ``--out``. Pure stdlib.

Usage:
    python3 scripts/aggregate_sft_asis.py [--out <dir>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

BASE = Path("data/kazakhstan/feature-hypothesis")

GEN = BASE / "generations"
ARC = BASE / "_archive_2026-05-31"

# Each run lists its row sources. kind="export" reads an export jsonl (rows already
# grouped one file); kind="all_episodes" reads all_episodes.jsonl, success-filters,
# and recovers per-turn rows from each episode's raw_training_rows. Both run through
# the same _recover_episode (curated rows kept as-is, empty per-turn rows recovered).
RUNS: list[dict] = [
    {
        "run_id": "20260524-rg26xw", "family": "89eeb998",
        "sources": [{"kind": "export",
                     "path": GEN / "20260524-rg26xw/generation_0/exports/sft/20260529T090213089635Z/sft_training_rows.jsonl"}],
        "note": "old raw-turn export; per-turn completions recovered",
    },
    {
        "run_id": "20260529-r2ligp", "family": "89eeb998",
        "sources": [{"kind": "export",
                     "path": GEN / "20260529-r2ligp/generation_0/exports/sft/20260530T201341384028Z/sft_training_rows.jsonl"}],
        "note": "final native raw-turn snapshot (d96e1970 reshape export skipped)",
    },
    {
        "run_id": "20260531-f2jcpm", "family": "ee073501",
        "sources": [
            {"kind": "export",
             "path": GEN / "20260531-f2jcpm/generation_0/exports/sft/20260531T123121404344Z/sft_training_rows.jsonl"},
            {"kind": "all_episodes",
             "path": GEN / "20260531-f2jcpm/generation_0/all_episodes.jsonl"},
        ],
        "note": "ee073501 export (58 decomposed) + per-turn recovery from success-episode raw turns",
    },
    {
        "run_id": "20260531-uz1hbx-aborted", "family": "ee073501",
        "sources": [{"kind": "export",
                     "path": ARC / "generation_0_run1_aborted/exports/sft/20260531T143432409877Z/sft_training_rows.jsonl"}],
        "note": "aborted run; native ee073501 kept as-is",
    },
]

_SPLIT = re.compile(r"(?m)^\[(\w+)\]\s*$")


def _blocks(prompt: str) -> list[tuple[str, str]]:
    parts = _SPLIT.split(prompt or "")
    it = iter(parts[1:])
    return [(role, body.strip()) for role, body in zip(it, it)]


def _step_num(row: dict) -> int | None:
    m = re.search(r"step_(\d+)", row.get("interaction_type", "") or "")
    return int(m.group(1)) if m else None


def _recover_episode(rows: list[dict]) -> tuple[list[dict], int]:
    """Backfill raw_response for empty per-turn rows of one episode (export order).

    Returns (kept_rows, n_phase_final_dropped). Curated rows (nonblank raw_response)
    pass through; empty per-turn rows get their completion recovered from the next
    same-phase prompt; each phase's final turn is dropped (never captured).
    """
    out = [r for r in rows if (r.get("raw_response") or "").strip()]
    dropped_final = 0

    turns = [r for r in rows if _step_num(r) is not None and not (r.get("raw_response") or "").strip()]
    phases: list[list[dict]] = []
    for r in turns:
        if phases and r.get("workflow_step") == phases[-1][-1].get("workflow_step"):
            phases[-1].append(r)
        else:
            phases.append([r])

    for ph in phases:
        for j in range(len(ph)):
            if j == len(ph) - 1:
                dropped_final += 1
                continue
            bi = _blocks(ph[j].get("prompt") or "")
            bn = _blocks(ph[j + 1].get("prompt") or "")
            if len(bn) <= len(bi) or [r for r, _ in bn[: len(bi)]] != [r for r, _ in bi]:
                dropped_final += 1
                continue
            added = bn[len(bi):]
            if not added or added[0][0] != "assistant" or not added[0][1].strip():
                dropped_final += 1
                continue
            row = dict(ph[j])
            row["raw_response"] = added[0][1]
            out.append(row)
    return out, dropped_final


def _rows_from_source(src: dict) -> tuple[list[dict], dict]:
    """Yield recovered rows + a stats dict for one source."""
    path: Path = src["path"]
    if not path.is_file():
        return [], {"status": "MISSING", "path": str(path)}

    recovered: list[dict] = []
    dropped_final = 0

    if src["kind"] == "export":
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        n_in = len(rows)
        byep: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            byep[str(r.get("row_id", "")).rsplit(":", 1)[0]].append(r)
        for ep_rows in byep.values():
            kept, df = _recover_episode(ep_rows)
            recovered.extend(kept)
            dropped_final += df
        stats = {"kind": "export", "path": str(path), "rows_in": n_in, "phase_final_dropped": dropped_final}
    else:  # all_episodes
        eps = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        succ = [e for e in eps if e.get("success")]
        for e in succ:
            kept, df = _recover_episode(e.get("raw_training_rows") or [])
            # Keep ONLY the recovered per-turn tool-call rows. The curated
            # rewrite_output narratives that also live in raw_training_rows are
            # redundant with this run's decomposed export, so we drop them here to
            # avoid double-representing the same narrative in two formats.
            recovered.extend(r for r in kept if _step_num(r) is not None)
            dropped_final += df
        stats = {"kind": "all_episodes", "path": str(path),
                 "episodes": len(eps), "success_episodes": len(succ),
                 "phase_final_dropped": dropped_final}
    return recovered, stats


def _pair_hash(row: dict) -> str:
    p = re.sub(r"\s+", " ", str(row.get("prompt", ""))).strip().lower()
    r = re.sub(r"\s+", " ", str(row.get("raw_response", ""))).strip().lower()
    return hashlib.sha256(f"{p}\n---\n{r}".encode("utf-8")).hexdigest()


def _trainable(row: dict) -> bool:
    if not row.get("success"):
        return False
    prompt, resp = row.get("prompt"), row.get("raw_response")
    return isinstance(prompt, str) and isinstance(resp, str) and bool(resp.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE / "aggregated_sft" / "20260601-pooled-natural"))
    args = ap.parse_args()

    pooled: list[dict] = []
    seen: set[str] = set()
    per_run: list[dict] = []
    dropped_dupes = 0
    long_rows = 0  # prompts > 50k chars (trainer may truncate/skip)

    for run in RUNS:
        run_id, family = run["run_id"], run["family"]
        run_kept = 0
        src_stats = []
        for src in run["sources"]:
            recovered, stats = _rows_from_source(src)
            kept = 0
            for r in recovered:
                if not _trainable(r):
                    continue
                h = _pair_hash(r)
                if h in seen:
                    dropped_dupes += 1
                    continue
                seen.add(h)
                r["source_run_id"] = run_id           # trainer ignores unknown fields
                r["source_kind"] = src["kind"]
                if len(r["prompt"]) > 50_000:
                    long_rows += 1
                pooled.append(r)
                kept += 1
            stats["rows_kept"] = kept
            src_stats.append(stats)
            run_kept += kept
        per_run.append({"run_id": run_id, "recipe_family": family, "note": run["note"],
                        "rows_kept": run_kept, "sources": src_stats})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "sft_training_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as fh:
        for r in pooled:
            fh.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": 3,
        "created_for": "pooled SFT finetune — NATURAL (as-exported) form, per-turn completions recovered",
        "policy": (
            "per-turn raw_response recovered from the assistant block in the next "
            "same-phase prompt (verbatim prompt, no transform); curated/decomposed rows "
            "kept as-is; f2jcpm additionally recovers per-turn rows from its SUCCESS "
            "episodes' raw turns in all_episodes (its ee073501 export had collapsed them); "
            "cross-run/-source exact dedup"
        ),
        "pooled_rows": len(pooled),
        "cross_run_exact_dupes_dropped": dropped_dupes,
        "long_prompt_rows_gt_50k_chars": long_rows,
        "caveats": [
            "89eeb998/all_episodes per-turn completions are mostly tool-call JSON (the model's actual turn output)",
            "successful-episode turns kept verbatim, incl. turns whose tool call then failed (no curation)",
            "each phase's final turn is dropped (completion never captured)",
            "f2jcpm: only 24/424 episodes succeeded; failed episodes excluded",
            "a few f2jcpm rows have very long prompts (data-heavy contexts); trainer may truncate/skip",
        ],
        "out_of_scope_runs": ["20260529-h7x7ix", "20260529-archive-gen0", "LIVE-generation_0"],
        "trainer_note": "src.train.qlora skips success-falsy; reads prompt+raw_response; all pooled rows satisfy this",
        "runs": per_run,
        "sft_training_rows_path": rows_path.name,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

    assert all(_trainable(r) for r in pooled), "every pooled row must be trainable"
    assert len({_pair_hash(r) for r in pooled}) == len(pooled), "pooled rows must be exact-unique"

    print(f"\n=== pooled (natural form, completions recovered) -> {rows_path} ===")
    print(f"  total trainable rows ....... {len(pooled)}")
    print(f"  cross-run dupes dropped .... {dropped_dupes}")
    print(f"  long prompts (>50k chars) .. {long_rows}")
    print(f"\n  {'run':26}{'family':10}{'kept':>6}   sources")
    for pr in per_run:
        srcs = ", ".join(
            f"{s['kind']}={s.get('rows_kept', 0)}"
            + (f"/{s['success_episodes']}succ-eps" if s.get("kind") == "all_episodes" else "")
            for s in pr["sources"]
        )
        print(f"  {pr['run_id']:26}{pr['recipe_family']:10}{pr['rows_kept']:6}   {srcs}")
    print(f"\n  manifest -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
