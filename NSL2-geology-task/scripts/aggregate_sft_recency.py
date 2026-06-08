"""Recency-weighted SFT aggregate (Approach C: newest-first as-is composition).

Motivation
----------
The ``v2-full-refresh`` base is a *flat union* of every recoverable run with no
recency cap, so the two longest pre-June runs (May-24 / May-29 generation data)
supply ~49% of it — data that predates basically every scoring/geometry fix in
the project. This composer instead builds the dataset **newest-first** and tilts
hard toward modern runs, down-sampling older runs that we still pull from.

Two complementary row populations are composed:

* ``synth`` — structured reasoning rows (dataset_hypothesis, analysis_plan,
  parent_*, feature_readout, coordinate_provenance, outcome_narrative,
  spatial_materialization). Reused *as already built* from the v2-full-refresh
  ``sft_training_rows.jsonl`` (each row carries ``source_run_id`` +
  ``record_meta.pair_kind``), so we do not re-run the heavy transform.
* ``asis`` — genuine agent transcript turns (code / explore / rewrite /
  translate), recovered per run from ``all_episodes.jsonl`` exactly like
  ``aggregate_sft_topup.py``.

Composition (pure ``compose()``, unit-tested in
``tests/test_aggregate_sft_recency.py``):

1. Exact prompt/response de-dup across everything, keeping the **newest** run's
   copy.
2. Split into *modern* (run completed >= ``--modern-cutoff``) and *legacy*.
3. ``modern_budget = round(modern_fraction * target)``; legacy gets the rest.
4. Fill each budget **newest-first**, per-run-capped (legacy capped tighter so a
   verbose old run can't dominate), and within a run **round-robin across
   (source, kind)** so capping never deletes a whole row family.
5. If modern under-supplies its budget the total simply shrinks — legacy is
   never grown past its budget, so the recency tilt is preserved.

Non-mutating: reads ``all_episodes.jsonl`` and the base file read-only and writes
only under ``--out``. The pooled file is consumable directly by
``src.train.qlora`` (it reads only ``prompt`` + ``raw_response`` after filtering
``success``); unknown provenance fields are ignored by the trainer.

Usage (inside the nix dev shell so the live ledger read has its libs):
    nix develop --command bash -c 'uv run python scripts/aggregate_sft_recency.py'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

BASE = Path("data/kazakhstan/feature-hypothesis")
# Fresh full-refresh base (rebuilt by aggregate_sft.py against the live run so the
# newest run's synth rows are current, not the stale 06-06 snapshot).
DEFAULT_BASE_SYNTH = BASE / "aggregated_sft/20260607-v2-full-refresh/sft_training_rows.jsonl"
DEFAULT_OUT = BASE / "aggregated_sft/20260607-recency-c"
DEFAULT_MODERN_CUTOFF = "2026-06-02"  # June-2 onward = "modern" (per-user pivot)

_SPLIT = re.compile(r"(?m)^\[(\w+)\]\s*$")


# ---------------------------------------------------------------------------
# pure composition core (no I/O — unit-tested)
# ---------------------------------------------------------------------------
def _run_order(cands: list[dict[str, Any]]) -> list[str]:
    """Run ids ordered newest-first by their candidates' recency."""
    recency: dict[str, str] = {}
    for c in cands:
        rid = c["run_id"]
        if c["recency"] > recency.get(rid, ""):
            recency[rid] = c["recency"]
    return [rid for rid, _ in sorted(recency.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]


def _take_from_run(cands: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    """Take up to ``cap`` rows from one run, round-robin across (source, kind)."""
    if cap <= 0:
        return []
    buckets: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for c in cands:
        buckets[(c["source"], c["kind"])].append(c)
    order = sorted(buckets)
    out: list[dict[str, Any]] = []
    while len(out) < cap and any(buckets[k] for k in order):
        for k in order:
            if len(out) >= cap:
                break
            if buckets[k]:
                out.append(buckets[k].pop(0))
    return out


def _select(
    pool: list[dict[str, Any]], budget: int, per_run_cap: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fill ``budget`` newest-first, each run bounded by ``per_run_cap``."""
    by_run: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in pool:
        by_run[c["run_id"]].append(c)
    selected: list[dict[str, Any]] = []
    chosen_ids: set[int] = set()
    for rid in _run_order(pool):
        if len(selected) >= budget:
            break
        cap = min(per_run_cap, budget - len(selected))
        taken = _take_from_run(by_run[rid], cap)
        for c in taken:
            chosen_ids.add(id(c))
        selected.extend(taken)
    leftover = [c for c in pool if id(c) not in chosen_ids]
    return selected, leftover


def compose(
    candidates: list[dict[str, Any]],
    *,
    target_rows: int,
    modern_fraction: float,
    modern_per_run_cap: int,
    legacy_per_run_cap: int,
) -> dict[str, Any]:
    """Compose a recency-tilted selection from tagged candidate rows.

    Each candidate is a dict with keys: run_id, recency (ISO str), is_modern
    (bool), source ('synth'|'asis'), kind (str), pair_hash (str), row (dict),
    and an optional ``pin`` (bool). **Pinned** candidates are always included in
    full and bypass the per-run cap — used to make the newer-run *synth* rows the
    guaranteed base of the dataset (they are never down-sampled).

    Budgeting:
      * ``modern_budget = round(modern_fraction * target_rows)``; pinned-modern
        rows fill it first, then unpinned modern (as-is) rows up to the budget.
      * ``legacy_budget = target - max(len(modern_selected), modern_budget)`` so
        legacy never grows past its fraction share — if modern under-supplies the
        total simply shrinks rather than diluting the recency tilt.

    Returns ``{"rows": [...selected row dicts...], "report": {...}}``.
    """
    # global exact de-dup, newest-first so the newest run's copy survives
    ordered = sorted(candidates, key=lambda c: (c["recency"], c["run_id"]), reverse=True)
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for c in ordered:
        if c["pair_hash"] in seen:
            continue
        seen.add(c["pair_hash"])
        uniq.append(c)

    pinned_modern = [c for c in uniq if c.get("pin") and c["is_modern"]]
    pinned_legacy = [c for c in uniq if c.get("pin") and not c["is_modern"]]
    modern_rest = [c for c in uniq if not c.get("pin") and c["is_modern"]]
    legacy_rest = [c for c in uniq if not c.get("pin") and not c["is_modern"]]

    modern_budget = round(modern_fraction * target_rows)

    # pinned modern synth is the always-in base; as-is fills the remainder
    rem_modern = max(0, modern_budget - len(pinned_modern))
    modern_fill, modern_left = _select(modern_rest, rem_modern, modern_per_run_cap)
    # if the cap (not scarcity) bound the fill, relax it to reach the budget —
    # still 100% modern, so it only strengthens the tilt
    if len(modern_fill) < rem_modern and modern_left:
        extra, _ = _select(modern_left, rem_modern - len(modern_fill), target_rows)
        modern_fill.extend(extra)
    modern_sel = pinned_modern + modern_fill

    legacy_budget = max(0, target_rows - max(len(modern_sel), modern_budget))
    legacy_fill, _ = _select(legacy_rest, max(0, legacy_budget - len(pinned_legacy)), legacy_per_run_cap)
    legacy_sel = pinned_legacy + legacy_fill

    selected = modern_sel + legacy_sel
    rows = [c["row"] for c in selected]

    report = {
        "total": len(selected),
        "modern_count": len(modern_sel),
        "legacy_count": len(legacy_sel),
        "pinned_modern_synth": len(pinned_modern),
        "modern_asis_fill": len(modern_fill),
        "modern_fraction_target": modern_fraction,
        "modern_fraction_realized": (len(modern_sel) / len(selected)) if selected else 0.0,
        "modern_budget": modern_budget,
        "legacy_budget": legacy_budget,
        "modern_per_run_cap": modern_per_run_cap,
        "legacy_per_run_cap": legacy_per_run_cap,
        "unique_candidates": len(uniq),
        "modern_available": len(pinned_modern) + len(modern_rest),
        "legacy_available": len(pinned_legacy) + len(legacy_rest),
        "per_run": dict(Counter(c["run_id"] for c in selected)),
        "per_source": dict(Counter(c["source"] for c in selected)),
        "per_kind": dict(Counter(c["kind"] for c in selected)),
        "per_era_source": dict(
            Counter(f"{'modern' if c['is_modern'] else 'legacy'}:{c['source']}" for c in selected)
        ),
    }
    return {"rows": rows, "report": report}


# ---------------------------------------------------------------------------
# I/O helpers (lazy / stdlib-only at module load so the pure core stays import-cheap)
# ---------------------------------------------------------------------------
def _pair_hash(row: dict[str, Any]) -> str:
    p = re.sub(r"\s+", " ", str(row.get("prompt", ""))).strip().lower()
    r = re.sub(r"\s+", " ", str(row.get("raw_response", ""))).strip().lower()
    return hashlib.sha256(f"{p}\n---\n{r}".encode("utf-8")).hexdigest()


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _embedded_dt(run_id: str) -> datetime | None:
    m = re.search(r"(\d{8})", run_id)
    if m:
        return _parse_dt(f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}T00:00:00")
    m = re.search(r"(\d{4}-\d{2}-\d{2})", run_id)
    return _parse_dt(f"{m.group(1)}T00:00:00") if m else None


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


def _blocks(prompt: str) -> list[tuple[str, str]]:
    parts = _SPLIT.split(prompt or "")
    it = iter(parts[1:])
    return [(role, body.strip()) for role, body in zip(it, it)]


def _step_num(row: dict[str, Any]) -> int | None:
    m = re.search(r"step_(\d+)", row.get("interaction_type", "") or "")
    return int(m.group(1)) if m else None


def _recover_episode(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover empty per-turn completions from the next same-phase prompt
    (identical to aggregate_sft_topup._recover_episode, minus the drop counter)."""
    out = [row for row in rows if (row.get("raw_response") or "").strip()]
    turns = [
        row
        for row in rows
        if _step_num(row) is not None and not (row.get("raw_response") or "").strip()
    ]
    phases: list[list[dict[str, Any]]] = []
    for row in turns:
        if phases and row.get("workflow_step") == phases[-1][-1].get("workflow_step"):
            phases[-1].append(row)
        else:
            phases.append([row])
    for phase_rows in phases:
        for i in range(len(phase_rows) - 1):
            before = _blocks(phase_rows[i].get("prompt") or "")
            after = _blocks(phase_rows[i + 1].get("prompt") or "")
            if len(after) <= len(before):
                continue
            if [role for role, _ in after[: len(before)]] != [role for role, _ in before]:
                continue
            added = after[len(before):]
            if not added or added[0][0] != "assistant" or not added[0][1].strip():
                continue
            row = dict(phase_rows[i])
            row["raw_response"] = added[0][1]
            out.append(row)
    return out


def _trainable(row: dict[str, Any], *, max_prompt_chars: int) -> bool:
    if not row.get("success"):
        return False
    prompt = row.get("prompt")
    response = row.get("raw_response")
    if not isinstance(prompt, str) or not isinstance(response, str) or not response.strip():
        return False
    return max_prompt_chars <= 0 or len(prompt) <= max_prompt_chars


@dataclass(frozen=True)
class _Ledger:
    run_id: str
    generation_dir: Path
    last_completed_at: datetime
    episodes: list[dict[str, Any]]


def _discover_ledgers(base: Path) -> list[_Ledger]:
    ledgers: list[_Ledger] = []
    for path in sorted(base.rglob("all_episodes.jsonl")):
        if "aggregated_sft" in path.parts:
            continue
        episodes: list[dict[str, Any]] = []
        last: datetime | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ep = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(ep, dict):
                continue
            episodes.append(ep)
            dt = _parse_dt(ep.get("completed_at") or ep.get("started_at"))
            if dt is not None and (last is None or dt > last):
                last = dt
        if episodes and last is not None:
            gen_dir = path.parent
            ledgers.append(
                _Ledger(_run_id_for_generation_dir(gen_dir, base), gen_dir, last, episodes)
            )
    return sorted(ledgers, key=lambda x: x.last_completed_at, reverse=True)


def _tag(row: dict[str, Any], *, run_id: str, recency: datetime | None, is_modern: bool,
         source: str, kind: str, pin: bool = False) -> dict[str, Any]:
    out = dict(row)
    out["source_run_id"] = run_id
    out["composition_source"] = source
    out["composition_kind"] = kind
    out["composition_era"] = "modern" if is_modern else "legacy"
    out["composition_pinned"] = pin
    return {
        "run_id": run_id,
        "recency": recency.isoformat() if recency else "",
        "is_modern": is_modern,
        "source": source,
        "kind": kind,
        "pin": pin,
        "pair_hash": _pair_hash(row),
        "row": out,
    }


def build_candidates(
    *,
    base_synth_path: Path,
    cutoff: datetime,
    max_prompt_chars: int,
    exclude_synth_pair_kinds: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ledgers = _discover_ledgers(BASE)
    recency_by_run = {lg.run_id: lg.last_completed_at for lg in ledgers}

    def recency_for(run_id: str) -> datetime | None:
        return recency_by_run.get(run_id) or _embedded_dt(run_id)

    candidates: list[dict[str, Any]] = []

    # synth pool: reuse the already-built v2 base rows (tagged with source_run_id)
    synth_kept = 0
    synth_excluded = Counter()
    for line in base_synth_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not _trainable(row, max_prompt_chars=max_prompt_chars):
            continue
        meta = row.get("record_meta") if isinstance(row.get("record_meta"), dict) else {}
        kind = str(meta.get("pair_kind") or meta.get("task_kind") or "?")
        if kind in exclude_synth_pair_kinds:
            synth_excluded[kind] += 1
            continue
        run_id = str(row.get("source_run_id") or "unknown-run")
        rec = recency_for(run_id)
        is_modern = bool(rec and rec >= cutoff)
        candidates.append(
            # newer-run synth rows are the pinned base — never down-sampled
            _tag(row, run_id=run_id, recency=rec, is_modern=is_modern,
                 source="synth", kind=kind, pin=is_modern)
        )
        synth_kept += 1

    # as-is pool: recover transcript turns per run (all eras)
    asis_kept = 0
    for lg in ledgers:
        is_modern = lg.last_completed_at >= cutoff
        for ep in lg.episodes:
            if not ep.get("success"):
                continue
            for row in _recover_episode(ep.get("raw_training_rows") or []):
                if not _trainable(row, max_prompt_chars=max_prompt_chars):
                    continue
                kind = str(row.get("workflow_step") or "asis")
                candidates.append(
                    _tag(row, run_id=lg.run_id, recency=lg.last_completed_at,
                         is_modern=is_modern, source="asis", kind=kind)
                )
                asis_kept += 1

    stats = {
        "synth_candidates": synth_kept,
        "asis_candidates": asis_kept,
        "synth_excluded_by_pair_kind": dict(synth_excluded),
        "discovered_runs": len(ledgers),
        "run_recency": {lg.run_id: lg.last_completed_at.isoformat() for lg in ledgers},
    }
    return candidates, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-synth", default=str(DEFAULT_BASE_SYNTH))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--target-rows", type=int, default=4500)
    ap.add_argument("--modern-cutoff", default=DEFAULT_MODERN_CUTOFF)
    ap.add_argument("--modern-fraction", type=float, default=0.78)
    ap.add_argument("--modern-per-run-cap-frac", type=float, default=0.18)
    ap.add_argument("--legacy-per-run-cap-frac", type=float, default=0.06)
    ap.add_argument("--max-prompt-chars", type=int, default=50_000)
    ap.add_argument("--exclude-synth-pair-kind", action="append", default=[])
    args = ap.parse_args()

    cutoff = _parse_dt(f"{args.modern_cutoff}T00:00:00")
    if cutoff is None:
        raise SystemExit(f"bad --modern-cutoff: {args.modern_cutoff!r}")

    candidates, stats = build_candidates(
        base_synth_path=Path(args.base_synth),
        cutoff=cutoff,
        max_prompt_chars=args.max_prompt_chars,
        exclude_synth_pair_kinds={k for k in args.exclude_synth_pair_kind if k},
    )

    modern_cap = round(args.modern_per_run_cap_frac * args.target_rows)
    legacy_cap = round(args.legacy_per_run_cap_frac * args.target_rows)
    result = compose(
        candidates,
        target_rows=args.target_rows,
        modern_fraction=args.modern_fraction,
        modern_per_run_cap=modern_cap,
        legacy_per_run_cap=legacy_cap,
    )
    rows, report = result["rows"], result["report"]

    # unique, stable row_ids for the emitted file
    for i, row in enumerate(rows):
        row["row_id"] = f"recency-c:{i}:{row.get('row_id', i)}"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "sft_training_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")

    # realized era breakdown for the manifest
    era_rows = Counter(r.get("composition_era", "?") for r in rows)
    manifest = {
        "schema_version": 1,
        "created_for": "Approach C — newest-first as-is composition, hard recency tilt",
        "base_synth_path": str(Path(args.base_synth)),
        "modern_cutoff": args.modern_cutoff,
        "config": {
            "target_rows": args.target_rows,
            "modern_fraction": args.modern_fraction,
            "modern_per_run_cap": modern_cap,
            "legacy_per_run_cap": legacy_cap,
            "max_prompt_chars": args.max_prompt_chars,
            "exclude_synth_pair_kinds": sorted({k for k in args.exclude_synth_pair_kind if k}),
        },
        "candidate_stats": stats,
        "report": report,
        "rows_by_era": dict(era_rows),
        "sft_training_rows_path": rows_path.name,
        "policy": (
            "Newest-first composition of synth reasoning rows (reused from the v2 base, "
            "tagged by source_run_id) plus as-is transcript rows (recovered per run). "
            "Exact prompt/response de-dup keeps the newest run's copy. Modern runs "
            f"(completed >= {args.modern_cutoff}) fill {args.modern_fraction:.0%} of the "
            "budget; legacy runs are capped tighter so no single old run dominates."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )

    print(f"\n=== recency-c → {rows_path} ===")
    print(f"  total rows ............ {report['total']}")
    print(f"  modern / legacy ....... {report['modern_count']} / {report['legacy_count']}"
          f"  ({report['modern_fraction_realized']:.1%} modern)")
    print(f"  by source ............. {report['per_source']}")
    print(f"  caps (modern/legacy) .. {modern_cap} / {legacy_cap}")
    print(f"  synth/asis candidates . {stats['synth_candidates']} / {stats['asis_candidates']}")
    top = sorted(report["per_run"].items(), key=lambda kv: kv[1], reverse=True)[:8]
    print("  top runs:")
    for rid, n in top:
        print(f"    {n:>5}  {rid[:64]}")
    print(f"  manifest → {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
