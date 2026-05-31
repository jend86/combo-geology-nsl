"""Compare hypothesis diversity between runs (RAW, pre-curation).

For each successful episode we backfill the hypothesis + whether it was a
crossbreed (has parents). We deliberately do NOT use the SFT export, because the
export's L2 curation family-caps the dominant cluster — which would hide the
monoculture we want to measure.

Metrics (per cohort: ALL successful, and CROSSBREED-only):
  n                  episodes with a recoverable hypothesis
  distinct           exact-distinct hypotheses (whitespace/case normalised)
  families           distinct lexical families (transform's _hypothesis_head)
  top_family_share   fraction in the single largest family   (higher = monoculture)
  norm_entropy       Shannon entropy of family dist / ln(families)  (1=even, 0=skewed)
  eff_families       exp(entropy) = effective number of families
  mean_jaccard       mean pairwise content-word Jaccard       (higher = less diverse; N-robust)
  ttr                distinct content words / total content tokens

Plus a matched-N subsample so the young run is compared fairly against the
mature one (distinct/families grow with N; mean_jaccard is N-robust).

usage: compare_diversity.py <run_dir_A> <label_A> <run_dir_B> <label_B>
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

from src.task.loader import load_task
from src.typing.trajectory import EpisodeTrajectory, GenerationData
from tasks.feature_hypothesis_kazakhstan import _STOP_WORDS, _content_words

TASK = "tasks.feature_hypothesis_kazakhstan.FeatureHypothesisKazakhstanTask"
_MAX_PAIRS = 40_000


def _family(h: str) -> str:
    words = [w for w in re.findall(r"[a-z]+", h.lower()) if w not in _STOP_WORDS]
    return " ".join(words[:7]) or "other"


def extract(run_dir: Path, tr) -> list[tuple[str, bool]]:
    """Return [(hypothesis, is_crossbreed)] for successful episodes, streaming."""
    out: list[tuple[str, bool]] = []
    path = run_dir / "all_episodes.jsonl"
    seen = 0
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not payload.get("success"):
            continue
        seen += 1
        ep = EpisodeTrajectory.from_dict(payload)
        gd = GenerationData(generation_id=ep.generation_id)
        gd.add_episode(ep)
        groups = gd.get_successful_training_row_groups()
        if not groups:
            continue
        rec = tr._backfill_record(groups[0], payload)
        hyp = str(rec.get("hypothesis") or "").strip()
        if not hyp:
            continue
        out.append((hyp, bool(rec.get("parent_ids"))))
    print(f"    [{run_dir.parts[-3]}] successful={seen} with_hypothesis={len(out)}")
    return out


def _mean_jaccard(content_words: list[set[str]], rng: random.Random) -> float:
    n = len(content_words)
    if n < 2:
        return 0.0
    pairs = list(combinations(range(n), 2))
    if len(pairs) > _MAX_PAIRS:
        pairs = rng.sample(pairs, _MAX_PAIRS)
    sims = []
    for i, j in pairs:
        a, b = content_words[i], content_words[j]
        if a or b:
            sims.append(len(a & b) / len(a | b))
    return sum(sims) / len(sims) if sims else 0.0


def metrics(hyps: list[str], rng: random.Random) -> dict:
    n = len(hyps)
    if n == 0:
        return {"n": 0}
    norm = [re.sub(r"\s+", " ", h.strip().lower()) for h in hyps]
    fams = Counter(_family(h) for h in hyps)
    nf = len(fams)
    top = max(fams.values()) / n
    probs = [c / n for c in fams.values()]
    H = -sum(p * math.log(p) for p in probs)
    cw = [_content_words(h) for h in hyps]
    tokens = [w for h in hyps for w in re.findall(r"[a-z]+", h.lower()) if w not in _STOP_WORDS]
    ttr = len(set(tokens)) / len(tokens) if tokens else 0.0
    return {
        "n": n,
        "distinct": len(set(norm)),
        "families": nf,
        "top_family_share": round(top, 3),
        "norm_entropy": round(H / math.log(nf), 3) if nf > 1 else 0.0,
        "eff_families": round(math.exp(H), 1),
        "mean_jaccard": round(_mean_jaccard(cw, rng), 3),
        "ttr": round(ttr, 3),
    }


def subsample(hyps: list[str], target_n: int, rng: random.Random, k: int = 200) -> dict:
    if len(hyps) <= target_n:
        return {"note": f"n={len(hyps)} <= target {target_n}; no subsample"}
    fam_counts, jac = [], []
    for _ in range(k):
        sample = rng.sample(hyps, target_n)
        fam_counts.append(len({_family(h) for h in sample}))
        jac.append(_mean_jaccard([_content_words(h) for h in sample], rng))
    return {
        "target_n": target_n,
        "mean_families": round(sum(fam_counts) / k, 2),
        "mean_jaccard": round(sum(jac) / k, 3),
    }


def report(label: str, data: list[tuple[str, bool]], rng: random.Random) -> dict:
    allh = [h for h, _ in data]
    cross = [h for h, c in data if c]
    surv = [h for h, c in data if not c]
    print(f"\n=== {label} ===")
    print(f"  episodes: total={len(allh)}  crossbreed={len(cross)}  survey={len(surv)}")
    print(f"  ALL        : {metrics(allh, rng)}")
    print(f"  CROSSBREED : {metrics(cross, rng)}")
    print(f"  SURVEY     : {metrics(surv, rng)}")
    return {"all": allh, "cross": cross, "surv": surv}


def main() -> None:
    a_dir, a_label, b_dir, b_label = (
        Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
    )
    task = load_task(TASK)
    tr = task.training_data_transforms()[0]
    rng = random.Random(0)

    print("Extracting (raw, pre-curation hypotheses)...")
    a = extract(a_dir, tr)
    b = extract(b_dir, tr)

    ra = report(a_label, a, rng)
    rb = report(b_label, b, rng)

    # Matched-N crossbreed comparison (fair vs the young run).
    n_target = min(len(ra["cross"]), len(rb["cross"]))
    print(f"\n=== MATCHED-N crossbreed comparison (target n={n_target}, 200 trials) ===")
    print(f"  {a_label} crossbreed @n={n_target}: {subsample(ra['cross'], n_target, rng)}")
    print(f"  {b_label} crossbreed @n={n_target}: {subsample(rb['cross'], n_target, rng)}")


if __name__ == "__main__":
    main()
