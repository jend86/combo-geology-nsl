#!/usr/bin/env python3
"""Sample + analyze episode failure modes and successes for the live run.

Failure taxonomy lives nested in all_episodes.jsonl:
  trajectory.termination_category / .termination_reason   (agent_failure, ...)
  task_breakdown.no_feature / .stage_completed
Admission verdicts live in the store (admitted/index.json + the .npy layers).

usage: python analyze_episodes.py [N_samples]
"""
import sys
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path("/home/elijah/combo-geology-nsl")
FH = REPO / "NSL2-geology-task/data/kazakhstan/feature-hypothesis"
GEN = FH / "generations/generation_0"
ST = FH / "store/teniz_basin"
SHAPE = (200, 200, 8)


def get_nested(d, path, default=None):
    cur = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur if cur is not None else default


def load_episodes():
    p = GEN / "all_episodes.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def reshape(a):
    return a.reshape(SHAPE) if a.size == int(np.prod(SHAPE)) else a


def ncol(f):
    f = reshape(f)
    return int(np.count_nonzero(f.any(axis=2))) if f.ndim == 3 else 0


def nvox(f):
    return int(np.count_nonzero(f))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    eps = load_episodes()
    print(f"=== EPISODES: {len(eps)} ===")
    if eps:
        succ = [e for e in eps if e.get("success") is True]
        fail = [e for e in eps if e.get("success") is not True]
        print(f"success: {len(succ)}/{len(eps)} ({100*len(succ)/len(eps):.0f}%)")
        # failure taxonomy
        tcat = Counter(get_nested(e, "trajectory.termination_category", "?") for e in fail)
        treas = Counter(get_nested(e, "trajectory.termination_reason", "?") for e in fail)
        nofeat = sum(1 for e in fail if get_nested(e, "task_breakdown.no_feature") is True)
        stage = Counter(get_nested(e, "task_breakdown.stage_completed", "?") for e in eps)
        print(f"FAILURE termination_category: {dict(tcat.most_common())}")
        print(f"FAILURE termination_reason  : {dict(treas.most_common(6))}")
        print(f"FAILURE no_feature (no layer built): {nofeat}/{len(fail)}")
        print(f"stage_completed (all eps)   : {dict(stage.most_common())}")
        durs = [e.get("duration_seconds") for e in eps if isinstance(e.get("duration_seconds"), (int, float))]
        turns = [e.get("llm_turns_count") for e in eps if isinstance(e.get("llm_turns_count"), (int, float))]
        if durs:
            print(f"duration_s: median={statistics.median(durs):.0f} "
                  f"min={min(durs):.0f} max={max(durs):.0f}")
        if turns:
            print(f"llm_turns : median={statistics.median(turns):.0f} max={max(turns)}")
        print(f"\n--- {min(n,len(succ))} SUCCESS samples (id, turns, dur, score) ---")
        for e in succ[-n:]:
            print(f"  {e.get('episode_id')} turns={e.get('llm_turns_count')} "
                  f"dur={e.get('duration_seconds',0):.0f}s score={e.get('score')}")
        print(f"--- {min(n,len(fail))} FAILURE samples (id, cat, no_feat, turns, dur) ---")
        for e in fail[-n:]:
            print(f"  {e.get('episode_id')} cat={get_nested(e,'trajectory.termination_category','?')} "
                  f"no_feat={get_nested(e,'task_breakdown.no_feature')} "
                  f"turns={e.get('llm_turns_count')} dur={e.get('duration_seconds',0):.0f}s")

    # admitted-layer coherence
    adm = sorted((ST / "admitted/layers").glob("*.npy"))
    print(f"\n=== ADMITTED LAYERS: {len(adm)} ===")
    for p in adm:
        try:
            f = reshape(np.load(p).astype(float))
            uniq = np.unique(f[f != 0])
            kind = "binary" if uniq.size <= 1 else f"{uniq.size}-val"
            print(f"  {p.stem[:46]:46s} cols={ncol(f):5d} vox={nvox(f):6d} {kind}")
        except Exception as e:
            print(f"  {p.stem[:46]} ERR {e}")
    rej = sorted((ST / "rejected").glob("ep_*"))
    print(f"=== REJECTED (store): {len(rej)} ===")
    for ep in rej[-min(n, len(rej)):]:
        md = ep / "metadata.json"
        if md.exists():
            try:
                ev = json.loads(md.read_text()).get("evaluate", {})
                print(f"  {ep.name[:22]} admitted={ev.get('admitted')} "
                      f"valid={ev.get('validity_passed')} "
                      f"lift={ev.get('candidate_predictor_lift_mean')} "
                      f"bic={ev.get('bic_delta')} note={ev.get('score_note')}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
