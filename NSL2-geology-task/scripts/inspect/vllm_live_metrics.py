#!/usr/bin/env python3
"""Live windowed monitor for a running inference server's Prometheus /metrics.

Scrapes the endpoint back-to-back every ``--window`` seconds and reports the
*windowed* deltas — preemptions, prefix-cache hit ratio, cached-prompt ratio,
and token throughput — instead of the cumulative-since-boot counters (which
are dominated by cold-start and mislead tuning). Use it as the live diagnostic
loop for capacity tuning; it runs against the active server with no restart.

Usage (inside the nix dev shell):

    uv run python scripts/inspect/vllm_live_metrics.py --window 60 --count 5
    uv run python scripts/inspect/vllm_live_metrics.py --window 30 --count 0 \
        --jsonl /tmp/vllm_windows.jsonl   # 0 = run until Ctrl-C

Counters that go backwards (server restart) surface as ``None`` for that
window rather than a bogus negative rate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Make ``src`` importable when run as a bare script.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.observability.vllm_metrics import (  # noqa: E402
    inference_metrics_delta,
    snapshot_inference_metrics,
)


def _fmt_pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


def _fmt(x, width: int = 8) -> str:
    if x is None:
        return "n/a".rjust(width)
    if isinstance(x, float):
        return f"{x:,.1f}".rjust(width)
    return f"{x:,}".rjust(width)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--backend", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument(
        "--window", type=float, default=60.0, help="seconds per window"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="number of windows to report (0 = until interrupted)",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--jsonl", default=None, help="append per-window delta dicts here"
    )
    args = parser.parse_args()

    def scrape():
        return snapshot_inference_metrics(
            args.url, backend=args.backend, api_key=args.api_key
        )

    prev = scrape()
    if prev is None:
        print(f"ERROR: could not scrape {args.url}", file=sys.stderr)
        return 1

    header = (
        f"# live windowed metrics for {args.url} "
        f"(window={args.window:g}s, backend={args.backend})\n"
        f"{'window':>6} {'preempt':>8} {'pfx_hit':>8} {'cached':>8} "
        f"{'kv_use':>7} {'run':>4} {'wait':>4} {'prmpt/s':>9} {'gen/s':>8}"
    )
    print(header, flush=True)

    jsonl_handle = open(args.jsonl, "a", encoding="utf-8") if args.jsonl else None
    # Accumulators for an overall summary across reported windows.
    tot_preempt = 0
    tot_pc_hits = 0
    tot_pc_queries = 0
    tot_prompt = 0
    tot_cached = 0
    tot_window = 0.0
    n = 0

    try:
        while args.count == 0 or n < args.count:
            time.sleep(args.window)
            curr = scrape()
            if curr is None:
                print("  (scrape failed; retrying next window)", flush=True)
                continue
            d = inference_metrics_delta(prev, curr)
            prev = curr
            n += 1

            print(
                f"{n:>6} {_fmt(d.preemptions):>8} {_fmt_pct(d.prefix_cache_hit_rate):>8} "
                f"{_fmt_pct(d.prompt_tokens_cached_rate):>8} "
                f"{_fmt_pct((d.kv_cache_usage_pct or 0) / 100 if d.kv_cache_usage_pct is not None else None):>7} "
                f"{_fmt(d.num_requests_running, 4):>4} {_fmt(d.num_requests_waiting, 4):>4} "
                f"{_fmt(d.prompt_tokens_per_second, 9):>9} "
                f"{_fmt(d.generation_tokens_per_second, 8):>8}",
                flush=True,
            )
            if jsonl_handle is not None:
                jsonl_handle.write(json.dumps(asdict(d)) + "\n")
                jsonl_handle.flush()

            if d.preemptions is not None:
                tot_preempt += d.preemptions
            if d.prefix_cache_hits is not None:
                tot_pc_hits += d.prefix_cache_hits
            if d.prefix_cache_queries is not None:
                tot_pc_queries += d.prefix_cache_queries
            if d.prompt_tokens is not None:
                tot_prompt += d.prompt_tokens
            if d.prompt_tokens_cached is not None:
                tot_cached += d.prompt_tokens_cached
            tot_window += d.window_seconds
    except KeyboardInterrupt:
        print("\n(interrupted)", flush=True)
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()

    if n > 0:
        overall_hit = tot_pc_hits / tot_pc_queries if tot_pc_queries > 0 else None
        overall_cached = tot_cached / tot_prompt if tot_prompt > 0 else None
        print(
            f"\n# SUMMARY over {n} windows ({tot_window:.0f}s): "
            f"preemptions={tot_preempt}, "
            f"prefix_cache_hit_rate={_fmt_pct(overall_hit).strip()}, "
            f"prompt_tokens_cached_rate={_fmt_pct(overall_cached).strip()}, "
            f"avg_prompt_tok/s={tot_prompt / tot_window if tot_window else 0:,.0f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
