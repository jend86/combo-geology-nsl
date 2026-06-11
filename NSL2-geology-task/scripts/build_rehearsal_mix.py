#!/usr/bin/env python3
"""Build the rehearsal mix as local SFT JSONL files.

Each rehearsal source is subsampled + reformatted into the uniform
``{prompt, raw_response, success, source}`` schema that
``src/train/qlora.py::_load_self_generated_sft_rows`` consumes. The files are
then passed as EXTRA ``--training-data`` paths alongside the task rows, so
rehearsal rides the identical Gemma chat-template + completion-mask +
concatenate-and-shuffle path as task data (format consistency by construction;
no trainer code change).

The adapter functions are pure (stdlib only); ``datasets`` is imported lazily so
unit tests can exercise the adapters without the heavy dependency.

Current lean is to allocate 30% of rows to rehearsal data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

DEFAULT_SEED = 3407
DEFAULT_OUT = Path("data/kazakhstan/feature-hypothesis/rehearsal/20260610-gen2-mix")

# Default per-source row budgets (~4k total = ~31% of the 9k task rows; chat
# prioritized). Tunable via CLI.
DEFAULT_ROWS = {
    "geology": 1000,
    "chat": 1200,
    "tool": 800,
    "code": 600,
    "math": 400,
}


# --------------------------------------------------------------------------- #
# Pure format adapters: source-row dict -> uniform SFT row dict (or None).
# --------------------------------------------------------------------------- #

def _row(prompt: str, raw_response: str, source: str) -> dict[str, Any] | None:
    prompt = (prompt or "").strip()
    raw_response = (raw_response or "").strip()
    if not prompt or not raw_response:
        return None
    return {"prompt": prompt, "raw_response": raw_response, "success": True, "source": source}


def adapt_chat(row: dict[str, Any]) -> dict[str, Any] | None:
    """UltraChat: ``{messages: [{role, content}, ...]}`` -> first user->assistant."""
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    user = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        if role == "user" and user is None:
            user = content
        elif role == "assistant" and user is not None:
            return _row(user, content, "chat")
    return None


def adapt_instruction(
    row: dict[str, Any], *, query_field: str, response_field: str, source: str
) -> dict[str, Any] | None:
    """Magicoder (instruction/response) / MetaMathQA (query/response)."""
    return _row(row.get(query_field, ""), row.get(response_field, ""), source)


def adapt_toolace(row: dict[str, Any]) -> dict[str, Any] | None:
    """ToolACE: ``{system, conversations:[{from, value}]}``.

    Keeps ToolACE's *self-consistent* format: the function-listing system prompt
    is carried into the prompt so the ``[Func(args)]`` response surface is
    self-justified (preserves general tool-selection reasoning without imposing a
    format that conflicts with Gemma's native ``<tool_call>`` template, which the
    harness elicits separately at inference). Response = first assistant turn.
    """
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        return None
    system_raw = row.get("system")
    system = system_raw if isinstance(system_raw, str) else ""
    user = None
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        who = turn.get("from")
        value = turn.get("value")
        if not isinstance(value, str):
            continue
        if who == "user" and user is None:
            user = value
        elif who == "assistant" and user is not None:
            prompt = f"{system.strip()}\n\n{user}".strip() if system.strip() else user
            return _row(prompt, value, "tool")
    return None


def adapt_geology(
    text: str, *, prompt_chars: int = 256, max_chars: int = 2048
) -> dict[str, Any] | None:
    """Raw geoscience passage -> continuation row (mirrors the legacy rehearsal
    loader in qlora.py so the geology behaviour is unchanged)."""
    if not isinstance(text, str):
        return None
    text = " ".join(text.split())
    if not text:
        return None
    text = text[:max_chars]
    excerpt = text[:prompt_chars].rstrip()
    prompt = "Continue the following geoscience passage"
    prompt += f":\n\n{excerpt}" if excerpt else "."
    return _row(prompt, text, "geology")


# --------------------------------------------------------------------------- #
# Source registry + I/O (lazy datasets import).
# --------------------------------------------------------------------------- #

def _make_sources(rows: dict[str, int], prompt_chars: int, max_chars: int) -> list[dict[str, Any]]:
    return [
        {
            "key": "geology", "dataset": "ClickNoow/5k-dataset-geogpt-fineweb",
            "split": "train", "rows": rows["geology"],
            "adapt": lambda r: adapt_geology(
                r.get("text", ""), prompt_chars=prompt_chars, max_chars=max_chars),
        },
        {
            "key": "chat", "dataset": "HuggingFaceH4/ultrachat_200k",
            "split": "train_sft", "rows": rows["chat"],
            "adapt": adapt_chat,
        },
        {
            "key": "tool", "dataset": "Team-ACE/ToolACE",
            "split": "train", "rows": rows["tool"],
            "adapt": adapt_toolace,
        },
        {
            "key": "code", "dataset": "ise-uiuc/Magicoder-Evol-Instruct-110K",
            "split": "train", "rows": rows["code"],
            "adapt": lambda r: adapt_instruction(
                r, query_field="instruction", response_field="response", source="code"),
        },
        {
            "key": "math", "dataset": "meta-math/MetaMathQA",
            "split": "train", "rows": rows["math"],
            "adapt": lambda r: adapt_instruction(
                r, query_field="query", response_field="response", source="math"),
        },
    ]


def build_source(spec: dict[str, Any], *, seed: int, buffer_size: int = 20000) -> list[dict[str, Any]]:
    from datasets import load_dataset  # lazy

    ds = load_dataset(spec["dataset"], split=spec["split"], streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    adapt: Callable[[dict[str, Any]], dict[str, Any] | None] = spec["adapt"]
    want = int(spec["rows"])
    out: list[dict[str, Any]] = []
    seen = 0
    for row in ds:
        seen += 1
        adapted = adapt(row)
        if adapted is not None:
            out.append(adapted)
            if len(out) >= want:
                break
        if seen > want * 50 + 100000:  # safety bound against pathological reject rates
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--prompt-chars", type=int, default=256)
    ap.add_argument("--max-chars", type=int, default=2048)
    for key, default in DEFAULT_ROWS.items():
        ap.add_argument(f"--{key}-rows", type=int, default=default)
    ap.add_argument("--only", action="append", default=[],
                    help="build only these source keys (repeatable)")
    args = ap.parse_args()

    rows = {k: getattr(args, f"{k}_rows") for k in DEFAULT_ROWS}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = _make_sources(rows, args.prompt_chars, args.max_chars)
    if args.only:
        sources = [s for s in sources if s["key"] in set(args.only)]

    manifest: dict[str, Any] = {"seed": args.seed, "sources": {}, "files": [], "total": 0}
    for spec in sources:
        built = build_source(spec, seed=args.seed)
        path = out_dir / f"rehearsal_{spec['key']}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for r in built:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["sources"][spec["key"]] = {
            "dataset": spec["dataset"], "split": spec["split"],
            "requested": spec["rows"], "written": len(built), "path": str(path),
        }
        manifest["files"].append(str(path))
        manifest["total"] += len(built)
        print(f"[{spec['key']:8}] {len(built):5}/{spec['rows']:<5} -> {path}", file=sys.stderr)
        if built:
            ex = built[0]
            print(f"           e.g. prompt[:80]={ex['prompt'][:80]!r}", file=sys.stderr)
            print(f"                resp[:80]={ex['raw_response'][:80]!r}", file=sys.stderr)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nTOTAL rehearsal rows: {manifest['total']} -> {out_dir}", file=sys.stderr)
    print("training-data flags:", file=sys.stderr)
    for f in manifest["files"]:
        print(f"  --training-data {f}", file=sys.stderr)


if __name__ == "__main__":
    main()
