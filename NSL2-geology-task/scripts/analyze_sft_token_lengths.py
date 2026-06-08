"""Measure the tokenized length distribution of an SFT dataset.

Run this BEFORE picking ``--max-seq-length`` for ``qlora.py`` so the choice is
data-driven rather than a guess. It mirrors the trainer's masking boundary: each
successful row is templated as ``apply_chat_template([user=prompt,
assistant=raw_response], add_generation_prompt=False)`` for the full sequence,
and the masked query prefix is templated as ``apply_chat_template([user=prompt],
add_generation_prompt=True)``.

Truncation matters because the trainer truncates the templated ``text`` to
``max_length`` with no packing. For a ``prompt + response`` sequence that drops
the *tail* (the assistant answer we are training on). So we report two failure
modes per threshold:

  * full > T   : sequence truncated; some of the response is lost.
  * prompt >= T: the whole response is lost (no learning signal left).

Usage (on the pod, where the base-model tokenizer is already cached):

    python scripts/analyze_sft_token_lengths.py \
        --rows data/kazakhstan/feature-hypothesis/aggregated_sft/20260606-v2-topup-4500/sft_training_rows.jsonl \
        --base-model unsloth/gemma-4-31B-it-unsloth-bnb-4bit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL = "unsloth/gemma-4-31B-it-unsloth-bnb-4bit"
DEFAULT_THRESHOLDS = (2048, 4096, 8192, 16384)


def _load_tokenizer(base_model: str) -> Any:
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(base_model)
    except Exception:  # noqa: BLE001 — VLM repos sometimes need the processor
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(base_model)


def _text_tokenizer(tokenizer: Any) -> Any:
    # Match qlora._tokenized_length: gemma-4's processor sends the first
    # positional arg to `images`; the inner `.tokenizer` is the text one.
    return getattr(tokenizer, "tokenizer", tokenizer)


def _token_count(text_tok: Any, text: str) -> int:
    encoded = text_tok(text, truncation=False, padding=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def _templated(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, help="path to sft_training_rows.jsonl")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="max_seq_length candidates to evaluate",
    )
    args = parser.parse_args()

    tokenizer = _load_tokenizer(args.base_model)
    text_tok = _text_tokenizer(tokenizer)

    full_lengths: list[int] = []
    prompt_lengths: list[int] = []
    skipped = 0
    rows_path = Path(args.rows)
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not payload.get("success"):
            skipped += 1
            continue
        prompt = payload.get("prompt")
        response = payload.get("raw_response")
        if not isinstance(prompt, str) or not isinstance(response, str):
            skipped += 1
            continue

        full_text = _templated(
            tokenizer,
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            add_generation_prompt=False,
        )
        prompt_text = _templated(
            tokenizer,
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
        full_lengths.append(_token_count(text_tok, full_text))
        prompt_lengths.append(_token_count(text_tok, prompt_text))

    n = len(full_lengths)
    if n == 0:
        raise SystemExit("no trainable rows found (success + str prompt/response)")

    full_sorted = sorted(full_lengths)
    print(f"\n=== SFT token-length analysis: {rows_path} ===")
    print(f"base model ........ {args.base_model}")
    print(f"trainable rows .... {n}  (skipped {skipped})")
    print("\nfull sequence (prompt + response) token lengths:")
    for label, q in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)):
        print(f"  {label} ............ {_percentile(full_sorted, q)}")
    print(f"  max ............ {full_sorted[-1]}")

    print("\ntruncation impact by max_seq_length:")
    print(f"  {'T':>7}  {'full>T (partial loss)':>24}  {'prompt>=T (total loss)':>24}")
    for threshold in sorted(args.thresholds):
        partial = sum(1 for length in full_lengths if length > threshold)
        total = sum(1 for length in prompt_lengths if length >= threshold)
        print(
            f"  {threshold:>7}  "
            f"{partial:>6} ({100 * partial / n:5.1f}%)            "
            f"{total:>6} ({100 * total / n:5.1f}%)"
        )
    print(
        "\nPick the smallest T where partial-loss% is acceptable; rows with "
        "prompt>=T are unrecoverable at that T (clip/drop them or raise T)."
    )


if __name__ == "__main__":
    main()
