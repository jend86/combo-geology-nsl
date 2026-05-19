"""Fetch Etherscan-verified Solidity source for a list of addresses.

Uses Etherscan's API v2 (single key, multi-chain). Writes each contract's
source to `tasks/forked_exploit_variations/<variation>/sources/<Label>.sol`.

Usage:
    ETHERSCAN_API_KEY=... uv run python scripts/fetch_verified_sources.py \
        --chain bsc \
        --variation pdz_flashloan_price_manip_2025_08 \
        --addresses 0x664201579057f50D23820d20558f4b61bd80BDda:TB_BUILD

If the contract is a "standard-json" verification, the script flattens the
first file (sufficient for most cases — exploits rarely touch every library
file). For proxy+implementation verifications, run this twice with the
resolved implementation address.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


CHAIN_IDS: dict[str, int] = {
    "mainnet": 1,
    "bsc": 56,
    "polygon": 137,
    "arbitrum": 42161,
    "base": 8453,
}


def fetch_source(api_key: str, chain: str, address: str) -> dict:
    chain_id = CHAIN_IDS[chain]
    url = (
        "https://api.etherscan.io/v2/api"
        f"?chainid={chain_id}"
        "&module=contract"
        "&action=getsourcecode"
        f"&address={address}"
        f"&apikey={urllib.parse.quote(api_key)}"
    )
    with urllib.request.urlopen(url, timeout=30) as r:
        payload = json.loads(r.read().decode())
    if payload.get("status") != "1":
        raise RuntimeError(
            f"{chain}:{address}: Etherscan returned {payload.get('message')!r}: "
            f"{payload.get('result')!r}"
        )
    result = payload["result"]
    if not result:
        raise RuntimeError(f"{chain}:{address}: empty result")
    item = result[0]
    if not item.get("SourceCode"):
        raise RuntimeError(
            f"{chain}:{address}: no verified source on Etherscan "
            f"(contract not verified? try a different address)"
        )
    return item


def extract_flat_source(item: dict) -> str:
    """Normalize Etherscan's three source-code formats into a single .sol string.

    Format 1: plain Solidity source in item["SourceCode"].
    Format 2: JSON object starting with `{` (multi-file standard input).
    Format 3: double-wrapped JSON starting with `{{` (Etherscan quirk for
              Solidity >0.6 standard-json).
    """
    src = item["SourceCode"]
    stripped = src.strip()
    if not stripped.startswith("{"):
        return src

    # Standard-json is either single- or double-wrapped. Try both.
    try:
        if stripped.startswith("{{"):
            parsed = json.loads(stripped[1:-1])
        else:
            parsed = json.loads(stripped)
    except json.JSONDecodeError:
        # Fallback: return as-is. Developer may need to hand-edit.
        return src

    sources = parsed.get("sources", {})
    if not sources:
        return src
    chunks: list[str] = []
    for path, spec in sources.items():
        content = spec.get("content", "")
        chunks.append(f"// =============================== {path}\n{content}\n")
    return "\n".join(chunks)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch verified source from Etherscan.")
    ap.add_argument("--chain", required=True, choices=list(CHAIN_IDS))
    ap.add_argument(
        "--variation", required=True,
        help="Variation directory name under tasks/forked_exploit_variations/"
    )
    ap.add_argument(
        "--addresses", required=True,
        help="Comma-separated list of 0xADDRESS:Label pairs"
    )
    ap.add_argument(
        "--variations-root",
        default=str(Path(__file__).parent.parent / "tasks" / "forked_exploit_variations"),
    )
    args = ap.parse_args()

    api_key = os.environ.get("ETHERSCAN_API_KEY")
    if not api_key:
        print("ETHERSCAN_API_KEY env var required", file=sys.stderr)
        return 2

    out_dir = Path(args.variations_root) / args.variation / "sources"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [p.strip() for p in args.addresses.split(",") if p.strip()]
    for pair in pairs:
        if ":" not in pair:
            print(f"SKIP {pair}: expected 0xADDRESS:Label", file=sys.stderr)
            continue
        address, label = pair.split(":", 1)
        print(f"Fetching {args.chain} {address} → {label}.sol")
        try:
            item = fetch_source(api_key, args.chain, address)
        except Exception as exc:
            print(f"  FAIL: {exc}", file=sys.stderr)
            continue
        source = extract_flat_source(item)
        target = out_dir / f"{label}.sol"
        target.write_text(source)
        print(f"  wrote {target} ({len(source)} bytes)")
        time.sleep(0.25)  # Etherscan free tier: 5 req/s

    return 0


if __name__ == "__main__":
    sys.exit(main())
