"""Attacker-view prewarm probes for the shared rpc-cache sqlite.

Fires the read-only RPC calls an exploiter agent would plausibly make while
investigating a new forked-exploit task, so those calls are cache-hits on
the first real training episode. Complements ForkedExploitTask.populate(),
which warms only the drain-target balance grid the scorer needs.

Probe plan sources (Approach-D schema):
  - inventory addresses from ``[[environment.contracts]]``
  - drain-target addresses + tracked tokens from ``[[success.drain_targets]]``
  - gain-token addresses from ``[[success.gain_tokens]]``
  - address literals regex-extracted from the DeFiHackLabs PoC and every
    sources/*.sol in the variation directory
  - per-chain AMM factory + router registry (hard-coded below)

All block-referenced RPC calls pin an explicit hex block number so the
rpc-cache (~/nsl2/docker/forked-exploit-compose/rpc-cache-src/main.go)
will cache them — "latest" or missing block refs are not cached.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docker.models.containers import Container
from loguru import logger

from tasks.common.foundry_exec import coerce_exec_result, exec_run_with_timeout


ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
DEAD_ADDRESS = "0x000000000000000000000000000000000000dead"
ATTACKER_EOA = "0x70997970c51812dc3a010c7d01b50e0d17dc79c8"  # anvil default #1

_PLACEHOLDER_ADDRESSES = {ZERO_ADDRESS, DEAD_ADDRESS}

# EIP-1967 storage slots + OpenZeppelin legacy transparent-proxy slot.
# Reading these tells an exploiter whether the target is a proxy and where
# the implementation lives — a near-universal first step.
_PROXY_SLOTS: tuple[tuple[str, str], ...] = (
    ("eip1967_impl",
     "0x360894a13ba1a3210667c828492db98dcbd39a2c8e1a6a07f0efc26c3a8ed5b1"),
    ("eip1967_admin",
     "0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103"),
    ("eip1967_beacon",
     "0xa3f0ad74e5423aebfd80d3ef4346578335a9a72aeaee59ff6cb3582b35133d50"),
    ("oz_legacy_impl",
     "0x7050c9e0f4ca769c69bd3a8ef740bc37934f8e2c036e5a723fd8ee048ed3f8c3"),
)

_ADMIN_INTROSPECT_SIGS: tuple[str, ...] = (
    "owner()(address)",
    "admin()(address)",
    "pendingOwner()(address)",
    "paused()(bool)",
    "implementation()(address)",
    "beacon()(address)",
)

_ERC20_METADATA_SIGS: tuple[str, ...] = (
    "name()(string)",
    "symbol()(string)",
    "decimals()(uint8)",
    "totalSupply()(uint256)",
)

_PAIR_SIGS: tuple[str, ...] = (
    "token0()(address)",
    "token1()(address)",
    "factory()(address)",
    "getReserves()(uint112,uint112,uint32)",
    "kLast()(uint256)",
    "price0CumulativeLast()(uint256)",
    "price1CumulativeLast()(uint256)",
)

_VAULT_SIGS: tuple[str, ...] = (
    "asset()(address)",
    "totalAssets()(uint256)",
)


@dataclass(frozen=True)
class FactoryRec:
    label: str
    address: str
    kind: str  # "uniV2" | "uniV3"


# Canonical AMM factories per chain. Source cross-references:
#   UniswapV2 — docs.uniswap.org/contracts/v2/reference/smart-contracts/factory
#   UniswapV3 — docs.uniswap.org/contracts/v3/reference/deployments
#   PancakeSwap V2 — docs.pancakeswap.finance/developers/smart-contracts
#   SushiSwap — docs.sushi.com/docs/Products/Classic%20AMM/Deployment%20Addresses
#   Aerodrome (Base) — aerodrome.finance/docs
# V3 getPool(tokenA,tokenB,fee) requires a fee-tier loop; left for a follow-up.
CHAIN_FACTORIES: dict[str, tuple[FactoryRec, ...]] = {
    "mainnet": (
        FactoryRec("UniswapV2", "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f", "uniV2"),
        FactoryRec("SushiSwapV2", "0xc0aee478e3658e2610c5f7a4a2e1777ce9e4f2ac", "uniV2"),
        FactoryRec("UniswapV3", "0x1f98431c8ad98523631ae4a59f267346ea31f984", "uniV3"),
    ),
    "bsc": (
        FactoryRec("PancakeV2", "0xca143ce32fe78f1f7019d7d551a6402fc5350c73", "uniV2"),
        FactoryRec("PancakeV1", "0xbcfccbde45ce874adcb698cc183debcf17952812", "uniV2"),
        FactoryRec("ApeSwap", "0x0841bd0b734e4f5853f0dd8d7ea041c241fb0da6", "uniV2"),
        FactoryRec("BakerySwap", "0x01bf7c66c6bd861915cdaae475042d3c4bae16a7", "uniV2"),
        FactoryRec("BiSwap", "0x858e3312ed3a876947ea49d572a7c42de08af7ee", "uniV2"),
    ),
    "polygon": (
        FactoryRec("QuickSwap", "0x5757371414417b8c6caad45baef941abc7d3ab32", "uniV2"),
        FactoryRec("SushiV2", "0xc35dadb65012ec5796536bd9864ed8773abc74c4", "uniV2"),
        FactoryRec("UniswapV3", "0x1f98431c8ad98523631ae4a59f267346ea31f984", "uniV3"),
    ),
    "arbitrum": (
        FactoryRec("SushiV2", "0xc35dadb65012ec5796536bd9864ed8773abc74c4", "uniV2"),
        FactoryRec("Camelot", "0x6eccab422d763ac031210895c81787e87b43a652", "uniV2"),
        FactoryRec("UniswapV3", "0x1f98431c8ad98523631ae4a59f267346ea31f984", "uniV3"),
    ),
    "base": (
        FactoryRec("Aerodrome", "0x420dd381b31aef6683db6b902084cb0ffece40da", "uniV2"),
        FactoryRec("BaseSwap", "0xfda619b6d20975be80a10332cd39b9a4b0faa8bb", "uniV2"),
        FactoryRec("UniswapV3", "0x33128a8fc17869897dce68ed026d694621f6fdfd", "uniV3"),
    ),
}

CHAIN_ROUTERS: dict[str, tuple[str, ...]] = {
    "mainnet": (
        "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # UniswapV2Router02
        "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f",  # SushiV2 router
    ),
    "bsc": (
        "0x10ed43c718714eb63d5aa57b78b54704e256024e",  # PancakeV2 router
    ),
    "polygon": (
        "0xa5e0829caced8ffdd4de3c43696c57f7d7a678ff",  # QuickSwap router
    ),
    "arbitrum": (
        "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506",  # SushiV2 router
    ),
    "base": (
        "0xcf77a3ba9a5ca399b7c97c74d54e5b1beb874e43",  # Aerodrome router
    ),
}


_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{40}")


def extract_addresses(paths: list[Path]) -> set[str]:
    """Harvest 40-hex-nibble address literals from Solidity source files.

    Lowercased, deduped, zero/dead placeholders stripped. Over-inclusive by
    design — every matched literal goes through the probe plan, and spurious
    non-address matches just produce cached reverts (which are themselves
    cheap future hits).
    """
    found: set[str] = set()
    for p in paths:
        if not p.exists() or not p.is_file():
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        for match in _ADDRESS_RE.finditer(text):
            addr = match.group(0).lower()
            if addr in _PLACEHOLDER_ADDRESSES:
                continue
            found.add(addr)
    return found


@dataclass(frozen=True)
class Probe:
    """A single cast-style command to execute inside task-exec-service.

    `cmd` is a list of tokens — joined with single-quote shell escaping when
    batched via `sh -c '...'`. `tag` is a short label used only for logging.
    """
    cmd: tuple[str, ...]
    tag: str


def build_probe_plan(
    variation_dir: Path,
    toml_doc: dict[str, Any],
    defihacklabs_root: Path,
    anvil_url: str = "http://anvil-service:8545",
    log_window: int = 100,
    log_total_blocks: int = 500,
) -> list[Probe]:
    """Enumerate every read-only RPC probe we want to cache for a variation.

    Safe to run offline — no network calls happen here; this just plans the
    command set. run_probes() is what actually fires them.
    """
    chain = toml_doc["chain"]
    chain_name = chain["name"]
    fork_block = int(chain["fork_block"])
    block_hex = hex(fork_block)

    variation_meta = toml_doc["variation"]
    source_rel = variation_meta.get("source_exploit_relpath") or ""

    # Approach-D sections. Drain targets / gain tokens are mode-gated at
    # load time, so either (but not both) may be absent; both default to [].
    env_contracts = (
        toml_doc.get("environment", {}) or {}
    ).get("contracts", []) or []
    success = toml_doc.get("success", {}) or {}
    drain_targets = success.get("drain_targets", []) or []
    gain_tokens = success.get("gain_tokens", []) or []
    attacker = toml_doc.get("attacker", {}) or {}

    sol_paths: list[Path] = []
    if source_rel:
        sol_paths.append(defihacklabs_root / source_rel)
    sources_dir = variation_dir / "sources"
    if sources_dir.is_dir():
        sol_paths.extend(sorted(sources_dir.glob("*.sol")))

    declared: set[str] = set()
    declared_tokens: set[str] = set()
    # Inventory contracts (agent-known; may or may not ship source).
    for c in env_contracts:
        declared.add(c["address"].lower())
    # Drain-target addresses + their tracked tokens.
    drain_target_addrs: set[str] = set()
    for dt in drain_targets:
        addr = dt["address"].lower()
        drain_target_addrs.add(addr)
        declared.add(addr)
        for tok in dt.get("tokens", []) or []:
            tok_addr = tok["address"].lower()
            if tok_addr == ZERO_ADDRESS:
                continue
            declared.add(tok_addr)
            declared_tokens.add(tok_addr)
    # Gain-token addresses.
    for tok in gain_tokens:
        tok_addr = tok["address"].lower()
        if tok_addr == ZERO_ADDRESS:
            continue
        declared.add(tok_addr)
        declared_tokens.add(tok_addr)

    whale = (attacker.get("impersonate_whale") or "").strip().lower()
    if whale.startswith("0x") and len(whale) == 42:
        declared.add(whale)
    for tok_addr in (attacker.get("initial_tokens") or {}).keys():
        declared.add(tok_addr.lower())

    extracted = extract_addresses(sol_paths)
    all_addresses = (declared | extracted) - _PLACEHOLDER_ADDRESSES

    probes: list[Probe] = []

    def _add(cmd: list[str], tag: str) -> None:
        probes.append(Probe(cmd=tuple(cmd), tag=tag))

    # --- chain-wide probes ---
    _add(["cast", "chain-id", "--rpc-url", anvil_url], "eth_chainId")
    _add(["cast", "rpc", "net_version", "--rpc-url", anvil_url], "net_version")
    for offset in (-3, -2, -1, 0, 1, 2, 3):
        blk_hex = hex(fork_block + offset)
        _add(
            ["cast", "rpc", "eth_getBlockByNumber", blk_hex, "false",
             "--rpc-url", anvil_url],
            f"eth_getBlockByNumber({offset:+d})",
        )
    _add(
        ["cast", "rpc", "eth_getBlockByNumber", block_hex, "true",
         "--rpc-url", anvil_url],
        "eth_getBlockByNumber(fork,full)",
    )

    # --- per-address probes ---
    holders = (ZERO_ADDRESS, DEAD_ADDRESS, ATTACKER_EOA)
    for addr in sorted(all_addresses):
        _add(["cast", "code", addr, "--block", block_hex,
              "--rpc-url", anvil_url], "eth_getCode")
        _add(["cast", "balance", addr, "--block", block_hex,
              "--rpc-url", anvil_url], "eth_getBalance")
        for slot_label, slot in _PROXY_SLOTS:
            _add(["cast", "storage", addr, slot, "--block", block_hex,
                  "--rpc-url", anvil_url],
                 f"eth_getStorageAt:{slot_label}")
        # Admin / ownership introspection — reverts on contracts that don't
        # expose these sigs, which is fine: the reverted response is cached.
        for sig in _ADMIN_INTROSPECT_SIGS:
            _add(["cast", "call", addr, sig, "--block", block_hex,
                  "--rpc-url", anvil_url], f"call:{sig}")
        # ERC-20 metadata — fire unconditionally.
        for sig in _ERC20_METADATA_SIGS:
            _add(["cast", "call", addr, sig, "--block", block_hex,
                  "--rpc-url", anvil_url], f"call:{sig}")
        # balanceOf probes against canonical holders.
        for holder in holders:
            _add(["cast", "call", addr,
                  "balanceOf(address)(uint256)", holder,
                  "--block", block_hex, "--rpc-url", anvil_url],
                 "call:balanceOf")
        # AMM pair introspection.
        for sig in _PAIR_SIGS:
            _add(["cast", "call", addr, sig, "--block", block_hex,
                  "--rpc-url", anvil_url], f"call:{sig}")
        # ERC-4626 vault introspection.
        for sig in _VAULT_SIGS:
            _add(["cast", "call", addr, sig, "--block", block_hex,
                  "--rpc-url", anvil_url], f"call:{sig}")

    # --- balanceOf(drain_target) for every declared (drain_target, token)
    # pair. For gain-mode variations drain_targets is empty so this is
    # a no-op — the per-token balanceOf probes against canonical holders
    # above already cover the attacker-side balances that matter.
    for dt in drain_targets:
        drain_addr = dt["address"].lower()
        dt_tokens = {
            tok["address"].lower()
            for tok in (dt.get("tokens", []) or [])
            if tok["address"].lower() != ZERO_ADDRESS
        }
        for tok_addr in dt_tokens:
            _add(["cast", "call", tok_addr,
                  "balanceOf(address)(uint256)", drain_addr,
                  "--block", block_hex, "--rpc-url", anvil_url],
                 "call:balanceOf(drain_target)")

    # --- allowance(drain_target, router) for canonical per-chain routers ---
    for router in CHAIN_ROUTERS.get(chain_name, ()):
        for drain_addr in sorted(drain_target_addrs):
            for tok_addr in declared_tokens:
                _add(["cast", "call", tok_addr,
                      "allowance(address,address)(uint256)",
                      drain_addr, router,
                      "--block", block_hex, "--rpc-url", anvil_url],
                     "call:allowance")

    # --- AMM factory getPair for every declared token pair ---
    factories = CHAIN_FACTORIES.get(chain_name, ())
    tokens_list = sorted(declared_tokens)
    for factory in factories:
        if factory.kind != "uniV2":
            continue
        for i, ta in enumerate(tokens_list):
            for tb in tokens_list[i + 1:]:
                _add(["cast", "call", factory.address,
                      "getPair(address,address)(address)", ta, tb,
                      "--block", block_hex, "--rpc-url", anvil_url],
                     f"call:getPair@{factory.label}")

    # --- eth_getLogs windows for every drain_target and declared token ---
    log_targets = drain_target_addrs | declared_tokens
    for target in sorted(log_targets):
        _append_log_probes(
            probes, target, fork_block,
            anvil_url=anvil_url,
            window=log_window,
            total=log_total_blocks,
        )

    return probes


def _append_log_probes(
    out: list[Probe],
    address: str,
    fork_block: int,
    *,
    anvil_url: str,
    window: int,
    total: int,
) -> None:
    """Cover a ``total``-block history in ``window``-sized numeric ranges.

    The rpc-cache requires both fromBlock and toBlock to be numeric for the
    response to persist (``rpc-cache-src/main.go:numericLogRange``). Upstream
    endpoints (Alchemy) typically cap eth_getLogs at a few hundred blocks
    per call — 100 is safely under every endpoint we use.
    """
    start = max(0, fork_block - total)
    cursor = start
    while cursor < fork_block:
        to_blk = min(cursor + window - 1, fork_block - 1)
        filter_json = (
            '{"address":"' + address + '",'
            '"fromBlock":"' + hex(cursor) + '",'
            '"toBlock":"' + hex(to_blk) + '"}'
        )
        out.append(Probe(
            cmd=("cast", "rpc", "eth_getLogs", filter_json,
                 "--rpc-url", anvil_url),
            tag="eth_getLogs",
        ))
        cursor = to_blk + 1


@dataclass
class ProbeStats:
    total: int = 0
    batches_ok: int = 0
    batches_failed: int = 0
    batches_timed_out: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "total_probes": self.total,
            "batches_ok": self.batches_ok,
            "batches_failed": self.batches_failed,
            "batches_timed_out": self.batches_timed_out,
        }


def run_probes(
    task_exec: Container,
    probes: list[Probe],
    *,
    batch_size: int = 25,
    timeout_per_batch_s: float = 180.0,
) -> ProbeStats:
    """Fire probes inside task-exec-service, batched via ``sh -c`` chains.

    Individual probe failures (reverts, bad addresses) are absorbed by
    ``|| true`` — what matters is that the request reached anvil and its
    response landed in the rpc-cache sqlite. Batch-level failures (shell
    missing, container killed) are counted separately.
    """
    stats = ProbeStats(total=len(probes))
    if not probes:
        return stats

    for i in range(0, len(probes), batch_size):
        batch = probes[i : i + batch_size]
        script = " ; ".join(
            " ".join(_shell_quote(tok) for tok in p.cmd)
            + " >/dev/null 2>&1 || true"
            for p in batch
        )
        try:
            raw = exec_run_with_timeout(
                task_exec,
                ["sh", "-c", script],
                timeout_s=timeout_per_batch_s,
            )
            exit_code, _out = coerce_exec_result(raw)
            if exit_code == 0:
                stats.batches_ok += 1
            else:
                stats.batches_failed += 1
                logger.warning(
                    f"probe batch {i}..{i + len(batch)} exit={exit_code}"
                )
        except Exception as exc:  # noqa: BLE001
            stats.batches_timed_out += 1
            logger.warning(
                f"probe batch {i}..{i + len(batch)} timed out / errored: {exc}"
            )
    return stats


_SAFE_SHELL_CHARS = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-_:/.=+@,"
)


def _shell_quote(token: str) -> str:
    """Single-quote a shell token unless it's made of known-safe chars only."""
    if token and all(c in _SAFE_SHELL_CHARS for c in token):
        return token
    return "'" + token.replace("'", "'\\''") + "'"
