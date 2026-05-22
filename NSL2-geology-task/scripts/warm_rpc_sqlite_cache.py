"""Pre-warm the shared rpc-cache sqlite by running each variation's populate().

Complements `scripts/warm_rpc_cache.py`, which warms Foundry's per-block JSON
cache baked into the anvil image. That cache covers only state the original
DeFiHackLabs PoC read. This script warms the runtime side — the sqlite at
`$RPC_CACHE_HOST_DIR` (see docker/forked-exploit-compose/.env) — with:

  1. Scorer prep (``task.populate()``): chain/block probes + victim-balance
     grid for every tracked (victim, token) pair.
  2. Attacker-view probes (``scripts/lib/attacker_probes.py``): the broader
     set of read-only calls an exploiter agent plausibly makes while
     investigating a new task — eth_getCode, EIP-1967 storage slots,
     ERC-20 metadata, AMM pair state, eth_getLogs windows, factory
     getPair lookups.

With the shared bind mount, one warmer run populates the sqlite that all
subsequent training-run slots share, so the first real training run never
pays Alchemy for these reads.

Usage:
    uv run python scripts/warm_rpc_sqlite_cache.py
    uv run python scripts/warm_rpc_sqlite_cache.py --only pdz_…
    uv run python scripts/warm_rpc_sqlite_cache.py --skip-probes
    uv run python scripts/warm_rpc_sqlite_cache.py --probe-log /tmp/warm.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import docker
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from scripts.lib.attacker_probes import Probe, build_probe_plan, run_probes
from src.container import compose_services, container_to_service
from tasks.forked_exploit import ForkedExploitTask

COMPOSE_DIR = PROJECT_ROOT / "docker" / "forked-exploit-compose"
COMPOSE_FILE = COMPOSE_DIR / "docker-compose.yml"
PROJECT_NAME = "nsl-rpc-cache-warmer"
VARIATIONS_ROOT = PROJECT_ROOT / "tasks" / "forked_exploit_variations"
DEFAULT_DEFIHACKLABS_ROOT = Path.home() / "DeFiHackLabs"


def _compose(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [
        "docker",
        "compose",
        "-p",
        PROJECT_NAME,
        "-f",
        str(COMPOSE_FILE),
        *args,
    ]
    return subprocess.run(cmd, check=check, text=True)


def _get_project_containers() -> list:
    client = docker.from_env()
    return client.containers.list(
        filters={"label": f"com.docker.compose.project={PROJECT_NAME}"}
    )


def _pick_task_exec(containers: list) -> object | None:
    for c in containers:
        if container_to_service(c) == "task-exec-service":
            return c
    return None


def _load_toml_doc(variation_name: str) -> dict:
    toml_path = VARIATIONS_ROOT / variation_name / "variation.toml"
    with toml_path.open("rb") as f:
        return tomllib.load(f)


def warm_all(
    *,
    keep_running: bool,
    skip_probes: bool,
    only: list[str] | None,
    probe_log: Path | None,
    defihacklabs_root: Path,
) -> None:
    task = ForkedExploitTask(task_config={})
    variations = task.list_variations()
    if only:
        filter_set = set(only)
        variations = [v for v in variations if v.name in filter_set]
        missing = filter_set - {v.name for v in variations}
        if missing:
            logger.warning(f"--only filter did not match: {sorted(missing)}")
    logger.info(
        f"Warming {len(variations)} variations via shared rpc-cache sqlite "
        f"(probes={'off' if skip_probes else 'on'})"
    )

    services = compose_services(COMPOSE_FILE, project_name=PROJECT_NAME)
    logger.info(f"Compose services: {services}")

    logger.info("Bringing up compose stack (compose waits for healthchecks)")
    _compose("up", "-d", "--build", "--wait")
    probe_log_fp = probe_log.open("w") if probe_log else None
    try:
        containers = _get_project_containers()
        logger.info(f"Containers ready: {[c.name for c in containers]}")
        task_exec = _pick_task_exec(containers)
        if task_exec is None and not skip_probes:
            raise RuntimeError(
                "task-exec-service container not found; cannot run probes."
            )

        successes, failures = 0, []
        total_probes_fired = 0
        for i, variation in enumerate(variations, start=1):
            logger.info(f"[{i}/{len(variations)}] Warming {variation.name}")
            try:
                task.populate(containers, variation)
                successes += 1
            except Exception as exc:
                logger.warning(f"  {variation.name}: populate failed: {exc}")
                failures.append((variation.name, str(exc)[:200]))
                continue  # no point probing if anvil isn't forked

            if skip_probes:
                continue

            try:
                toml_doc = _load_toml_doc(variation.name)
                probes: list[Probe] = build_probe_plan(
                    variation_dir=VARIATIONS_ROOT / variation.name,
                    toml_doc=toml_doc,
                    defihacklabs_root=defihacklabs_root,
                )
                logger.info(
                    f"  probes planned: {len(probes)} "
                    f"(chain={toml_doc['chain']['name']}, "
                    f"fork_block={toml_doc['chain']['fork_block']})"
                )
                stats = run_probes(task_exec, probes)
                total_probes_fired += stats.total
                logger.info(f"  probes: {stats.as_dict()}")
                if probe_log_fp is not None:
                    probe_log_fp.write(
                        json.dumps(
                            {
                                "variation": variation.name,
                                **stats.as_dict(),
                            }
                        )
                        + "\n"
                    )
                    probe_log_fp.flush()
            except Exception as exc:
                logger.warning(f"  {variation.name}: probe phase failed: {exc}")
                failures.append((variation.name, f"probe: {exc!s:.200s}"))

        logger.info(
            f"Done: {successes}/{len(variations)} populated, "
            f"{len(failures)} failed; {total_probes_fired} probes fired"
        )
        for name, msg in failures:
            logger.info(f"  FAILED {name}: {msg}")
    finally:
        if probe_log_fp is not None:
            probe_log_fp.close()
        if keep_running:
            logger.info(
                f"Leaving stack up (project={PROJECT_NAME}). "
                f"Run `docker compose -p {PROJECT_NAME} -f {COMPOSE_FILE} down` to stop."
            )
        else:
            logger.info("Tearing down compose stack")
            _compose("down", "--remove-orphans", check=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--keep-running",
        action="store_true",
        help="Leave the compose stack up after warming (useful for debugging).",
    )
    ap.add_argument(
        "--skip-probes",
        action="store_true",
        help="Only run task.populate(); skip the attacker-view probe phase.",
    )
    ap.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="VARIATION",
        help="Restrict warming to the named variation(s). Repeat for multiple.",
    )
    ap.add_argument(
        "--probe-log",
        type=Path,
        default=None,
        help="Optional JSONL sink for per-variation probe stats.",
    )
    ap.add_argument(
        "--defihacklabs-root",
        type=Path,
        default=DEFAULT_DEFIHACKLABS_ROOT,
        help=f"Path to DeFiHackLabs repo (default: {DEFAULT_DEFIHACKLABS_ROOT}). "
        "Used for address extraction from source_exploit_relpath.",
    )
    args = ap.parse_args()
    warm_all(
        keep_running=args.keep_running,
        skip_probes=args.skip_probes,
        only=args.only,
        probe_log=args.probe_log,
        defihacklabs_root=args.defihacklabs_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
