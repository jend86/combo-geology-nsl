"""Smoke test: populate + verify + synthetic-drain every non-blocked variation.

Requires:
- docker compose up in docker/forked-exploit-compose (all five services healthy)
- compose-local `.env` populated with RPC_URL_* keys before `docker compose up`
- Each variation's rpc_cache/ either present or the RPC will be hit live

Reports per-variation: populate success, verify success, synthetic-drain
reward, agent-side cheat-block confirmation.

Usage:
    uv run python scripts/smoke_fork_all.py
    uv run python scripts/smoke_fork_all.py --only pdz_flashloan_price_manip_2025_08
"""

from __future__ import annotations

import argparse
import json
import sys

import docker
from loguru import logger

from src.task.types import CapabilityExecutionContext
from tasks.forked_exploit import (
    ATTACKER_ADDRESS,
    ForkedExploitTask,
    ForkedVariation,
)


def _get_slot_containers(client: docker.DockerClient) -> list:
    # Look for containers from our compose project (name starts with
    # "forked-exploit-compose-"). Return the five services.
    wanted = (
        "anvil-service",
        "rpc-cache-service",
        "proxy-service",
        "task-exec-service",
        "agent-service",
    )
    containers = []
    for c in client.containers.list():
        name = c.name or ""
        if "forked-exploit-compose" in name and any(w in name for w in wanted):
            containers.append(c)
    return containers


def _compose_service_name(container) -> str:
    return container.labels.get("com.docker.compose.service", "")


def _service_container(containers: list, service_name: str):
    for container in containers:
        if _compose_service_name(container) == service_name:
            return container
    raise RuntimeError(f"missing compose service: {service_name}")


def _is_healthy(container) -> tuple[bool, str]:
    container.reload()
    health = container.attrs.get("State", {}).get("Health", {})
    status = health.get("Status", "")
    return status in {"", "healthy"}, status or "none"


def verify_rpc_cache_service_healthy(containers: list) -> tuple[bool, str]:
    rpc_cache = _service_container(containers, "rpc-cache-service")
    healthy, status = _is_healthy(rpc_cache)
    return healthy, f"rpc-cache-service health={status}"


def verify_rpc_cache_wiring(
    task: ForkedExploitTask, containers: list
) -> tuple[bool, str]:
    task_exec = task._pick_container(containers, "task_exec")  # noqa: SLF001
    exit_code, out = task._exec(  # noqa: SLF001
        task_exec,
        [
            "sh",
            "-c",
            "test -f /shared/anvil/config.env && "
            "grep -q '^FORK_URL=http://rpc-cache-service:8546$' /shared/anvil/config.env",
        ],
    )
    if exit_code != 0:
        return (
            False,
            out.decode(errors="replace")[:200]
            or "anvil FORK_URL not routed via rpc-cache",
        )
    return True, ""


def synthetic_drain(
    task: ForkedExploitTask, containers: list, variation: ForkedVariation
) -> bool:
    """Impersonate the first drain_target and transfer a small amount of the
    first tracked token to the attacker. Measurement grid should pick up the
    drain.

    Skipped for gain-only variations (no drain_targets).
    """
    if not variation.drain_targets or not variation.drain_targets[0].tokens:
        return False
    target = variation.drain_targets[0]
    tok = target.tokens[0]
    task_exec = task._pick_container(containers, "task_exec")  # noqa: SLF001

    # Small drain: 1% of the target's balance (fall back to 1 if balance is 0).
    bal = (
        task._native_balance(task_exec, target.address)
        if tok.is_native
        else task._erc20_balance_of(task_exec, tok.address, target.address)
    )
    amount = max(1, bal // 100)

    # Impersonate target, transfer, stop-impersonate. Task-side RPC, unfiltered.
    task_rpc = "http://anvil-service:8545"
    task._exec(
        task_exec,
        [
            "sh",
            "-c",
            f"cast rpc anvil_impersonateAccount {target.address} --rpc-url {task_rpc}",
        ],
    )
    # Impersonated account needs ETH for gas (real tx flow).
    task._exec(
        task_exec,
        [
            "sh",
            "-c",
            f"cast rpc anvil_setBalance {target.address} 0xde0b6b3a7640000 --rpc-url {task_rpc}",
        ],
    )
    if tok.is_native:
        task._exec(
            task_exec,
            [
                "sh",
                "-c",
                f"cast send {ATTACKER_ADDRESS} --value {amount} "
                f"--unlocked --from {target.address} --rpc-url {task_rpc}",
            ],
            timeout_s=30,
        )
    else:
        task._exec(
            task_exec,
            [
                "sh",
                "-c",
                f"cast send {tok.address} 'transfer(address,uint256)' "
                f"{ATTACKER_ADDRESS} {amount} "
                f"--unlocked --from {target.address} --rpc-url {task_rpc}",
            ],
            timeout_s=30,
        )
    task._exec(
        task_exec,
        [
            "sh",
            "-c",
            f"cast rpc anvil_stopImpersonatingAccount {target.address} --rpc-url {task_rpc}",
        ],
    )
    return True


def verify_agent_cheat_block(task: ForkedExploitTask, containers: list) -> bool:
    """Ensure the agent-side proxy rejects anvil_setBalance."""
    agent = task._pick_container(containers, "agent")  # noqa: SLF001
    exit_code, out = task._exec(
        agent,
        [  # noqa: SLF001
            "sh",
            "-c",
            f"cast rpc anvil_setBalance {ATTACKER_ADDRESS} 0x1 --rpc-url $RPC_URL 2>&1 "
            f"|| true",  # cast returns non-zero on RPC error; we want the message
        ],
    )
    text = out.decode(errors="replace").lower()
    return "method not found" in text or "-32601" in text


# A no-op Attack contract. Compiles cleanly, deploys cleanly, and its
# attack() is a pure no-op. We wrap it in the ```solidity fence the task's
# _ATTACK_SOL_RE expects so execute_mode() treats it as a real agent output.
#
# This is the regression guard for two agent-side failure modes: anvil signer
# chain-id mismatch on forked chains, and proxy rejection of raw signed
# transactions. If either regression returns, the forge create step fails with
# a chain-id error or the proxy rejects the signed tx; either way,
# execute_mode returns success=False here.
_NOOP_ATTACK_RAW_RESPONSE = """\
Results: no-op regression check.

```solidity
// SPDX-License-Identifier: UNLICENSED
pragma solidity {solc_version};

contract Attack {{
    function attack() external {{}}
    receive() external payable {{}}
    fallback() external payable {{}}
}}
```
"""


def attack_lifecycle_smoke(
    task: ForkedExploitTask,
    containers: list,
    variation: ForkedVariation,
) -> tuple[bool, str]:
    """Run a no-op Attack.sol through execute_mode("exploiter", ...) to exercise
    the full agent-side forge build → forge create → cast send path, exactly
    as the training loop would. Returns (success, diagnostic).

    Catches signer chain-id mismatch and proxy-allowlist regressions that the
    populate → synthetic_drain path never touches (the synthetic drain
    impersonates a victim via task-exec, bypassing the agent signing pipeline
    entirely)."""
    raw = _NOOP_ATTACK_RAW_RESPONSE.format(solc_version=variation.solc_version)
    invs = task.parse_response(raw, invoked_capability="exploiter")
    if not invs:
        return False, "parse_response returned no invocations (regex miss?)"
    result = task.execute_capability(
        invs[0], containers, variation, CapabilityExecutionContext("smoke", None, {})
    )
    if result is None or not result.success:
        tail = (result.output.get("stderr") if result else "") or ""
        return False, tail[-800:]
    return True, str(result.output.get("stdout", ""))[:200]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="Single variation name to test")
    args = ap.parse_args()

    client = docker.from_env()
    containers = _get_slot_containers(client)
    if len(containers) < 5:
        print(
            f"Expected 5 compose services running; got {len(containers)}. "
            f"Run `docker compose up -d --build` in docker/forked-exploit-compose/",
            file=sys.stderr,
        )
        return 2

    task_cfg: dict = {
        "defihacklabs_root": "/home/elijah/DeFiHackLabs",
    }
    task = ForkedExploitTask(task_cfg)

    variations = [v for v in task.list_variations() if isinstance(v, ForkedVariation)]
    if args.only:
        variations = [v for v in variations if v.name == args.only]
    if not variations:
        print("No matching live variations", file=sys.stderr)
        return 2

    summary: list[dict] = []
    for variation in variations:
        logger.info(
            f"--- {variation.name} ({variation.chain_name}:{variation.fork_block}) ---"
        )
        row: dict = {"variation": variation.name}
        try:
            task.reset(containers)
            rpc_cache_ok, rpc_cache_msg = verify_rpc_cache_service_healthy(containers)
            row["rpc_cache_service_healthy"] = rpc_cache_ok
            if not rpc_cache_ok:
                raise RuntimeError(rpc_cache_msg)
            outcome = task.populate(containers, variation)
            row["populate"] = outcome.results[0].success
            wiring_ok, wiring_msg = verify_rpc_cache_wiring(task, containers)
            row["rpc_cache_wiring"] = wiring_ok
            if not wiring_ok:
                raise RuntimeError(wiring_msg)
            ok = task.verify_population(
                containers,
                variation,
                outcome.episode_context,
                private_context=outcome.private_context,
            )
            row["verify"] = ok
            initial = task.measure_initial_state(
                containers,
                outcome.episode_context,
                private_context=outcome.private_context,
            )
            synthetic_drain(task, containers, variation)
            from src.task.types import EpisodeArtifacts

            final = task.measure_final_state(
                containers,
                outcome.episode_context,
                EpisodeArtifacts(final_response=None),
                private_context=outcome.private_context,
            )
            reward = task._compute_reward_with_variation(variation, initial, final)  # noqa: SLF001
            row["synthetic_drain_reward"] = reward.value
            row["synthetic_drain_success"] = reward.success
            row["agent_cheat_blocked"] = verify_agent_cheat_block(task, containers)
            # After the synthetic drain measurements, exercise the agent
            # signing path end-to-end with a no-op Attack.sol. This guards
            # against signer chain-id mismatch and allowlist regressions.
            attack_ok, attack_msg = attack_lifecycle_smoke(task, containers, variation)
            row["attack_lifecycle"] = attack_ok
            if not attack_ok:
                row["attack_lifecycle_error"] = attack_msg
        except Exception as exc:
            row["error"] = str(exc)[:300]
            logger.exception(f"{variation.name} failed")
        summary.append(row)

    print(json.dumps(summary, indent=2, default=str))
    fails = [
        r
        for r in summary
        if "error" in r
        or not r.get("rpc_cache_service_healthy")
        or not r.get("rpc_cache_wiring")
        or not r.get("populate")
        or not r.get("verify")
        or r.get("attack_lifecycle") is False
    ]
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
