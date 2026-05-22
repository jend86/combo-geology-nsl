"""Memory Cleanup Task — reference implementation.

Agents operate inside Alpine Linux containers and attempt to free
disk space by identifying and removing unnecessary files.

Extracted from the original hardcoded task logic in:
- src/container.py (variations, reset, populate, verify)
- src/mode_prompts.py (prompts)
- src/mode_parsers.py (mode output parsing)
- src/tool/docker.py (measurement helpers)
- scripts/run_episode.py (pre/post measurement, reward)
"""

import os
import re
import shlex
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from docker.models.containers import Container
from loguru import logger

from src.task.base import TaskEnvironmentError, TaskSpec
from src.task.types import (
    BudgetConstraints,
    Capability,
    CapabilityExecutionContext,
    CapabilityInvocation,
    CapabilityResult,
    EpisodeArtifacts,
    EpisodeConstraints,
    PopulationOutcome,
    PopulationResult,
    StepConstraints,
    SuccessConstraints,
    TaskPromptSpec,
    TaskReward,
    Variation,
    Workflow,
    WorkflowStep,
)
from tasks.common.agent_exec import run_python_in_agent


# ---------------------------------------------------------------------------
# Task-specific data types
# ---------------------------------------------------------------------------


@dataclass
class MemoryCleanupVariation(Variation):
    """Cleanup-specific variation carrying the shell commands and expected KB.

    Fields:
        commands: dd commands to populate the container (mkdir excluded).
        expected_kb: calibrated for du -sk block-aligned output (4KB blocks).
        mkdir_cmd: explicit mkdir command run before dd commands.
        created_dirs: directories created by mkdir_cmd (verification targets).
        tolerance_override: if set, overrides global tolerance for this variation.
    """

    commands: list[str] = field(default_factory=list)
    expected_kb: float = 0.0
    mkdir_cmd: str = ""
    created_dirs: list[str] = field(default_factory=list)
    tolerance_override: float | None = None


@dataclass
class MemoryCleanupState:
    """Typed episode state for memory-cleanup (the `StateT` parameter).

    Used symmetrically: both measure_initial_state and measure_final_state
    return this type.
    """

    used_kb: dict[str, float]  # container_id -> used KB in cleanup paths (du -sk)
    filesystem_groups: dict[str, list[str]]  # fs_id -> list of container_ids


# ---------------------------------------------------------------------------
# Docker exec timeout helper (absorbed from container.py)
# ---------------------------------------------------------------------------

_EXEC_TIMEOUT_S = 120


def _exec_run_with_timeout(
    container: Container,
    cmd: Any,
    timeout_s: float = _EXEC_TIMEOUT_S,
    **kwargs: Any,
) -> Any:
    """Run container.exec_run with a timeout."""
    result: list[Any] = [None]
    exception: list[BaseException | None] = [None]

    def _run() -> None:
        try:
            result[0] = container.exec_run(cmd, **kwargs)
        except BaseException as exc:
            exception[0] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        raise TaskEnvironmentError(
            f"Docker exec timed out after {timeout_s}s on "
            f"{container.name or container.id}: {cmd}"
        )
    if exception[0] is not None:
        raise exception[0]
    return result[0]


def _coerce_exec_result(exec_result: Any) -> tuple[int, bytes]:
    if hasattr(exec_result, "exit_code") and hasattr(exec_result, "output"):
        exit_code = exec_result.exit_code if exec_result.exit_code is not None else -1
        output = exec_result.output if exec_result.output is not None else b""
        return int(exit_code), bytes(output)
    if isinstance(exec_result, tuple) and len(exec_result) >= 2:
        exit_code = exec_result[0] if exec_result[0] is not None else -1
        output = exec_result[1] if exec_result[1] is not None else b""
        return int(exit_code), bytes(output)
    raise TypeError(f"Unsupported exec result type: {type(exec_result)!r}")


# ---------------------------------------------------------------------------
# Task implementation
# ---------------------------------------------------------------------------


class MemoryCleanupTask(TaskSpec[MemoryCleanupState]):
    """Clear disk space in Linux containers.

    Agents operate inside Alpine Linux containers and attempt to free
    disk space by identifying and removing unnecessary files.

    Recommended action budget: 12 (default).
    """

    name = "memory-cleanup"
    description = "Free disk space in Linux containers by removing unnecessary files"
    metric_name = "space_freed_kb"
    metric_unit = "KB"
    higher_is_better = True
    agent_service_name = "service-a"

    def __init__(self, task_config: dict[str, Any]) -> None:
        self._docker_compose_dir = task_config.get(
            "docker_compose_dir", "docker/special-learn-compose"
        )
        self._tolerance = task_config.get("tolerance", 0.10)
        self._cleanup_paths = task_config.get(
            "cleanup_paths",
            ["/tmp", "/var/log", "/var/cache", "/var/tmp", "/home/alice"],
        )
        self._success_threshold_kb = task_config.get("success_threshold_kb", 0.0)

    @property
    def docker_compose_dir(self) -> str:
        return self._docker_compose_dir

    def episode_constraints(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> EpisodeConstraints:
        return EpisodeConstraints(
            budgets=BudgetConstraints(
                max_task_tool_calls=15,
                max_llm_turns=18,
            ),
            success=SuccessConstraints(min_task_tool_calls_for_success=1),
            step_overrides={
                "plan_cleanup": StepConstraints(
                    success=SuccessConstraints(min_task_tool_calls_for_success=0),
                ),
            },
        )

    # --- Variations ---

    def list_variations(self) -> list[Variation]:
        """Return 5 dd-based filesystem variations.

        Each variation has:
        - mkdir_cmd: run first to create target directories
        - commands: dd commands only (mkdir excluded)
        - expected_kb: calibrated for du -sk block-aligned output (4KB blocks)
        - created_dirs: directories created by mkdir_cmd
        - tolerance_override: per-variation tolerance (V5 only)
        """
        return [
            MemoryCleanupVariation(
                name="variation_1_heavy",
                description="Heavy mix - lots of large files across all categories",
                mkdir_cmd="mkdir -p /tmp/cleanup /tmp/sessions /var/log/app /var/cache/pkg /tmp/db_temp /home/alice/trash /var/tmp/build",
                created_dirs=[
                    "/tmp/cleanup",
                    "/var/log/app",
                    "/var/cache/pkg",
                    "/tmp/db_temp",
                    "/home/alice/trash",
                    "/var/tmp/build",
                ],
                commands=[
                    "for i in $(seq 1 40); do dd if=/dev/zero of=/tmp/cleanup/temp_$i.tmp bs=100K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 30); do dd if=/dev/zero of=/var/log/app/app_$i.log bs=100K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 25); do dd if=/dev/zero of=/var/cache/pkg/cache_$i.dat bs=100K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 15); do dd if=/dev/zero of=/tmp/db_temp/table_$i.tmp bs=200K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 20); do dd if=/dev/zero of=/home/alice/trash/deleted_$i.bak bs=50K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 10); do dd if=/dev/zero of=/var/tmp/build/obj_$i.o bs=200K count=1 conv=fdatasync status=none; done",
                ],
                # 40×100 + 30×100 + 25×100 + 15×200 + 20×52 + 10×200 = 15540
                # (50K → ceil(51200/4096)=13 blocks = 52KB)
                expected_kb=15540,
            ),
            MemoryCleanupVariation(
                name="variation_2_medium",
                description="Medium mix - balanced file sizes and counts",
                mkdir_cmd="mkdir -p /tmp/work /var/log/system /var/cache/app /tmp/downloads /home/alice/.cache /var/tmp/sql",
                created_dirs=[
                    "/tmp/work",
                    "/var/log/system",
                    "/var/cache/app",
                    "/tmp/downloads",
                    "/home/alice/.cache",
                    "/var/tmp/sql",
                ],
                commands=[
                    "for i in $(seq 1 50); do dd if=/dev/zero of=/tmp/work/work_$i.tmp bs=50K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 40); do dd if=/dev/zero of=/var/log/system/sys_$i.log bs=50K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 35); do dd if=/dev/zero of=/var/cache/app/app_$i.cache bs=75K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 15); do dd if=/dev/zero of=/tmp/downloads/download_$i.part bs=150K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 30); do dd if=/dev/zero of=/home/alice/.cache/thumb_$i.png bs=20K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 20); do dd if=/dev/zero of=/var/tmp/sql/query_$i.tmp bs=75K count=1 conv=fdatasync status=none; done",
                ],
                # 50×52 + 40×52 + 35×76 + 15×152 + 30×20 + 20×76 = 11740
                # (50K→52, 75K→76, 150K→152, 20K→20 exact)
                expected_kb=11740,
            ),
            MemoryCleanupVariation(
                name="variation_3_many_small",
                description="Many small files - tests handling of numerous small files",
                mkdir_cmd="mkdir -p /tmp/fragments /var/log/debug /var/cache/thumbnails /tmp/sessions /home/alice/temp /var/tmp/locks",
                created_dirs=[
                    "/tmp/fragments",
                    "/var/log/debug",
                    "/var/cache/thumbnails",
                    "/tmp/sessions",
                    "/home/alice/temp",
                    "/var/tmp/locks",
                ],
                commands=[
                    "for i in $(seq 1 200); do dd if=/dev/zero of=/tmp/fragments/frag_$i.tmp bs=10K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 100); do dd if=/dev/zero of=/var/log/debug/debug_$i.log bs=20K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 150); do dd if=/dev/zero of=/var/cache/thumbnails/thumb_$i.jpg bs=10K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 80); do dd if=/dev/zero of=/tmp/sessions/sess_$i.lock bs=5K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 60); do dd if=/dev/zero of=/home/alice/temp/temp_$i.dat bs=30K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 100); do dd if=/dev/zero of=/var/tmp/locks/lock_$i.pid bs=5K count=1 conv=fdatasync status=none; done",
                ],
                # 200×12 + 100×20 + 150×12 + 80×8 + 60×32 + 100×8 = 9560
                # (10K→12, 20K→20 exact, 5K→8, 30K→32)
                expected_kb=9560,
            ),
            MemoryCleanupVariation(
                name="variation_4_large_sparse",
                description="Large sparse files - few but large files",
                mkdir_cmd="mkdir -p /tmp/backups /var/log/archives /var/cache/packages /tmp/exports /home/alice/old /var/tmp/dumps",
                created_dirs=[
                    "/tmp/backups",
                    "/var/log/archives",
                    "/var/cache/packages",
                    "/tmp/exports",
                    "/home/alice/old",
                    "/var/tmp/dumps",
                ],
                commands=[
                    "for i in $(seq 1 5); do dd if=/dev/zero of=/tmp/backups/backup_$i.tar bs=1M count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 8); do dd if=/dev/zero of=/var/log/archives/archive_$i.log.gz bs=500K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 10); do dd if=/dev/zero of=/var/cache/packages/pkg_$i.deb bs=300K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 6); do dd if=/dev/zero of=/tmp/exports/export_$i.sql bs=500K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 12); do dd if=/dev/zero of=/home/alice/old/old_$i.bak bs=200K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 4); do dd if=/dev/zero of=/var/tmp/dumps/dump_$i.sql bs=500K count=1 conv=fdatasync status=none; done",
                ],
                # 5×1024 + 8×500 + 10×300 + 6×500 + 12×200 + 4×500 = 19520
                # (all sizes 4KB-aligned, zero block overhead)
                expected_kb=19520,
            ),
            MemoryCleanupVariation(
                name="variation_5_mixed_realistic",
                description="Realistic mixed scenario - mimics real system",
                mkdir_cmd="mkdir -p /tmp/app_temp /var/log/nginx /var/cache/apt /tmp/.build /home/alice/.local/share/Trash /var/tmp/mysql",
                created_dirs=[
                    "/tmp/app_temp",
                    "/var/log/nginx",
                    "/var/cache/apt",
                    "/tmp/.build",
                    "/home/alice/.local/share/Trash",
                    "/var/tmp/mysql",
                ],
                commands=[
                    "for i in $(seq 1 25); do size=$((RANDOM % 200 + 50)); dd if=/dev/zero of=/tmp/app_temp/temp_$i.tmp bs=${size}K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 20); do dd if=/dev/zero of=/var/log/nginx/access_$i.log bs=150K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 15); do dd if=/dev/zero of=/var/cache/apt/pkg_$i.deb bs=300K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 30); do dd if=/dev/zero of=/tmp/.build/obj_$i.o bs=75K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 18); do dd if=/dev/zero of=/home/alice/.local/share/Trash/file_$i.old bs=100K count=1 conv=fdatasync status=none; done",
                    "for i in $(seq 1 10); do dd if=/dev/zero of=/var/tmp/mysql/tmp_table_$i.ibd bs=200K count=1 conv=fdatasync status=none; done",
                ],
                # Deterministic part: 20×152 + 15×300 + 30×76 + 18×100 + 10×200 = 13620
                # RANDOM part: 25 files × [50-249]K → range [14920, 19920], mean ≈ 17400
                expected_kb=17400,
                tolerance_override=0.25,
            ),
        ]

    # --- Prompts ---

    def prompt_spec(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> TaskPromptSpec:
        return TaskPromptSpec(
            system_instruction=(
                "You are a Linux storage administrator working inside a "
                "networked container. Your goal is to free non-volatile "
                "storage space by removing unnecessary files."
            ),
            capabilities=[
                Capability(
                    name="run_python",
                    description=(
                        "Execute a Python script inside the agent container. "
                        "Use Python's standard library + os/shutil/subprocess "
                        "to inspect or clean the filesystem. Returns stdout, "
                        "stderr, and exit code."
                    ),
                    runs_code=True,
                    schema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python source to execute.",
                            },
                            "timeout_s": {
                                "type": "integer",
                                "description": (
                                    "Wall-clock timeout (seconds). Default 60."
                                ),
                            },
                        },
                        "required": ["code"],
                    },
                ),
            ],
        )

    def workflow(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> Workflow | None:
        cleanup_paths = ", ".join(self._cleanup_paths)
        return Workflow(
            steps=(
                WorkflowStep(
                    name="explore_container",
                    description="Scan the filesystem to understand disk usage",
                    prompt=(
                        "Explore the container filesystem to understand current disk usage. "
                        f"Inspect the cleanup paths ({cleanup_paths}) "
                        "and identify the largest files and directories, their sizes, "
                        "and what appears safe to remove. "
                        "Build a complete picture of space usage before making any changes. "
                        "Do not delete anything yet."
                    ),
                    inherit_all_capabilities=False,
                    capabilities=("run_python",),
                    is_entry=True,
                    next_steps=("plan_cleanup",),
                    context_mode="inherit",
                ),
                WorkflowStep(
                    name="plan_cleanup",
                    description="Reason through the cleanup strategy without executing",
                    prompt=(
                        "Based on your exploration findings, plan your cleanup strategy. "
                        "Reason through: which files and directories to remove, in what order, "
                        "what the expected space savings are, and any risks to avoid. "
                        "Do not use any tools — this is a reasoning step only. "
                        "Write out your complete plan before proceeding."
                    ),
                    inherit_all_capabilities=False,
                    capabilities=(),
                    next_steps=("execute_cleanup",),
                    context_mode="inherit",
                ),
                WorkflowStep(
                    name="execute_cleanup",
                    description="Execute the cleanup plan and verify results",
                    prompt=(
                        "Execute your cleanup plan by removing the files and "
                        "directories you identified as safe to delete. "
                        "After cleanup, verify the freed space by measuring "
                        "disk usage in the cleanup paths again."
                    ),
                    inherit_all_capabilities=False,
                    capabilities=("run_python",),
                    next_steps=(),
                    context_mode="inherit",
                ),
            )
        )

    # --- Response parsing ---

    def parse_response(
        self,
        raw_response: str,
        *,
        invoked_capability: str | None = None,
    ) -> list[CapabilityInvocation]:
        """Extract metric reports from the agent's response.

        ``invoked_capability`` here is a *harness phase tag* — set by
        :class:`OrchestratorModeHarness` to the mode name, not a
        ``Capability.name``. External harnesses may leave it ``None``; when
        the response self-identifies a metric report, we still surface it.
        """
        if invoked_capability not in (None, "explorer"):
            return []
        payload = self._parse_explorer_output(raw_response)
        if not payload:
            return []
        name = (
            "explorer_metric_report"
            if invoked_capability == "explorer"
            else "metric_report"
        )
        return [CapabilityInvocation(name=name, input=payload)]

    def execute_capability(
        self,
        invocation: CapabilityInvocation,
        containers: list[Container],
        variation: Variation,
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        if invocation.name in {"explorer_metric_report", "metric_report"}:
            return CapabilityResult(
                name=invocation.name,
                output=invocation.input,
                success=True,
            )
        if invocation.name != "run_python":
            return super().execute_capability(invocation, containers, variation, ctx)
        code = invocation.input.get("code")
        if not isinstance(code, str) or not code.strip():
            return CapabilityResult(
                name=invocation.name,
                output={},
                success=False,
                error="run_python requires a non-empty 'code' string",
            )
        timeout_s = int(invocation.input.get("timeout_s") or 60)
        if not containers:
            return CapabilityResult(
                name=invocation.name,
                output={},
                success=False,
                error="no agent container available",
            )
        out = run_python_in_agent(containers[0], code, timeout_s=timeout_s)
        return CapabilityResult(
            name=invocation.name,
            output=out,
            success=bool(out["success"]),
            error=None if out["success"] else (out.get("stderr") or None),
        )

    @staticmethod
    def _parse_explorer_output(raw_response: str) -> dict[str, Any]:
        """Extract space_freed_kb and operation classification."""
        space_freed_kb = 0.0
        matched_metric = False

        # Look for explicit KB mentions
        kb_match = re.search(r"(\d+\.?\d*)\s*KB", raw_response, re.IGNORECASE)
        if kb_match:
            space_freed_kb = float(kb_match.group(1))
            matched_metric = True
        else:
            # Look for MB mentions and convert to KB
            mb_match = re.search(r"(\d+\.?\d*)\s*MB", raw_response, re.IGNORECASE)
            if mb_match:
                space_freed_kb = float(mb_match.group(1)) * 1024
                matched_metric = True
            else:
                # Look for GB mentions and convert to KB
                gb_match = re.search(r"(\d+\.?\d*)\s*GB", raw_response, re.IGNORECASE)
                if gb_match:
                    space_freed_kb = float(gb_match.group(1)) * 1024 * 1024
                    matched_metric = True
                else:
                    freed_match = re.search(
                        r"freed\s+(\d+\.?\d*)", raw_response, re.IGNORECASE
                    )
                    if freed_match:
                        space_freed_kb = float(freed_match.group(1))
                        matched_metric = True

        if not matched_metric:
            return {}

        result: dict[str, Any] = {"space_freed_kb": space_freed_kb}

        # Classify operation type
        if re.search(r"\b(deleted|removed|cleaned)\b", raw_response, re.IGNORECASE):
            result["operation_type"] = "deletion"
        elif re.search(r"\b(compressed|archived)\b", raw_response, re.IGNORECASE):
            result["operation_type"] = "compression"

        return result

    # --- Environment Lifecycle ---

    def reset(self, containers: list[Container]) -> None:
        """Clean up standard paths across all containers."""
        # Each path needs its own /* glob to clear CONTENTS, not the dir itself
        path_globs = " ".join(f"{p}/*" for p in self._cleanup_paths)
        dotfile_globs = " ".join(f"{p}/.[!.]* {p}/..?*" for p in self._cleanup_paths)
        mkdir_paths = " ".join(self._cleanup_paths)
        cleanup_cmd = (
            f"sh -c 'rm -rf {path_globs} {dotfile_globs} "
            f"/home/alice/.cache /home/alice/.local/share/Trash "
            f"/home/alice/trash /home/alice/old /home/alice/temp "
            f"2>/dev/null; "
            f"mkdir -p {mkdir_paths} "
            f"2>/dev/null || true'"
        )
        for container in containers:
            try:
                _exec_run_with_timeout(
                    container,
                    cleanup_cmd,
                    stdout=False,
                    stderr=False,
                )
            except Exception as exc:
                raise TaskEnvironmentError(
                    f"Failed to reset container {container.id}: {exc}"
                ) from exc

    def populate(
        self,
        containers: list[Container],
        variation: Variation,
    ) -> PopulationOutcome:
        assert isinstance(variation, MemoryCleanupVariation)

        # Step 1: Create directories (explicit mkdir_cmd, not commands[0]).
        # Check exit code: a silent mkdir failure later surfaces as an
        # opaque dd "exit 1" with no diagnostic, since dd's stderr inside
        # the for-loop is swallowed.
        if variation.mkdir_cmd:
            for container in containers:
                assert container.id is not None
                try:
                    exec_result = _exec_run_with_timeout(
                        container,
                        ["sh", "-c", variation.mkdir_cmd],
                    )
                except Exception as exc:
                    raise TaskEnvironmentError(
                        f"mkdir failed on {container.id}: {exc}",
                        container_ids=[container.id],
                    ) from exc
                exit_code, output = _coerce_exec_result(exec_result)
                if exit_code != 0:
                    detail = output.decode(errors="replace").strip() or "(no output)"
                    raise TaskEnvironmentError(
                        f"mkdir failed on {container.id} (exit {exit_code}): "
                        f"{detail}",
                        container_ids=[container.id],
                    )

        # Step 2: Measure baseline on created dirs (absorbs empty-dir overhead).
        # Three attempts: under 12-way parallel populate the first du runs
        # against peak overlay2 contention; one retry was insufficient.
        measurement_dirs = variation.created_dirs or None
        baseline_kb: dict[str, float] = {}
        baseline_max_attempts = 3
        for container in containers:
            assert container.id is not None
            cid = container.id
            measured = None
            for attempt in range(baseline_max_attempts):
                measured = self._measure_population_kb(
                    container,
                    dirs=measurement_dirs,
                )
                if measured is not None:
                    break
                if attempt < baseline_max_attempts - 1:
                    time.sleep(1)
            if measured is None:
                raise TaskEnvironmentError(
                    f"Baseline measurement failed for container {cid}",
                    container_ids=[cid],
                )
            baseline_kb[cid] = measured
            # Also key by name for flexible lookup in verify_population
            if container.name:
                baseline_kb[container.name] = measured

        # Step 3: Wait for created_dirs to be visible to subsequent
        # exec_runs. mkdir from step 1 reported exit 0 inside the
        # container, but under heavy parallel I/O the new dirent is not
        # always visible to the *next* exec_run on its first attempt —
        # overlay2 metadata propagation is queued behind concurrent
        # writes from the other 11 slots. Without this poll, the dd loop
        # below fails with `dd: failed to open '/.../foo.dat': No such
        # file or directory`. Bounded to 1s total per container.
        if variation.created_dirs:
            quoted_dirs = " ".join(
                shlex.quote(d) for d in variation.created_dirs
            )
            wait_script = (
                f"for d in {quoted_dirs}; do "
                f"n=0; while [ ! -d \"$d\" ] && [ $n -lt 10 ]; do "
                f"sleep 0.1; n=$((n+1)); done; "
                f"done"
            )
            for container in containers:
                assert container.id is not None
                try:
                    _exec_run_with_timeout(
                        container, ["sh", "-c", wait_script]
                    )
                except Exception as exc:
                    raise TaskEnvironmentError(
                        f"dir-visibility wait failed on {container.id}: {exc}",
                        container_ids=[container.id],
                    ) from exc

        # Step 4: Run dd commands. Use `sh -ec` so the first failing dd in
        # a `for i in ...` loop aborts the loop immediately (otherwise the
        # loop's exit code is the LAST dd's, hiding the actual failing
        # command). Capture combined stdout+stderr so diagnostics survive.
        # Per-file flushing is baked into each dd via `conv=fdatasync`,
        # so no global `sync` is needed (and global sync would block on
        # all dirty pages system-wide, hanging under heavy parallel I/O).
        results: list[PopulationResult] = []
        for container in containers:
            assert container.id is not None
            try:
                for command in variation.commands:
                    exec_result = _exec_run_with_timeout(
                        container,
                        ["sh", "-ec", command],
                    )
                    exit_code, output = _coerce_exec_result(exec_result)
                    if exit_code != 0:
                        detail = (
                            output.decode(errors="replace").strip()
                            or "(no output)"
                        )
                        raise TaskEnvironmentError(
                            f"Population command failed on "
                            f"{container.id} (exit {exit_code}): "
                            f"{command} -> {detail}"
                        )
                results.append(
                    PopulationResult(
                        container_id=container.id,
                        variation_name=variation.name,
                        description=variation.description,
                        success=True,
                        details={"expected_kb": variation.expected_kb},
                    )
                )
            except Exception as exc:
                logger.error(f"Failed to populate {container.id}: {exc}")
                results.append(
                    PopulationResult(
                        container_id=container.id,
                        variation_name=variation.name,
                        description=variation.description,
                        success=False,
                        error_message=str(exc),
                        details={"expected_kb": variation.expected_kb},
                    )
                )
        return PopulationOutcome(
            results=results,
            episode_context={"baseline_kb": baseline_kb},
        )

    def verify_population(
        self,
        containers: list[Container],
        variation: Variation,
        episode_context: dict[str, Any],
        *,
        private_context: dict[str, Any] | None = None,
    ) -> bool:
        """KB-based delta tolerance check (measured - baseline).

        Measures only variation.created_dirs (scoped to variation-owned paths)
        and uses tolerance_override if set on the variation.
        """
        assert isinstance(variation, MemoryCleanupVariation)
        expected_kb = variation.expected_kb
        tolerance = (
            variation.tolerance_override
            if variation.tolerance_override is not None
            else self._tolerance
        )
        lower_bound = expected_kb * (1.0 - tolerance)
        upper_bound = expected_kb * (1.0 + tolerance)
        baseline_kb: dict[str, float | None] = episode_context.get("baseline_kb", {})
        measurement_dirs = variation.created_dirs or None

        for container in containers:
            measured_kb = self._measure_population_kb(
                container,
                dirs=measurement_dirs,
            )
            if measured_kb is None:
                time.sleep(0.5)
                measured_kb = self._measure_population_kb(
                    container,
                    dirs=measurement_dirs,
                )
            if measured_kb is None:
                logger.error(f"Container {container.id} population measurement failed")
                return False
            # baseline_kb is keyed by container name (from ContainerManager),
            # while container.id is the full SHA. Try name, then id, then prefix.
            assert container.id is not None
            cid = container.id
            cname = container.name or ""
            container_baseline: float | None = None
            for key in (cname, cid):
                if key in baseline_kb:
                    container_baseline = baseline_kb[key]
                    break
            else:
                container_baseline = next(
                    (
                        v
                        for k, v in baseline_kb.items()
                        if v is not None and (cid.startswith(k) or k.startswith(cid))
                    ),
                    None,
                )
            if container_baseline is None:
                logger.error(f"Container {container.id} baseline measurement was None")
                return False
            delta_kb = measured_kb - container_baseline
            if not (lower_bound <= delta_kb <= upper_bound):
                logger.warning(
                    f"Container {container.id} population verification failed: "
                    f"delta={delta_kb:.1f}KB "
                    f"(measured={measured_kb:.1f}KB - "
                    f"baseline={container_baseline:.1f}KB), "
                    f"expected={expected_kb}KB ±{tolerance:.0%} "
                    f"[{lower_bound:.1f}–{upper_bound:.1f}KB]"
                )
                return False
        return True

    # --- Measurement ---

    def _measure_state(
        self,
        containers: list[Container],
    ) -> MemoryCleanupState:
        """Measure used KB in cleanup paths per container and group by filesystem.

        Uses du (via _measure_population_kb) rather than df because df reads
        the shared Docker backing device — under parallel workers, writes
        from other containers pollute each container's df delta and can also
        cause the command itself to fail transiently.
        """
        used_kb: dict[str, float] = {}
        for container in containers:
            assert container.id is not None
            measured = None
            # 5 attempts with 1s/2s/3s/4s backoff (10s total). Sized for
            # 12-way parallel overlay2 contention: the prior 3-attempt
            # 0.5s-step budget cleared too fast against the kernel
            # writeback queue under heavy concurrent dd loads.
            max_attempts = 5
            for attempt in range(max_attempts):
                measured = self._measure_population_kb(container)
                if measured is not None:
                    break
                if attempt < max_attempts - 1:
                    delay = 1.0 * (attempt + 1)
                    logger.warning(
                        f"Measurement retry {attempt + 1}/{max_attempts - 1} "
                        f"for {container.name or container.id} "
                        f"(sleeping {delay:.1f}s)"
                    )
                    time.sleep(delay)
            if measured is None:
                raise TaskEnvironmentError(
                    f"Used-space measurement failed on {container.id} "
                    f"after {max_attempts} attempts",
                    container_ids=[container.id],
                )
            used_kb[container.id] = measured

        filesystem_groups = self._group_by_filesystem(containers)
        return MemoryCleanupState(
            used_kb=used_kb,
            filesystem_groups=filesystem_groups,
        )

    def measure_initial_state(
        self,
        containers: list[Container],
        episode_context: dict[str, Any],
        *,
        private_context: dict[str, Any] | None = None,
    ) -> MemoryCleanupState:
        return self._measure_state(containers)

    def measure_final_state(
        self,
        containers: list[Container],
        episode_context: dict[str, Any],
        artifacts: EpisodeArtifacts,
        *,
        private_context: dict[str, Any] | None = None,
    ) -> MemoryCleanupState:
        # Symmetric measurement — same logic as initial, artifacts unused.
        return self._measure_state(containers)

    # --- Reward (pure) ---

    def compute_reward(
        self,
        initial: MemoryCleanupState,
        final: MemoryCleanupState,
        artifacts: EpisodeArtifacts,
    ) -> TaskReward:
        """Pure computation — no container access."""
        space_freed_kb = self._calculate_deduplicated_space_freed(
            initial.used_kb,
            final.used_kb,
            initial.filesystem_groups,
        )
        return TaskReward(
            value=space_freed_kb,
            success=space_freed_kb > self._success_threshold_kb,
            breakdown={
                "space_freed_kb": space_freed_kb,
                "filesystem_groups": initial.filesystem_groups,
                "initial_used_kb": initial.used_kb,
                "final_used_kb": final.used_kb,
            },
        )

    # --- Private helpers ---

    @staticmethod
    def _group_by_filesystem(
        containers: list[Container],
    ) -> dict[str, list[str]]:
        """Group container IDs by backing filesystem."""
        groups: dict[str, list[str]] = {}
        for container in containers:
            container_id = container.id
            if not container_id:
                continue
            fs_id = _get_container_fs_id(container)
            groups.setdefault(fs_id, []).append(container_id)
        return groups

    @staticmethod
    def _calculate_deduplicated_space_freed(
        initial_used: dict[str, float],
        final_used: dict[str, float],
        filesystem_groups: dict[str, list[str]],
    ) -> float:
        """Compute total space freed by counting each backing filesystem once.

        Inputs are *used* KB (du), so space freed = initial - final. Dedup
        averages members of the same filesystem group; groups are summed.
        """
        total = 0.0
        for fs_id, container_ids in filesystem_groups.items():
            measurements = [
                (initial_used[cid], final_used[cid])
                for cid in container_ids
                if cid in initial_used and cid in final_used
            ]
            if not measurements:
                continue
            avg_before = sum(before for before, _ in measurements) / len(measurements)
            avg_after = sum(after for _, after in measurements) / len(measurements)
            total += avg_before - avg_after
        return total

    def _measure_population_kb(
        self,
        container: Container,
        dirs: list[str] | None = None,
    ) -> float | None:
        """Measure total KB in target directories via du.

        Uses du (not df) because df measures the shared backing device —
        under parallel workers, all containers' writes pollute each other's
        df deltas. du on each container's overlay is isolated.

        Args:
            dirs: if provided, measure only these directories instead of
                  self._cleanup_paths. Used to scope verification to
                  variation-created subdirs.
        """
        paths = list(dirs or self._cleanup_paths)
        quoted = " ".join(shlex.quote(p) for p in paths)
        # Pre-filter missing dirs; suppress du's stderr (benign TOCTOU noise
        # from concurrent deletes is expected); capture du's rc before the awk
        # pipe so real failures (permission / IO) don't get laundered into
        # awk's exit 0. Sentinel-prefixed stdout keeps parsing robust to any
        # remaining stderr leakage.
        # Use positional parameters ($@) rather than a space-joined string so
        # paths containing spaces or shell metacharacters survive word-splitting.
        cmd = (
            "command -v du >/dev/null 2>&1 || exit 127\n"
            "set --\n"
            f"for d in {quoted}; do\n"
            '  [ -e "$d" ] && set -- "$@" "$d"\n'
            "done\n"
            "if [ $# -eq 0 ]; then\n"
            "  echo '__MEASURE_KB__=0'\n"
            "  exit 0\n"
            "fi\n"
            'out=$(du -sk "$@" 2>/dev/null); rc=$?\n'
            "if [ $rc -ne 0 ]; then\n"
            '  echo "__MEASURE_ERR__=$rc"\n'
            "  exit $rc\n"
            "fi\n"
            "printf '%s\\n' \"$out\" | awk '{t+=$1} END{printf \"__MEASURE_KB__=%d\\n\", t+0}'\n"
        )
        try:
            raw = _exec_run_with_timeout(container, ["sh", "-c", cmd])
            exit_code, output = _coerce_exec_result(raw)
        except Exception as exc:
            logger.warning(
                f"Population measurement failed for "
                f"{container.name or container.id}: {exc}"
            )
            return None

        if exit_code != 0:
            logger.warning(
                f"Population measurement failed for "
                f"{container.name or container.id}: "
                f"exit_code={exit_code} output={output!r}"
            )
            return None

        text = output.decode(errors="replace")
        for line in reversed(text.splitlines()):
            if line.startswith("__MEASURE_KB__="):
                try:
                    return float(line.split("=", 1)[1])
                except ValueError:
                    break
        logger.warning(
            f"Population measurement: no sentinel in output for "
            f"{container.name or container.id}: {text!r}"
        )
        return None


def _get_container_fs_id(container: Container) -> str:
    """Get a stable identifier for a container's backing filesystem."""
    try:
        container.reload()
        graph_data = container.attrs.get("GraphDriver", {}).get("Data", {})
        upper_dir = graph_data.get("UpperDir")
        if upper_dir:
            stat_result = os.stat(upper_dir)
            return (
                f"device:{os.major(stat_result.st_dev)}:{os.minor(stat_result.st_dev)}"
            )
    except Exception:
        pass

    container_id_prefix = container.id[:12] if container.id else "unknown"
    return f"container:{container_id_prefix}"
