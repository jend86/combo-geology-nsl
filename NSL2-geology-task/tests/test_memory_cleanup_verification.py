"""Tests for population verification: scoped measurement, calibrated expected_kb,
tolerance override, and drift protection.
"""

import math
import re
from unittest.mock import MagicMock, patch

import pytest

from tasks.memory_cleanup import MemoryCleanupTask, MemoryCleanupVariation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(**overrides) -> MemoryCleanupTask:
    config = {"tolerance": 0.10, **overrides}
    return MemoryCleanupTask(config)


def _parse_mkdir_paths(mkdir_cmd: str) -> set[str]:
    """Extract paths from a 'mkdir -p /a /b /c' command string."""
    # Match everything after 'mkdir -p'
    m = re.search(r"mkdir\s+-p\s+(.+)", mkdir_cmd)
    if not m:
        return set()
    return set(m.group(1).strip().split())


def _parse_dd_output_dirs(commands: list[str]) -> set[str]:
    """Extract parent directories of all dd of=<path> targets."""
    dirs: set[str] = set()
    for cmd in commands:
        for m in re.finditer(r"of=([^\s]+)", cmd):
            path = m.group(1)
            # Parent directory: everything up to the last /
            parent = path.rsplit("/", 1)[0]
            if parent:
                dirs.add(parent)
    return dirs


def _compute_block_aligned_kb(bs_str: str) -> int:
    """Compute du -sk value for a single dd file with given bs.

    bs_str is like '100K', '5K', '1M', '200K'.
    Returns the block-aligned KB value (ceil(bytes / 4096) * 4).
    """
    bs_str = bs_str.strip()
    if bs_str.endswith("M"):
        byte_size = int(bs_str[:-1]) * 1024 * 1024
    elif bs_str.endswith("K"):
        byte_size = int(bs_str[:-1]) * 1024
    else:
        byte_size = int(bs_str)
    blocks = math.ceil(byte_size / 4096)
    return blocks * 4  # KB


def _compute_variation_block_aligned_kb(commands: list[str]) -> float:
    """Compute total du -sk calibrated KB from dd commands.

    Parses 'for i in $(seq 1 N); do dd ... bs=XK count=1 ...; done'
    and non-loop 'dd ... bs=XK count=1 ...' commands.
    """
    total = 0
    for cmd in commands:
        # Match loop pattern: seq 1 N ... bs=XK count=1
        loop_match = re.search(r"seq\s+1\s+(\d+)\).*bs=(\S+)\s+count=(\d+)", cmd)
        if loop_match:
            count_files = int(loop_match.group(1))
            bs = loop_match.group(2)
            count_blocks = int(loop_match.group(3))
            # Assume count=1 for simplicity (all current variations use count=1)
            per_file_kb = _compute_block_aligned_kb(bs)
            total += count_files * per_file_kb * count_blocks
            continue
        # Match direct dd: bs=XK count=1
        dd_match = re.search(r"bs=(\S+)\s+count=(\d+)", cmd)
        if dd_match:
            bs = dd_match.group(1)
            count_blocks = int(dd_match.group(2))
            per_file_kb = _compute_block_aligned_kb(bs)
            total += per_file_kb * count_blocks
    return float(total)


# ---------------------------------------------------------------------------
# Drift protection tests
# ---------------------------------------------------------------------------


class TestDriftProtection:
    """Verify created_dirs, mkdir_cmd, and dd commands stay in sync."""

    def test_variation_created_dirs_cover_all_dd_paths(self):
        """Every dd output path's parent directory must be in created_dirs."""
        task = _make_task()
        for variation in task.list_variations():
            assert isinstance(variation, MemoryCleanupVariation)
            assert hasattr(variation, "created_dirs"), (
                f"{variation.name}: missing created_dirs field"
            )
            dd_dirs = _parse_dd_output_dirs(variation.commands)
            created = set(variation.created_dirs)
            uncovered = dd_dirs - created
            assert not uncovered, (
                f"{variation.name}: dd paths write to dirs not in created_dirs: "
                f"{uncovered}"
            )

    def test_variation_created_dirs_all_have_dd_targets(self):
        """Every created_dirs entry must have at least one dd target underneath."""
        task = _make_task()
        for variation in task.list_variations():
            assert isinstance(variation, MemoryCleanupVariation)
            dd_dirs = _parse_dd_output_dirs(variation.commands)
            for cdir in variation.created_dirs:
                has_target = any(d == cdir or d.startswith(cdir + "/") for d in dd_dirs)
                assert has_target, (
                    f"{variation.name}: created_dirs entry '{cdir}' has no "
                    f"dd targets underneath. Dead configuration."
                )

    def test_variation_mkdir_cmd_covers_created_dirs(self):
        """mkdir_cmd must create all directories in created_dirs.

        created_dirs may be a subset of mkdir paths (some dirs are created
        for the scenario but have no dd targets and are not verified).
        """
        task = _make_task()
        for variation in task.list_variations():
            assert isinstance(variation, MemoryCleanupVariation)
            assert hasattr(variation, "mkdir_cmd"), (
                f"{variation.name}: missing mkdir_cmd field"
            )
            mkdir_paths = _parse_mkdir_paths(variation.mkdir_cmd)
            created = set(variation.created_dirs)
            missing = created - mkdir_paths
            assert not missing, (
                f"{variation.name}: created_dirs entries not in mkdir_cmd: {missing}"
            )


class TestCalibration:
    """Verify expected_kb matches block-aligned du -sk arithmetic."""

    def test_expected_kb_matches_block_aligned_arithmetic(self):
        """For deterministic variations (V1-V4), expected_kb must equal
        the block-aligned total from dd commands."""
        task = _make_task()
        for variation in task.list_variations():
            assert isinstance(variation, MemoryCleanupVariation)
            if "mixed_realistic" in variation.name:
                # V5 uses $RANDOM — skip deterministic check
                continue
            computed = _compute_variation_block_aligned_kb(variation.commands)
            assert computed == variation.expected_kb, (
                f"{variation.name}: block-aligned arithmetic gives "
                f"{computed}KB but expected_kb is {variation.expected_kb}KB"
            )


# ---------------------------------------------------------------------------
# Tolerance / verification tests
# ---------------------------------------------------------------------------


class TestVerifyPopulation:
    """Tests for verify_population behavior with scoped measurement."""

    def _make_mock_container(self, cid="abc123", name="test-container"):
        c = MagicMock()
        c.id = cid
        c.name = name
        return c

    def test_passes_exact_delta(self):
        """Delta exactly matching expected_kb should pass."""
        task = _make_task(tolerance=0.10)
        variation = task.list_variations()[0]
        assert isinstance(variation, MemoryCleanupVariation)
        container = self._make_mock_container()
        baseline = 24.0  # empty-dir overhead
        episode_context = {"baseline_kb": {container.id: baseline}}

        with patch.object(
            task,
            "_measure_population_kb",
            return_value=baseline + variation.expected_kb,
        ):
            result = task.verify_population(
                [container],
                variation,
                episode_context,
            )
        assert result is True

    def test_fails_outside_tolerance(self):
        """Delta at 1.11× expected should fail with ±10% tolerance."""
        task = _make_task(tolerance=0.10)
        variation = task.list_variations()[0]
        assert isinstance(variation, MemoryCleanupVariation)
        container = self._make_mock_container()
        baseline = 24.0
        episode_context = {"baseline_kb": {container.id: baseline}}

        with patch.object(
            task,
            "_measure_population_kb",
            return_value=baseline + variation.expected_kb * 1.11,
        ):
            result = task.verify_population(
                [container],
                variation,
                episode_context,
            )
        assert result is False

    def test_passes_within_tolerance(self):
        """Delta at 1.09× expected should pass with ±10% tolerance."""
        task = _make_task(tolerance=0.10)
        variation = task.list_variations()[0]
        assert isinstance(variation, MemoryCleanupVariation)
        container = self._make_mock_container()
        baseline = 24.0
        episode_context = {"baseline_kb": {container.id: baseline}}

        with patch.object(
            task,
            "_measure_population_kb",
            return_value=baseline + variation.expected_kb * 1.09,
        ):
            result = task.verify_population(
                [container],
                variation,
                episode_context,
            )
        assert result is True

    def test_fails_on_zero_delta(self):
        """Zero delta (measured == baseline) should fail."""
        task = _make_task(tolerance=0.10)
        variation = task.list_variations()[0]
        assert isinstance(variation, MemoryCleanupVariation)
        container = self._make_mock_container()
        baseline = 100.0
        episode_context = {"baseline_kb": {container.id: baseline}}

        with patch.object(
            task,
            "_measure_population_kb",
            return_value=baseline,  # delta = 0
        ):
            result = task.verify_population(
                [container],
                variation,
                episode_context,
            )
        assert result is False

    def test_fails_on_measurement_none(self):
        """None measurement should fail verification."""
        task = _make_task(tolerance=0.10)
        variation = task.list_variations()[0]
        assert isinstance(variation, MemoryCleanupVariation)
        container = self._make_mock_container()
        episode_context = {"baseline_kb": {container.id: 0.0}}

        with patch.object(
            task,
            "_measure_population_kb",
            return_value=None,
        ):
            result = task.verify_population(
                [container],
                variation,
                episode_context,
            )
        assert result is False

    def test_v5_uses_tolerance_override(self):
        """V5 should use its tolerance_override instead of global tolerance."""
        task = _make_task(tolerance=0.10)  # global = 10%
        # Find V5
        v5 = None
        for v in task.list_variations():
            if "mixed_realistic" in v.name:
                v5 = v
                break
        assert v5 is not None, "V5 (mixed_realistic) not found"
        assert isinstance(v5, MemoryCleanupVariation)
        assert hasattr(v5, "tolerance_override"), "V5 missing tolerance_override"
        assert v5.tolerance_override is not None, (
            "V5 tolerance_override should not be None"
        )
        assert v5.tolerance_override > task._tolerance, (
            f"V5 tolerance_override ({v5.tolerance_override}) should be wider "
            f"than global tolerance ({task._tolerance})"
        )

        # Verify it's actually used: a delta at 1.20× should pass with 25%
        # but fail with 10%
        container = self._make_mock_container()
        baseline = 24.0
        episode_context = {"baseline_kb": {container.id: baseline}}

        with patch.object(
            task,
            "_measure_population_kb",
            return_value=baseline + v5.expected_kb * 1.20,
        ):
            result = task.verify_population(
                [container],
                v5,
                episode_context,
            )
        assert result is True, (
            "V5 with 1.20× delta should pass with ±25% tolerance_override"
        )


# ---------------------------------------------------------------------------
# Measurement hardening tests
# ---------------------------------------------------------------------------


class TestMeasurementHardening:
    """Tests for _measure_population_kb command construction."""

    def _capture_measurement_cmd(self, task, dirs=None):
        """Call _measure_population_kb with a mock container and return the
        command string that was passed to exec_run."""
        container = MagicMock()
        container.name = "test"
        container.id = "test123"
        container.exec_run.return_value = (0, b"__MEASURE_KB__=100\n")

        task._measure_population_kb(container, dirs=dirs)

        # exec_run is called with ["sh", "-c", cmd_string]
        call_args = container.exec_run.call_args
        if call_args is None:
            pytest.fail("exec_run was not called")
        cmd_arg = call_args[0][0]  # first positional arg
        if isinstance(cmd_arg, list):
            # ["sh", "-c", "actual command"]
            return cmd_arg[-1]
        return str(cmd_arg)

    def test_measurement_command_has_du_guard(self):
        """du command must be guarded with 'command -v du'."""
        task = _make_task()
        cmd = self._capture_measurement_cmd(task)
        assert "command -v du" in cmd, f"Measurement command missing du guard: {cmd}"

    def test_measurement_captures_du_exit_code_independently(self):
        """du's exit code must be captured independently of the awk pipeline.

        Benign TOCTOU noise during concurrent deletes is suppressed at source
        with 2>/dev/null, but real du failures must not be laundered into
        awk's exit 0. The command captures du's rc before feeding output to
        awk, and emits __MEASURE_ERR__=<rc> on failure.
        """
        task = _make_task()
        cmd = self._capture_measurement_cmd(task)
        assert "rc=$?" in cmd, f"du exit code not captured independently: {cmd}"
        assert "__MEASURE_ERR__=" in cmd, f"no error sentinel for failed du: {cmd}"
        assert "__MEASURE_KB__=" in cmd, f"no kb sentinel for successful du: {cmd}"

    def test_measurement_scoped_to_dirs_when_provided(self):
        """When dirs param is provided, du should target those dirs."""
        task = _make_task()
        custom_dirs = ["/tmp/cleanup", "/var/log/app"]
        cmd = self._capture_measurement_cmd(task, dirs=custom_dirs)
        for d in custom_dirs:
            assert d in cmd, f"Custom dir '{d}' not in measurement command: {cmd}"
        # Should NOT contain the default cleanup paths when dirs is overridden
        assert "/home/alice" not in cmd or "/home/alice" in " ".join(custom_dirs), (
            f"Default paths leak into scoped measurement: {cmd}"
        )


# ---------------------------------------------------------------------------
# Global tolerance default
# ---------------------------------------------------------------------------


class TestMeasurementParser:
    """Sentinel-based parsing of _measure_population_kb output."""

    def _mock_container(self, exec_return):
        c = MagicMock()
        c.name = "test"
        c.id = "test123"
        c.exec_run.return_value = exec_return
        return c

    def test_happy_path_returns_float(self):
        task = _make_task()
        c = self._mock_container((0, b"__MEASURE_KB__=12345\n"))
        assert task._measure_population_kb(c) == 12345.0

    def test_reads_last_sentinel_when_noise_present(self):
        task = _make_task()
        c = self._mock_container(
            (0, b"some line\n__MEASURE_KB__=111\nmore\n__MEASURE_KB__=777\ntail\n")
        )
        assert task._measure_population_kb(c) == 777.0

    def test_all_missing_returns_zero(self):
        task = _make_task()
        c = self._mock_container((0, b"__MEASURE_KB__=0\n"))
        assert task._measure_population_kb(c) == 0.0

    def test_du_failure_returns_none(self, caplog):
        task = _make_task()
        c = self._mock_container((1, b"__MEASURE_ERR__=1\n"))
        with caplog.at_level("WARNING"):
            result = task._measure_population_kb(c)
        assert result is None

    def test_missing_sentinel_returns_none(self):
        task = _make_task()
        c = self._mock_container((0, b"garbage output without sentinel\n"))
        assert task._measure_population_kb(c) is None

    def test_exception_path_returns_none(self):
        task = _make_task()
        c = MagicMock()
        c.name = "test"
        c.id = "test123"
        c.exec_run.side_effect = RuntimeError("boom")
        assert task._measure_population_kb(c) is None


class TestMeasurementShellIntegration:
    """Run the actual generated sh -c command in a subprocess to validate FD
    plumbing, quoting, and POSIX-sh compatibility that unit mocks can't."""

    def _cmd_for(self, task, dirs):
        import subprocess

        container = MagicMock()
        container.name = "t"
        container.id = "t"
        captured = {}

        def fake_exec_run(args, **_kwargs):
            captured["cmd"] = args[-1]
            return (0, b"__MEASURE_KB__=0\n")

        container.exec_run.side_effect = fake_exec_run
        task._measure_population_kb(container, dirs=dirs)
        return captured["cmd"]

    def test_all_existing_dirs_returns_kb_sentinel(self, tmp_path):
        import subprocess

        task = _make_task()
        d1 = tmp_path / "a"
        d1.mkdir()
        (d1 / "f").write_bytes(b"x" * 8192)
        cmd = self._cmd_for(task, [str(d1)])
        result = subprocess.run(
            ["sh", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0, result.stderr
        assert re.search(r"^__MEASURE_KB__=\d+$", result.stdout, re.M), result.stdout

    def test_mix_missing_and_existing(self, tmp_path):
        import subprocess

        task = _make_task()
        d1 = tmp_path / "a"
        d1.mkdir()
        (d1 / "f").write_bytes(b"x" * 8192)
        missing = str(tmp_path / "nope")
        cmd = self._cmd_for(task, [str(d1), missing])
        result = subprocess.run(
            ["sh", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0, result.stderr
        assert re.search(r"^__MEASURE_KB__=\d+$", result.stdout, re.M), result.stdout

    def test_all_missing_returns_zero_sentinel(self, tmp_path):
        import subprocess

        task = _make_task()
        missing1 = str(tmp_path / "nope1")
        missing2 = str(tmp_path / "nope2")
        cmd = self._cmd_for(task, [missing1, missing2])
        result = subprocess.run(
            ["sh", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0, result.stderr
        assert "__MEASURE_KB__=0" in result.stdout

    def test_paths_with_spaces_are_quoted_safely(self, tmp_path):
        import subprocess

        task = _make_task()
        d1 = tmp_path / "dir with space"
        d1.mkdir()
        (d1 / "f").write_bytes(b"x" * 4096)
        cmd = self._cmd_for(task, [str(d1)])
        result = subprocess.run(
            ["sh", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0, result.stderr
        assert re.search(r"^__MEASURE_KB__=\d+$", result.stdout, re.M), result.stdout


class TestToleranceDefault:
    """Verify global tolerance has been tightened."""

    def test_default_tolerance_is_ten_percent(self):
        """Default tolerance should be 0.10, not the old 0.20."""
        task = MemoryCleanupTask({})
        assert task._tolerance == 0.10, (
            f"Default tolerance is {task._tolerance}, expected 0.10"
        )
