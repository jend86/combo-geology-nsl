import unittest
from typing import Dict, Tuple
from unittest.mock import MagicMock, patch

from src.tool.docker import (
    FilesystemGroup,
    calculate_deduplicated_space_freed,
    group_containers_by_filesystem,
    wait_and_get_container,
)


class DockerSpaceDedupTests(unittest.TestCase):
    def test_calculates_space_once_for_shared_filesystem(self) -> None:
        measurements = {
            "container_a": (1000.0, 1352.0),
            "container_b": (1000.0, 1352.0),
        }
        groups = [
            FilesystemGroup(
                filesystem_id="fs_shared",
                container_ids=["container_a", "container_b"],
            )
        ]

        total_freed_kb = calculate_deduplicated_space_freed(measurements, groups)

        self.assertEqual(total_freed_kb, 352.0)

    def test_all_containers_in_group_fail_measurement_returns_zero(self) -> None:
        measurements: Dict[str, Tuple[float, float]] = {}
        groups = [
            FilesystemGroup(
                filesystem_id="fs_shared",
                container_ids=["container_a", "container_b"],
            )
        ]

        total_freed_kb = calculate_deduplicated_space_freed(measurements, groups)

        self.assertEqual(total_freed_kb, 0.0)

    def test_mixed_groups_some_with_no_measurements(self) -> None:
        measurements = {
            "container_c": (500.0, 800.0),
        }
        groups = [
            FilesystemGroup(
                filesystem_id="fs_1",
                container_ids=["container_a", "container_b"],
            ),
            FilesystemGroup(
                filesystem_id="fs_2",
                container_ids=["container_c"],
            ),
        ]

        total_freed_kb = calculate_deduplicated_space_freed(measurements, groups)

        self.assertEqual(total_freed_kb, 300.0)

    def test_partial_container_failure_within_group(self) -> None:
        measurements = {
            "container_a": (1000.0, 1400.0),
        }
        groups = [
            FilesystemGroup(
                filesystem_id="fs_shared",
                container_ids=["container_a", "container_b"],
            )
        ]

        total_freed_kb = calculate_deduplicated_space_freed(measurements, groups)

        self.assertEqual(total_freed_kb, 400.0)

    @patch("src.tool.docker.get_container_backing_fs_id")
    def test_groups_containers_by_shared_filesystem(
        self,
        get_fs_id: MagicMock,
    ) -> None:
        get_fs_id.side_effect = ["fs_1", "fs_1", "fs_2"]

        docker_client = MagicMock()
        containers = []
        for container_id in ("a", "b", "c"):
            container = MagicMock()
            container.id = container_id
            containers.append(container)

        groups = group_containers_by_filesystem(docker_client, containers)
        grouped = {group.filesystem_id: group.container_ids for group in groups}

        self.assertEqual(grouped["fs_1"], ["a", "b"])
        self.assertEqual(grouped["fs_2"], ["c"])


class WaitForContainerTests(unittest.TestCase):
    @patch("src.tool.docker.time.sleep")
    def test_running_container_uses_shell_noop_probe(self, sleep: MagicMock) -> None:
        container = MagicMock()
        container.status = "running"
        container.exec_run.return_value = (0, b"")

        docker_client = MagicMock()
        docker_client.containers.get.return_value = container

        with patch("src.tool.docker.time.time", side_effect=[0.0, 0.0]):
            result = wait_and_get_container(docker_client, "container-a", timeout=1)

        self.assertIs(result, container)
        container.exec_run.assert_called_once_with(["sh", "-c", "true"])
        sleep.assert_not_called()

    @patch("src.tool.docker.time.sleep")
    def test_fails_fast_for_terminal_container_state(self, sleep: MagicMock) -> None:
        container = MagicMock()
        container.status = "exited"
        container.logs.return_value = b"process crashed"

        docker_client = MagicMock()
        docker_client.containers.get.return_value = container

        with patch("src.tool.docker.time.time", side_effect=[0.0, 0.0, 2.0]):
            with self.assertRaises(RuntimeError) as context:
                wait_and_get_container(docker_client, "container-a", timeout=1)

        self.assertIn("terminal state 'exited'", str(context.exception))
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
