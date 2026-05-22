import io
import os
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, cast

from loguru import logger
from result import Err, Ok, Result

import docker
import docker.errors
from docker import DockerClient
from docker.models.containers import Container as DockerContainer
from src.helper import timeout


@dataclass
class FilesystemGroup:
    filesystem_id: str
    container_ids: List[str]


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int


def write_code_in_con(
    client: DockerClient,
    container: DockerContainer,
    host_cache_folder: Path,
    code: str,
    postfix: str,
    in_container_path: str = "/",
) -> Tuple[str, str]:
    """Write code into a temporary file in the host machine first then to the container.

    Algorithm:
    - Write code into a temporary file in the host machine
    - Create a tar archive containing the file
    - Copy the tar archive to the container's root directory
    - Check if the file exists in the container

    Args:
        code (str): The code to write into the container
        postfix (str): The type identifier for the agent, used in the file path
        in_container_path (str, optional): The base path in the container to write the code to. Defaults to "/".

    Raises:
        Exception: If the file cannot be written to the container or if verification fails

    Returns:
        Tuple[str, str]:
            - The path to the temporary file in the container
            - The reflected code (content of the file as read from the container)
    """
    # Create temp file name with timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file_name = f"temp_script_{current_time}.py"
    temp_file_path = f"{in_container_path}/{temp_file_name}"

    # Create host file path and ensure directory exists
    # logger.info(f"Writing file {temp_file_name} into host machine")
    host_path = host_cache_folder / temp_file_name
    host_path.parent.mkdir(parents=True, exist_ok=True)
    host_path.write_text(code)

    # Create a tar archive in memory
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        tar.add(host_path, arcname=temp_file_name)
    tar_stream.seek(0)

    # Copy the file to the container's root directory
    # logger.info(f"Writing file {temp_file_name} into container")

    succeed = container.put_archive(path=in_container_path, data=tar_stream.read())

    if not succeed:
        raise Exception("Failed to write code into the container")

    # Check if file exists in container
    check_exist_command = (
        f"test -f {temp_file_path} && echo 'File exists' || echo 'File does not exist'"
    )
    check_exist_result = container.exec_run(cmd=["/bin/sh", "-c", check_exist_command])

    if b"File exists" not in check_exist_result.output:
        logger.error(
            f"File verification failed: {check_exist_result.output.decode('utf-8')}"
        )
        raise Exception(
            f"File verification failed: {check_exist_result.output.decode('utf-8')}"
        )

    # Read the file content
    reflected_code = container.exec_run(cmd=["cat", temp_file_path]).output.decode(
        "utf-8"
    )
    assert isinstance(reflected_code, str)

    return temp_file_path, reflected_code


def run_code_in_con(
    container: DockerContainer,
    in_container_script_path: str,
    timeout_seconds: int = 600,
) -> Result[ExecutionResult, str]:
    """Run a Python script path inside a container and return its output.

    Args:
        container: The container to run the code in.
        in_container_script_path: The in-container path to the Python script.
        timeout_seconds: Maximum execution time before failure.

    Returns:
        Result[ExecutionResult, str]:
            - Ok: Execution result with stdout, stderr, and exit code.
            - Err: Error details when execution fails or times out.

    Note:
        - After execution, any remaining Python processes are killed.
    """
    command_str = f"python -u {in_container_script_path}"
    cmd = ["/bin/sh", "-c", command_str]  # Execute via shell
    timeout_bool = False
    timeout_checker = time.time()
    try:
        with timeout(seconds=timeout_seconds):
            python_exit_code, python_output = cast(
                Tuple[int, Tuple[bytes | None, bytes | None]],
                container.exec_run(
                    cmd=cmd,
                    demux=True,
                    stream=False,  # Wait for the command to finish and return all output at once
                ),
            )
            if time.time() - timeout_checker > timeout_seconds:
                timeout_bool = True
            stdout_bytes, stderr_bytes = python_output
            stdout = (
                stdout_bytes.decode("utf-8", errors="replace")
                if stdout_bytes is not None
                else ""
            )
            stderr = (
                stderr_bytes.decode("utf-8", errors="replace")
                if stderr_bytes is not None
                else ""
            )
    except TimeoutError as e:
        return Err(f"ContainerManager.run_code_in_con: Code ran too long, error: \n{e}")
    except docker.errors.ContainerError as e:
        return Err(f"ContainerManager.run_code_in_con: Container error, error: \n{e}")
    if timeout_bool:
        return Err(
            "ContainerManager.run_code_in_con: Code ran too long, output: \n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )

    container.exec_run(cmd="kill -9 $(pidof python)")

    return Ok(ExecutionResult(stdout=stdout, stderr=stderr, exit_code=python_exit_code))


def get_container_free_disk_space_kb_v2(
    container: DockerContainer,
) -> float:
    """
    Gets the available disk space for the root filesystem inside a container.

    Returns:
        float: Available space in Kilobytes.
    """
    try:
        # Execute df -k to get disk usage in Kilobytes
        exit_code, output = container.exec_run("df -k /")

        if exit_code == 0 and isinstance(output, bytes):
            logger.debug(
                f"Successfully retrieved storage info using `df -k /`. Output: \n{output.decode().strip()}"
            )

            lines = output.decode().strip().splitlines()
            # Ensure we have the data line to parse
            if len(lines) < 2:
                raise Exception(
                    "`df -k /` output is in an unexpected format: not enough lines."
                )

            storage_info = lines[1].split()

            # Ensure the line has enough columns
            if len(storage_info) < 4:
                raise Exception(
                    "`df -k /` output is in an unexpected format: not enough columns."
                )

            # Index 3 corresponds to the 'Available' column with df
            available_kb = float(storage_info[3])  # This value is in Kilobytes

            return available_kb
        else:
            # Correctly reference the command that was executed
            error_output = output.decode() if isinstance(output, bytes) else str(output)
            raise Exception(
                f"Failed to retrieve storage information with `df -k /`. Exit code: {exit_code}, Output: {error_output}"
            )

    except Exception as e:
        logger.error(f"Storage info retrieval failed: {e}")
        raise e


def get_container_backing_fs_id(
    client: DockerClient,
    container: DockerContainer,
) -> str:
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
    except Exception as exc:
        logger.debug(
            f"Failed to determine device-based filesystem id for {container.id}: {exc}"
        )

    try:
        info = client.info()
        docker_root_dir = info.get("DockerRootDir")
        storage_driver = info.get("StorageDriver", "unknown")
        if docker_root_dir:
            return f"daemon:{storage_driver}:{docker_root_dir}"
    except Exception as exc:
        logger.debug(
            f"Failed to determine daemon-based filesystem id for {container.id}: {exc}"
        )

    container_id_prefix = container.id[:12] if container.id else "unknown"
    return f"container:{container_id_prefix}"


def group_containers_by_filesystem(
    client: DockerClient,
    containers: List[DockerContainer],
) -> List[FilesystemGroup]:
    groups: Dict[str, FilesystemGroup] = {}

    for container in containers:
        if container.id is None:
            logger.warning(
                "Skipping container with missing id when grouping filesystems"
            )
            continue

        filesystem_id = get_container_backing_fs_id(client, container)
        if filesystem_id not in groups:
            groups[filesystem_id] = FilesystemGroup(
                filesystem_id=filesystem_id,
                container_ids=[],
            )
        groups[filesystem_id].container_ids.append(container.id)

    return list(groups.values())


def calculate_deduplicated_space_freed(
    measurements: Dict[str, Tuple[float, float]],
    filesystem_groups: List[FilesystemGroup],
) -> float:
    """
    Compute total space delta by counting each backing filesystem once.

    measurements maps container_id -> (before_free_kb, after_free_kb).
    """
    total_space_freed_kb = 0.0

    for group in filesystem_groups:
        group_measurements = [
            measurements[container_id]
            for container_id in group.container_ids
            if container_id in measurements
        ]
        if not group_measurements:
            continue

        avg_before_kb = sum(before for before, _ in group_measurements) / len(
            group_measurements
        )
        avg_after_kb = sum(after for _, after in group_measurements) / len(
            group_measurements
        )

        total_space_freed_kb += avg_after_kb - avg_before_kb

    return total_space_freed_kb


def wait_and_get_container(
    client: DockerClient, container_name: str, timeout: int = 30
) -> DockerContainer:
    """Wait for container to be fully running and return the container object."""
    start_time = time.time()
    terminal_states = {"exited", "dead"}

    while time.time() - start_time < timeout:
        try:
            container = client.containers.get(container_name)
            container.reload()

            if container.status in terminal_states:
                logs = ""
                try:
                    logs_output = container.logs(tail=20)
                    logs = (
                        logs_output.decode("utf-8", errors="replace")
                        if isinstance(logs_output, bytes)
                        else str(logs_output)
                    ).strip()
                except Exception:
                    logs = "<unavailable>"

                detail = f" Last logs:\n{logs}" if logs else ""
                raise RuntimeError(
                    f"Container {container_name} is in terminal state '{container.status}'.{detail}"
                )

            if container.status == "running":
                # Test if Docker exec is usable without requiring procps in the image.
                exit_code, output = container.exec_run(["sh", "-c", "true"])
                if exit_code == 0:
                    logger.info(f"Container {container_name} is fully running")
                    return container

            logger.debug(f"Container status: {container.status}, waiting...")
            time.sleep(1)

        except docker.errors.NotFound:
            logger.debug(f"Container {container_name} not found yet, waiting...")
            time.sleep(1)
            continue

    raise TimeoutError(
        f"Container {container_name} did not start properly within {timeout} seconds"
    )
