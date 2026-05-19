"""Smoke tests for the `g2v:analysis` Docker image.

Asserts the structural guarantees documented in
`docs/design/09-docker-runtime.md` §5.6 and in `docker/Dockerfile.analysis`:
the scorer-free analysis image must NOT have `graph_to_voxel.refinement`
(or `graph_to_voxel` at all, or `loopstructural`) importable, and it MUST
have the analysis libraries the geology task expects (`pandas`, `polars`,
`duckdb`, `geopandas`, `rasterio`, `shapely`, plus the `ripgrep` binary).

These tests are skipped unless:
  - Docker is reachable (`docker info` succeeds), and
  - The `g2v:analysis` image is already built (we do **not** build it here —
    image builds are slow and belong in CI, not unit-test runs).

To build the image locally before running these:
    docker compose build analysis
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

IMAGE = "g2v:analysis"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def _image_present(tag: str) -> bool:
    res = subprocess.run(
        ["docker", "image", "inspect", tag],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
    )
    return res.returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_available() or not _image_present(IMAGE),
    reason=f"requires docker + a pre-built {IMAGE} image (run `docker compose build analysis`)",
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a one-shot command inside the analysis image and return the result."""
    return subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "", IMAGE, *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize(
    "module",
    [
        "graph_to_voxel",
        "graph_to_voxel.refinement",
        "loopstructural",
        "pyvista",
        "matplotlib",
        "mcp",
        "zarr",
        "xarray",
        "networkx",
    ],
)
def test_scorer_packages_absent(module: str) -> None:
    """The analysis image must not be able to import representation-side code.

    This is the structural guarantee that shell access (via `analysis_shell`
    in a consumer task) cannot reach the IC criterion or the engine.
    """
    res = _run("python", "-c", f"import {module}")
    assert res.returncode != 0, (
        f"Expected `import {module}` to fail in {IMAGE}, but it succeeded.\n"
        f"stdout: {res.stdout!r}\nstderr: {res.stderr!r}"
    )
    assert "ModuleNotFoundError" in res.stderr or "ImportError" in res.stderr, (
        f"`import {module}` failed for the wrong reason in {IMAGE}:\n"
        f"stdout: {res.stdout!r}\nstderr: {res.stderr!r}"
    )


@pytest.mark.parametrize(
    "module",
    [
        "pandas",
        "polars",
        "numpy",
        "scipy",
        "pyarrow",
        "duckdb",
        "geopandas",
        "rasterio",
        "shapely",
        "pyproj",
        "fiona",
    ],
)
def test_analysis_packages_present(module: str) -> None:
    """The analysis image must ship the libraries the geology task expects."""
    res = _run("python", "-c", f"import {module}")
    assert res.returncode == 0, (
        f"Expected `import {module}` to succeed in {IMAGE}.\n"
        f"stdout: {res.stdout!r}\nstderr: {res.stderr!r}"
    )


@pytest.mark.parametrize("binary", ["rg", "jq", "file", "less"])
def test_shell_tooling_present(binary: str) -> None:
    """Command-line workhorses needed by `analysis_shell` consumers."""
    res = _run("sh", "-c", f"command -v {binary}")
    assert res.returncode == 0, (
        f"Expected `{binary}` on PATH in {IMAGE}.\n"
        f"stdout: {res.stdout!r}\nstderr: {res.stderr!r}"
    )
