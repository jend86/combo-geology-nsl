from __future__ import annotations

from pathlib import Path

from src.task.types import CapabilityExecutionContext
from tasks.feature_hypothesis_kazakhstan import FeatureHypothesisKazakhstanTask


class _FakeContainer:
    def __init__(self, service: str) -> None:
        self.name = f"fake-{service}"
        self.id = self.name
        self.attrs = {
            "Config": {"Labels": {"com.docker.compose.service": service}}
        }


def _task(tmp_path: Path) -> FeatureHypothesisKazakhstanTask:
    return FeatureHypothesisKazakhstanTask(
        {
            "store_dir": str(tmp_path / "store"),
            "kg_dir": str(tmp_path / "knowledge"),
        }
    )


def _ctx(tmp_path: Path, *, phase_records: dict | None = None) -> CapabilityExecutionContext:
    return CapabilityExecutionContext(
        episode_id="ep_gen0_0001_123",
        workflow_step="code",
        episode_context={
            "episode_id": "task_ep_1",
            "framework_episode_id": "ep_gen0_0001_123",
            "run_id": "run-abc/unsafe",
            "train_data_save_folder": str(tmp_path / "train_data"),
            "store_dir": str(tmp_path / "store" / "teniz_basin"),
            "kg_dir": str(tmp_path / "knowledge" / "teniz_basin"),
            "phase_records": phase_records or {},
        },
    )


def test_kazakhstan_execution_submit_uses_analysis_container_and_repo_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = _task(tmp_path)
    analysis = _FakeContainer("analysis")
    captured: dict = {}

    from voxel_features.mcp.tools import execution_tools

    def fake_submit(**kwargs):
        captured.update(kwargs)
        return {"success": True, "execution_id": "exec_1", "status": "pending"}

    monkeypatch.setattr(execution_tools, "execution_submit", fake_submit)

    result = task._exec_execution_capability(
        [analysis],
        {"code": "print('ok')"},
        _ctx(tmp_path),
        "execution_submit",
    )

    assert result.success is True
    assert captured["container"] is analysis
    artifact_root = Path(captured["artifact_root"])
    assert tmp_path / "train_data" / "artifacts" in artifact_root.parents
    assert artifact_root.name == "ep_gen0_0001_123"
    assert artifact_root.parent.name == "run-abc_unsafe"
    assert "/tmp/voxel-features" not in str(artifact_root)


def test_execution_tool_records_framework_artifact_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from voxel_features.mcp.tools import execution_tools

    execution_tools._sessions.clear()
    execution_tools._executions.clear()
    captured: dict = {}

    def fake_execute(record):
        captured["artifact_root"] = record.artifact_root
        record.status = execution_tools.ExecutionStatus.COMPLETED
        record.exit_code = 0

    class ImmediateThread:
        def __init__(self, *, target, args, name):
            self.target = target
            self.args = args
            self.name = name

        def start(self):
            self.target(*self.args)

    artifact_root = tmp_path / "train_data" / "artifacts" / "run" / "episode"
    monkeypatch.setattr(execution_tools, "_execute_code_in_thread", fake_execute)
    monkeypatch.setattr(execution_tools.threading, "Thread", ImmediateThread)

    result = execution_tools.execution_submit(
        code="x = 1",
        session_id="artifact-root-test",
        artifact_root=str(artifact_root),
    )

    assert result["success"] is True
    assert captured["artifact_root"] == str(artifact_root)
    record = execution_tools._executions[result["execution_id"]]
    assert record.artifact_root == str(artifact_root)


def test_execution_tool_default_artifact_root_is_repo_local(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from voxel_features.mcp.tools import execution_tools

    monkeypatch.delenv("VFM_ARTIFACT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    assert execution_tools._default_artifact_root() == str(
        tmp_path / "data" / "execution-artifacts"
    )


def test_get_experiment_summary_compacts_large_code_outputs(tmp_path: Path) -> None:
    task = _task(tmp_path)
    ctx = _ctx(
        tmp_path,
        phase_records={
            "hypothesise": {"hypothesis": "h", "data_spec": {"features": ["x"]}},
            "code": {
                "code_executed": "print('x')\n" * 1000,
                "result_summary": "stdout line\n" * 2000,
                "artifact_directory": str(tmp_path / "train_data" / "artifacts"),
                "artifact_files": ["a.csv", "b.json"],
            },
        },
    )

    result = task._exec_get_experiment_summary([], ctx)

    assert result.success is True
    assert len(result.output["code_executed"]) < 2500
    assert len(result.output["result_summary"]) < 3000
    assert result.output["code_executed_truncated"] is True
    assert result.output["result_summary_truncated"] is True
    assert result.output["artifact_count"] == 2


def test_scoring_fails_fast_when_requested_layer_does_not_exist(tmp_path: Path) -> None:
    task = _task(tmp_path)
    ctx = _ctx(tmp_path)

    result = task._exec_scoring_capability(
        [],
        {"name": "missing_layer", "dtype": "float"},
        ctx,
        "scoring_create_feature_layer",
    )

    assert result.success is False
    assert "missing_layer" in (result.error or "")
    assert "spatial_add" in (result.error or "")
    assert "evaluate" not in ctx.episode_context.get("phase_records", {})


def test_scoring_uses_last_translated_layer_when_requested_name_is_stale(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = _task(tmp_path)
    store_dir = tmp_path / "store" / "teniz_basin"
    scratch_dir = store_dir / "scratch" / "task_ep_1"
    admitted_dir = store_dir / "admitted"
    admitted_dir.mkdir(parents=True, exist_ok=True)

    from voxel_features.spatial import SpatialVoxelStore
    from voxel_features.store import GridSpec
    from voxel_features.mcp.tools import scoring_tools
    from tasks.feature_hypothesis_kazakhstan import _KAZAKHSTAN_TENIZ_GRID

    np = __import__("numpy")

    grid = GridSpec.from_dict(_KAZAKHSTAN_TENIZ_GRID)
    store = SpatialVoxelStore(scratch_dir, grid, read_only_overlay=admitted_dir)
    store.add_layer("actual_layer", np.zeros(grid.shape), dtype="float")

    called: dict = {}

    def fake_score(store, **kwargs):
        called.update(kwargs)
        return {
            "success": True,
            "layer_name": kwargs["name"],
            "bic_delta": 0.0,
            "admitted": False,
        }

    monkeypatch.setattr(scoring_tools, "scoring_create_feature_layer", fake_score)
    ctx = _ctx(
        tmp_path,
        phase_records={"translate": {"feature_layer_name": "actual_layer"}},
    )

    result = task._exec_scoring_capability(
        [],
        {"name": "stale_layer", "dtype": "float"},
        ctx,
        "scoring_create_feature_layer",
    )

    assert result.success is True
    assert called["name"] == "actual_layer"
    assert ctx.episode_context["phase_records"]["evaluate"]["layer_name"] == "actual_layer"
