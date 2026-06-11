"""Phase-0 safety fix: run_generation_only must accept a stable --run-id so
remote/on-pod runs are resumable by a predictable run id.

Mirrors run_train_loop.py's --run-id plumbing; the runtime
(open_backend_runtime) already accepts run_id, so this only covers the CLI +
main() wiring.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import scripts.run_generation_only as rgo


def test_parser_accepts_run_id() -> None:
    parser = rgo._build_parser()

    ns = parser.parse_args(["config.toml", "--run-id", "abc-123"])
    assert ns.run_id == "abc-123"

    # Defaults to None so a fresh generated id is used when omitted.
    assert parser.parse_args(["config.toml"]).run_id is None


def test_main_propagates_run_id_to_runtime(tmp_path: Path) -> None:
    cfg = MagicMock()
    cfg.generation.generation_output_dir = str(tmp_path / "gen")

    with (
        patch.object(rgo, "_load_config", return_value=cfg),
        patch.object(rgo, "ensure_configured_harness"),
        patch.object(rgo, "open_backend_runtime") as m_obr,
        patch.object(rgo, "run_generation"),
        patch.object(rgo, "save_generation_data"),
    ):
        m_obr.return_value.__enter__.return_value.run_id = "stable-run"
        rgo.main("config.toml", run_id="stable-run")

    # open_backend_runtime(config, run_id="stable-run")
    _, kwargs = m_obr.call_args
    assert kwargs.get("run_id") == "stable-run"
