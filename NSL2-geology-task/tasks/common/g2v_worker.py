"""Line-framed JSON worker for the geology graph task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tasks.common.g2v_shim import G2VShim


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="g2v library-mode JSONL worker")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--imports-subdir", default="")
    parser.add_argument("--line-protocol", choices=["stdio"], default="stdio")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    imports_root = workspace / args.imports_subdir if args.imports_subdir else None
    shim = G2VShim(workspace=workspace, imports_root=imports_root)

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                tool = str(request.get("tool") or "")
                payload = shim.dispatch(tool, request.get("args") or {})
            except BaseException as exc:  # noqa: BLE001 - worker stays alive
                payload = {
                    "error": "worker_request_error",
                    "type": type(exc).__name__,
                    "detail": str(exc),
                }
            sys.stdout.write(json.dumps(payload, default=_json_default, sort_keys=True) + "\n")
            sys.stdout.flush()
    finally:
        shim.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
