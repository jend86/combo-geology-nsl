from __future__ import annotations

import asyncio
import json
import sys
import traceback
from pathlib import Path

from nat.utils import run_workflow


WORK = Path("/work")
FINAL = WORK / "final_answer.txt"


async def _run_one_step(step_cfg: Path, prompt: str) -> str:
    return await run_workflow(config_file=str(step_cfg), prompt=prompt, to_type=str) or ""


async def _main() -> int:
    workflow_manifest = WORK / "workflow.json"
    final_text = ""
    try:
        if workflow_manifest.exists():
            manifest = json.loads(workflow_manifest.read_text())
            prior_outputs: list[tuple[str, str]] = []
            for step in manifest["steps"]:
                step_prompt = step["prompt"]
                if step.get("inherit_context") and prior_outputs:
                    inherited = "\n\n".join(
                        f"[{name} output]\n{out}" for name, out in prior_outputs
                    )
                    step_prompt = f"{inherited}\n\n{step_prompt}"
                output = await _run_one_step(WORK / step["config"], step_prompt)
                prior_outputs.append((step["name"], output))
                final_text = output
        else:
            prompt = (WORK / "query.txt").read_text()
            final_text = await _run_one_step(WORK / "agent.yaml", prompt)
        return 0
    except NotImplementedError as exc:
        print(f"[nsl] aiq HITL invoked: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        print(f"[nsl] aiq failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        FINAL.write_text(final_text)


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
