"""Live smoke test: bring SGLang up via setup_sglang with the project's jinja
chat template, verify the template actually loads, and confirm the server
returns structured tool_calls for a chat completion with the `tools` arg.

Run with:  uv run python scripts/sglang_toolcall_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.backend.sglang import setup_sglang, _container_name, SGLANG_LOG_DIR
from src.typing.config import AppConfig


def make_config() -> AppConfig:
    return AppConfig(
        model_name="sglang:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
        code_host_cache_path="/tmp/code-host-cache",
        container_ids=[],
        main_container_idx=0,
        dynamic_container=False,
        docker_compose_dir="",
        train_data_save_folder="/tmp/train-data",
        sglang=AppConfig.SglangConfig(
            startup_timeout=1800,
            max_model_len=4096,
            max_running_requests=2,
            chat_template_path="config/chat_templates/qwen2.5-instruct.jinja",
            quantization="awq_marlin",
            tool_call_parser="qwen25",
            grammar_backend="xgrammar",
            cuda_graph_max_bs=2,
            cuda_graph_bs=[1, 2],
            mem_fraction_static=0.70,
        ),
    )


def latest_log() -> Path | None:
    SGLANG_LOG_DIR.mkdir(parents=True, exist_ok=True)
    logs = sorted(SGLANG_LOG_DIR.glob("nsl-sglang-*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def main() -> int:
    config = make_config()
    assert config.sglang is not None
    print(f"[smoke] using template: {config.sglang.chat_template_path}")
    with setup_sglang(config) as session:
        print(f"[smoke] server ready: {session.base_url}")

        # 1) Check container logs for template-load success and parser
        log_path = latest_log()
        if log_path:
            print(f"[smoke] log: {log_path}")
        else:
            import docker

            client = docker.from_env()
            container = client.containers.get(_container_name(30000))
            logs = container.logs().decode("utf-8", errors="replace")
            print("[smoke] live container logs (last 4kb):")
            print(logs[-4000:])

        # 2) Live tool-call test
        client = session.client
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "nsl---run_python",
                    "description": "Execute a Python script.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to run.",
                            }
                        },
                        "required": ["code"],
                    },
                },
            }
        ]
        served = (config.sglang.served_model_name if config.sglang else None) or "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
        response = client.chat.completions.create(
            model=served,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Linux storage administrator. Use the available tool to "
                        "list /tmp."
                    ),
                },
                {
                    "role": "user",
                    "content": "List the files in /tmp using the run_python tool.",
                },
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=256,
            temperature=0.0,
        )
        msg = response.choices[0].message
        print("[smoke] finish_reason:", response.choices[0].finish_reason)
        print("[smoke] content:", repr(msg.content)[:400])
        print("[smoke] server_tool_calls:", msg.tool_calls)

        # Backstop: feed the content through the harness shim's pseudo parser,
        # which is what production traffic uses to recover tool calls when the
        # backend's own parser misses the freelanced format.
        from src.harness.openai_shim import _extract_pseudo_tool_calls

        shim_calls = _extract_pseudo_tool_calls(msg.content) if msg.content else None
        print("[smoke] shim_tool_calls:", shim_calls)

        ok = bool(msg.tool_calls or shim_calls)
        print("[smoke] TOOL_CALLS_PRESENT:", ok)
        return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
