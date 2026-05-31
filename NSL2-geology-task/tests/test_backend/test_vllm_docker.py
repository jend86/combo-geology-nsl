import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.genner.Base import Genner
from src.typing.config import AppConfig


def make_app_config(
    model_name: str,
    gpu_mem: float = 0.85,
    chat_template_path: str | None = None,
    local_model_path: str | None = None,
    lora_adapter_path: str | None = None,
    served_model_name: str | None = None,
    compile_cache_dir: str | None = None,
    enable_auto_tool_choice: bool | None = None,
    tool_call_parser: str | None = None,
    reasoning_parser: str | None = None,
) -> AppConfig:
    has_vllm = any(
        v is not None
        for v in (
            chat_template_path,
            local_model_path,
            lora_adapter_path,
            served_model_name,
            compile_cache_dir,
            enable_auto_tool_choice,
            tool_call_parser,
            reasoning_parser,
        )
    )
    return AppConfig(
        model_name=model_name,
        gpu_memory_utilization=gpu_mem,
        code_host_cache_path="/tmp/code-host-cache",
        container_ids=[],
        main_container_idx=0,
        dynamic_container=False,
        docker_compose_dir="",
        train_data_save_folder="/tmp/train-data",
        vllm=AppConfig.VllmConfig(
            chat_template_path=chat_template_path,
            local_model_path=local_model_path,
            lora_adapter_path=lora_adapter_path,
            served_model_name=served_model_name,
            compile_cache_dir=compile_cache_dir,
            enable_auto_tool_choice=enable_auto_tool_choice,
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
        )
        if has_vllm
        else None,
    )


def write_mock_lora_adapter(adapter_dir: str | Path) -> Path:
    resolved_dir = Path(adapter_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    (resolved_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (resolved_dir / "adapter_model.safetensors").write_text(
        "weights",
        encoding="utf-8",
    )
    return resolved_dir


class TestBuildVllmDockerConfig(unittest.TestCase):
    def test_get_network_mode_defaults_to_auto(self) -> None:
        from src.backend.vllm import _get_network_mode

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_get_network_mode(), "auto")

    def test_get_network_mode_falls_back_on_invalid_value(self) -> None:
        from src.backend.vllm import _get_network_mode

        with patch.dict(os.environ, {"NSL_VLLM_NETWORK_MODE": "invalid"}, clear=True):
            self.assertEqual(_get_network_mode(), "auto")

    def test_build_vllm_config_loads_chat_template_from_path(self) -> None:
        from src.backend.vllm import _build_vllm_config

        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
            handle.write("{{ messages[0]['content'] }}")
            handle.flush()

            config = make_app_config(
                "vllm:unsloth/Qwen2.5-Coder-7B-bnb-4bit",
                chat_template_path=handle.name,
            )

            vllm_config = _build_vllm_config(
                config,
                endpoint="http://127.0.0.1:8000",
            )

        self.assertEqual(vllm_config.chat_template, "{{ messages[0]['content'] }}")

    def test_build_vllm_config_resolves_legacy_model_alias(self) -> None:
        from src.backend.vllm import _build_vllm_config

        config = make_app_config("vllm:qwen2.5:7b-instruct")

        vllm_config = _build_vllm_config(
            config,
            endpoint="http://127.0.0.1:8000",
        )

        self.assertEqual(vllm_config.model, "Qwen/Qwen2.5-7B-Instruct")

    def test_vllm_endpoint_list_config_parses(self) -> None:
        config = make_app_config("vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
        config.vllm = AppConfig.VllmConfig(
            endpoints=[
                {
                    "base_url": "https://ep0.example/v1",
                    "capacity": 6,
                    "api_key_env": "VLLM_EP0_KEY",
                },
                {"base_url": "https://ep1.example", "capacity": 12},
            ]
        )

        self.assertEqual(len(config.vllm.endpoints), 2)
        self.assertEqual(config.vllm.endpoints[0].base_url, "https://ep0.example/v1")
        self.assertEqual(config.vllm.endpoints[0].capacity, 6)
        self.assertEqual(config.vllm.endpoints[0].api_key_env, "VLLM_EP0_KEY")

    def test_vllm_legacy_endpoint_string_config_parses(self) -> None:
        config = make_app_config("vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
        config.vllm = AppConfig.VllmConfig(endpoint="http://127.0.0.1:8000")

        self.assertEqual(config.vllm.endpoint, "http://127.0.0.1:8000")

    def test_build_vllm_config_infers_hermes_parser_for_qwen_models(self) -> None:
        from src.backend.vllm import _build_vllm_config

        config = make_app_config("vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")

        vllm_config = _build_vllm_config(
            config,
            endpoint="http://127.0.0.1:8000",
        )

        self.assertTrue(vllm_config.enable_auto_tool_choice)
        self.assertEqual(vllm_config.tool_call_parser, "hermes")

    def test_build_vllm_config_no_reasoning_parser_by_default(self) -> None:
        """reasoning_parser is opt-in; not auto-inferred from model name.

        Even Qwen3 thinking models pay per-step reasoning overhead — only
        configs that explicitly opt in (the harness wants <think> stripped
        from chat content) get the flag.
        """
        from src.backend.vllm import _build_vllm_config

        config = make_app_config("vllm:Qwen/Qwen3-30B-A3B-AWQ")

        vllm_config = _build_vllm_config(
            config,
            endpoint="http://127.0.0.1:8000",
        )

        self.assertIsNone(vllm_config.reasoning_parser)

    def test_build_vllm_config_threads_explicit_reasoning_parser(self) -> None:
        from src.backend.vllm import _build_vllm_config

        config = make_app_config(
            "vllm:Qwen/Qwen3-30B-A3B-AWQ",
            reasoning_parser="qwen3",
        )

        vllm_config = _build_vllm_config(
            config,
            endpoint="http://127.0.0.1:8000",
        )

        self.assertEqual(vllm_config.reasoning_parser, "qwen3")

    def test_builds_container_command_with_reasoning_parser(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen3-30B-A3B-AWQ",
            0.85,
            needs_bnb=False,
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            reasoning_parser="qwen3",
        )

        self.assertIn("--reasoning-parser", cmd)
        self.assertEqual(cmd[cmd.index("--reasoning-parser") + 1], "qwen3")

    def test_builds_container_command_omits_reasoning_parser_when_unset(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
        )

        self.assertNotIn("--reasoning-parser", cmd)

    def test_builds_container_command_for_standard_model(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-7B-Instruct",
            0.85,
            needs_bnb=False,
        )
        self.assertIn("--model", cmd)
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", cmd)
        self.assertIn("--gpu-memory-utilization", cmd)
        self.assertIn("0.85", cmd)
        self.assertNotIn("--quantization", cmd)

    def test_builds_container_command_for_bnb_model(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "unsloth/Qwen2.5-Coder-7B-bnb-4bit",
            0.85,
            needs_bnb=True,
        )
        self.assertIn("--quantization", cmd)
        self.assertIn("bitsandbytes", cmd)
        self.assertIn("--load-format", cmd)

    def test_builds_container_command_with_chat_template(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "unsloth/Qwen2.5-Coder-7B-bnb-4bit",
            0.85,
            chat_template="{{ messages[0]['content'] }}",
            needs_bnb=True,
        )

        self.assertIn("--chat-template", cmd)
        self.assertIn("{{ messages[0]['content'] }}", cmd)

    def test_builds_container_command_with_auto_tool_choice_flags(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
        )

        self.assertIn("--enable-auto-tool-choice", cmd)
        self.assertIn("--tool-call-parser", cmd)
        self.assertEqual(cmd[cmd.index("--tool-call-parser") + 1], "hermes")

    def test_resolve_model_for_container_local_path(self) -> None:
        from src.backend.vllm import (
            LOCAL_MODEL_CONTAINER_PATH,
            _resolve_model_for_container,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_arg, host_mount, needs_bnb = _resolve_model_for_container(
                "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
                tmpdir,
            )

            self.assertEqual(model_arg, LOCAL_MODEL_CONTAINER_PATH)
            self.assertEqual(host_mount, str(Path(tmpdir).resolve()))
            self.assertFalse(needs_bnb)

    def test_resolve_model_for_container_hf_model(self) -> None:
        from src.backend.vllm import _resolve_model_for_container

        model_arg, host_mount, needs_bnb = _resolve_model_for_container(
            "unsloth/Qwen2.5-Coder-7B-bnb-4bit",
            None,
        )

        self.assertEqual(model_arg, "unsloth/Qwen2.5-Coder-7B-bnb-4bit")
        self.assertIsNone(host_mount)
        self.assertTrue(needs_bnb)

    def test_build_container_command_with_lora(self) -> None:
        from src.backend.vllm import (
            LOCAL_ADAPTER_CONTAINER_PATH,
            _build_container_command,
        )

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            lora_adapter_path=LOCAL_ADAPTER_CONTAINER_PATH,
            needs_bnb=False,
        )

        self.assertIn("--enable-lora", cmd)
        self.assertIn("--lora-modules", cmd)
        self.assertIn(f"adapter={LOCAL_ADAPTER_CONTAINER_PATH}", cmd)
        self.assertIn("--max-lora-rank", cmd)
        self.assertIn("64", cmd)

    def test_build_container_command_local_no_bnb(self) -> None:
        from src.backend.vllm import (
            LOCAL_MODEL_CONTAINER_PATH,
            _build_container_command,
        )

        cmd = _build_container_command(
            LOCAL_MODEL_CONTAINER_PATH,
            0.85,
            needs_bnb=False,
        )

        self.assertNotIn("--quantization", cmd)
        self.assertNotIn("bitsandbytes", cmd)

    def test_build_container_command_with_served_model_name(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            served_model_name="nsl-qwen-v1",
            needs_bnb=False,
        )

        self.assertIn("--served-model-name", cmd)
        self.assertIn("nsl-qwen-v1", cmd)

    def test_build_container_command_with_max_model_len(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
            max_model_len=8192,
        )

        self.assertIn("--max-model-len", cmd)
        self.assertIn("8192", cmd)

    def test_build_container_command_with_max_num_seqs(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
            max_num_seqs=10,
        )

        self.assertIn("--max-num-seqs", cmd)
        self.assertIn("10", cmd)

    def test_build_container_command_with_chunked_prefill(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
            enable_chunked_prefill=True,
        )

        self.assertIn("--enable-chunked-prefill", cmd)

    def test_build_container_command_with_tensor_parallel_size(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
            tensor_parallel_size=2,
        )

        self.assertIn("--tensor-parallel-size", cmd)
        self.assertEqual(cmd[cmd.index("--tensor-parallel-size") + 1], "2")

    def test_build_container_command_with_data_parallel_size(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
            data_parallel_size=2,
        )

        self.assertIn("--data-parallel-size", cmd)
        self.assertEqual(cmd[cmd.index("--data-parallel-size") + 1], "2")

    def test_build_container_command_with_pipeline_parallel_size(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
            pipeline_parallel_size=2,
        )

        self.assertIn("--pipeline-parallel-size", cmd)
        self.assertEqual(cmd[cmd.index("--pipeline-parallel-size") + 1], "2")

    def test_build_container_command_with_kv_cache_dtype(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
            kv_cache_dtype="fp8",
        )

        self.assertIn("--kv-cache-dtype", cmd)
        self.assertEqual(cmd[cmd.index("--kv-cache-dtype") + 1], "fp8")

    def test_build_container_command_with_max_num_batched_tokens(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
            max_num_batched_tokens=8192,
        )

        self.assertIn("--max-num-batched-tokens", cmd)
        self.assertEqual(cmd[cmd.index("--max-num-batched-tokens") + 1], "8192")

    def test_build_container_command_omits_tuning_params_when_unset(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
        )

        self.assertNotIn("--max-model-len", cmd)
        self.assertNotIn("--max-num-seqs", cmd)
        self.assertNotIn("--max-num-batched-tokens", cmd)
        self.assertNotIn("--enable-chunked-prefill", cmd)
        self.assertNotIn("--tensor-parallel-size", cmd)
        self.assertNotIn("--data-parallel-size", cmd)
        self.assertNotIn("--pipeline-parallel-size", cmd)
        self.assertNotIn("--kv-cache-dtype", cmd)
        self.assertNotIn("--disable-custom-all-reduce", cmd)

    def test_build_container_command_with_disable_custom_all_reduce(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
            disable_custom_all_reduce=True,
        )

        self.assertIn("--disable-custom-all-reduce", cmd)

    def test_build_container_command_with_enforce_eager(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
            enforce_eager=True,
        )

        self.assertIn("--enforce-eager", cmd)

    def test_build_container_command_with_cudagraph_mode(self) -> None:
        import json

        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
            cudagraph_mode="piecewise",
        )

        self.assertIn("--compilation-config", cmd)
        payload = cmd[cmd.index("--compilation-config") + 1]
        self.assertEqual(
            json.loads(payload),
            {
                "cudagraph_mode": "PIECEWISE",
                "inductor_compile_config": {
                    "combo_kernels": False,
                    "benchmark_combo_kernel": False,
                },
            },
        )

    def test_build_container_command_with_compile_cache_dir(self) -> None:
        import json

        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
            compile_cache_dir="/tmp/vllm_compile_cache",
        )

        self.assertIn("--compilation-config", cmd)
        payload = cmd[cmd.index("--compilation-config") + 1]
        self.assertEqual(
            json.loads(payload),
            {
                "cache_dir": "/tmp/vllm_compile_cache",
                "inductor_compile_config": {
                    "combo_kernels": False,
                    "benchmark_combo_kernel": False,
                },
            },
        )

    def test_build_container_command_merges_compilation_config(self) -> None:
        import json

        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
            cudagraph_mode="piecewise",
            compile_cache_dir="/tmp/vllm_compile_cache",
        )

        self.assertEqual(cmd.count("--compilation-config"), 1)
        payload = cmd[cmd.index("--compilation-config") + 1]
        self.assertEqual(
            json.loads(payload),
            {
                "cache_dir": "/tmp/vllm_compile_cache",
                "cudagraph_mode": "PIECEWISE",
                "inductor_compile_config": {
                    "combo_kernels": False,
                    "benchmark_combo_kernel": False,
                },
            },
        )

    def test_compile_cache_namespace_per_model_and_deterministic(self) -> None:
        # Cache must be model-keyed: switching models must route to a different
        # subdir so a Qwen-era graph is never silently reloaded for Gemma. Same
        # args must produce identical paths (deterministic).
        from src.backend.vllm import _compile_cache_namespace

        gemma = _compile_cache_namespace(
            model="QuantTrio/gemma-4-31B-it-AWQ",
            tp=2,
            max_model_len=32768,
            cudagraph_mode="piecewise",
            enforce_eager=False,
            lora_enabled=False,
        )
        qwen = _compile_cache_namespace(
            model="Qwen/Qwen2.5-Coder-7B-Instruct",
            tp=2,
            max_model_len=32768,
            cudagraph_mode="piecewise",
            enforce_eager=False,
            lora_enabled=False,
        )
        gemma_again = _compile_cache_namespace(
            model="QuantTrio/gemma-4-31B-it-AWQ",
            tp=2,
            max_model_len=32768,
            cudagraph_mode="piecewise",
            enforce_eager=False,
            lora_enabled=False,
        )

        self.assertNotEqual(gemma, qwen)
        self.assertEqual(gemma, gemma_again)
        # First segment must be the human-scannable model bucket.
        self.assertIn("gemma-4-31B-it-AWQ", gemma.split("/", 1)[0])
        self.assertIn("Qwen2.5-Coder-7B-Instruct", qwen.split("/", 1)[0])
        # Path-safe: only one slash, the structural separator we emit.
        self.assertEqual(gemma.count("/"), 1)
        self.assertEqual(qwen.count("/"), 1)

    def test_compile_cache_namespace_discriminates_runtime_knobs(self) -> None:
        from src.backend.vllm import _compile_cache_namespace

        baseline = _compile_cache_namespace(
            model="QuantTrio/gemma-4-31B-it-AWQ",
            tp=2,
            max_model_len=32768,
            cudagraph_mode="piecewise",
            enforce_eager=False,
            lora_enabled=False,
        )
        variants = [
            _compile_cache_namespace(
                model="QuantTrio/gemma-4-31B-it-AWQ",
                tp=1,
                max_model_len=32768,
                cudagraph_mode="piecewise",
                enforce_eager=False,
                lora_enabled=False,
            ),
            _compile_cache_namespace(
                model="QuantTrio/gemma-4-31B-it-AWQ",
                tp=2,
                max_model_len=8192,
                cudagraph_mode="piecewise",
                enforce_eager=False,
                lora_enabled=False,
            ),
            _compile_cache_namespace(
                model="QuantTrio/gemma-4-31B-it-AWQ",
                tp=2,
                max_model_len=32768,
                cudagraph_mode="full",
                enforce_eager=False,
                lora_enabled=False,
            ),
            _compile_cache_namespace(
                model="QuantTrio/gemma-4-31B-it-AWQ",
                tp=2,
                max_model_len=32768,
                cudagraph_mode="piecewise",
                enforce_eager=True,
                lora_enabled=False,
            ),
        ]
        for v in variants:
            self.assertNotEqual(v, baseline)

    def test_compile_cache_namespace_discriminates_lora_enabled(self) -> None:
        from src.backend.vllm import _compile_cache_namespace

        base_only = _compile_cache_namespace(
            model="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            tp=1,
            max_model_len=8192,
            cudagraph_mode="piecewise",
            enforce_eager=False,
            lora_enabled=False,
        )
        with_lora = _compile_cache_namespace(
            model="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            tp=1,
            max_model_len=8192,
            cudagraph_mode="piecewise",
            enforce_eager=False,
            lora_enabled=True,
        )
        self.assertNotEqual(base_only, with_lora)
        # Same model bucket, different fingerprint
        self.assertEqual(base_only.split("/")[0], with_lora.split("/")[0])

    def test_build_container_command_disables_combo_kernels_by_default(self) -> None:
        # Workaround for vLLM 0.19.x default `combo_kernels=True` causing
        # inductor arg-count mismatches on quantized + heterogeneous-head-dim
        # models (e.g. gemma-4 AWQ at TP>=2). Ship combo_kernels disabled until
        # upstream resolves it.
        import json

        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
        )

        self.assertNotIn("--enforce-eager", cmd)
        self.assertIn("--compilation-config", cmd)
        payload = cmd[cmd.index("--compilation-config") + 1]
        self.assertEqual(
            json.loads(payload),
            {
                "inductor_compile_config": {
                    "combo_kernels": False,
                    "benchmark_combo_kernel": False,
                },
            },
        )


class TestVllmContainerLifecycle(unittest.TestCase):
    """Tests that setup_vllm correctly manages the Docker container lifecycle."""

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm.is_http_ready", return_value=True)
    def test_reuses_existing_server(
        self,
        mock_http: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        """When a server is already running, no container should be created."""
        from src.backend.vllm import setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)
        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with setup_vllm(config) as session:
            self.assertIsNotNone(session.genner)
            self.assertIsNone(session.process)
            self.assertIn("endpoint_pool", session.extras)
            pool = session.extras["endpoint_pool"]
            self.assertEqual(pool.endpoint_ids(), ["vllm-0"])

    @patch.dict(os.environ, {"VLLM_EP0_KEY": "secret-0"}, clear=False)
    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._is_http_ready_with_auth", return_value=True)
    def test_configured_external_endpoints_build_endpoint_pool(
        self,
        mock_ready: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        from src.backend.vllm import setup_vllm

        fake0 = MagicMock(spec=Genner)
        fake0.identifier = "vllm"
        fake1 = MagicMock(spec=Genner)
        fake1.identifier = "vllm"
        mock_get_genner.side_effect = [fake0, fake1]
        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")
        config.vllm = AppConfig.VllmConfig(
            endpoints=[
                {
                    "id": "local",
                    "base_url": "http://127.0.0.1:8000",
                    "capacity": 3,
                    "api_key_env": "VLLM_EP0_KEY",
                },
                {
                    "id": "remote",
                    "base_url": "https://remote.example/v1",
                    "capacity": 5,
                },
            ]
        )

        with setup_vllm(config) as session:
            pool = session.extras["endpoint_pool"]
            self.assertEqual(pool.endpoint_ids(), ["local", "remote"])
            self.assertEqual(pool.healthy_capacity(), 8)
            self.assertEqual(session.extras["metrics_api_key"], "secret-0")

        self.assertEqual(mock_openai_cls.call_count, 2)
        self.assertEqual(mock_openai_cls.call_args_list[0].kwargs["api_key"], "secret-0")
        self.assertEqual(mock_openai_cls.call_args_list[1].kwargs["api_key"], "dummy")
        mock_ready.assert_any_call(
            "http://127.0.0.1:8000/v1/models",
            api_key="secret-0",
        )

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._get_host_ip", return_value="10.0.0.8")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_creates_and_removes_container(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_host_ip: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        """When no server exists, a Docker container should be created and removed on exit."""
        from docker.errors import NotFound
        from src.backend.vllm import setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)
        mock_container = MagicMock()
        mock_container.name = "nsl-vllm-8000"
        mock_container.id = "abc123"
        mock_docker = MagicMock()
        # First get() is stale check (no stale), second is post-creation
        mock_docker.containers.get.side_effect = [
            NotFound("not found"),
            mock_container,
        ]
        mock_docker_cls.from_env.return_value = mock_docker

        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with patch.dict(os.environ, {"NSL_VLLM_NETWORK_MODE": "auto"}, clear=False):
            with setup_vllm(config) as session:
                self.assertIsNotNone(session.genner)
                mock_start.assert_called_once()

        wait_urls = mock_wait.call_args.args[0]
        self.assertEqual(
            wait_urls,
            [
                "http://127.0.0.1:8000/v1/models",
                "http://10.0.0.8:8000/v1/models",
            ],
        )
        self.assertEqual(mock_openai_cls.call_count, 1)

        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._get_host_ip", return_value="10.0.0.8")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_hostip_mode_uses_only_host_ip_endpoint(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_host_ip: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        from docker.errors import NotFound
        from src.backend.vllm import setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)
        mock_container = MagicMock()
        mock_container.name = "nsl-vllm-8000"
        mock_container.id = "abc123"
        mock_docker = MagicMock()
        mock_docker.containers.get.side_effect = [
            NotFound("not found"),
            mock_container,
        ]
        mock_docker_cls.from_env.return_value = mock_docker
        mock_wait.return_value = "http://10.0.0.8:8000/v1/models"

        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with patch.dict(os.environ, {"NSL_VLLM_NETWORK_MODE": "hostip"}, clear=True):
            with setup_vllm(config) as session:
                self.assertIsNotNone(session.genner)

        self.assertEqual(
            mock_wait.call_args.args[0], ["http://10.0.0.8:8000/v1/models"]
        )
        self.assertEqual(
            mock_openai_cls.call_args.kwargs["base_url"], "http://10.0.0.8:8000/v1"
        )
        self.assertEqual(
            mock_wait.call_args.kwargs["primary_timeout_s"],
            max(config.inference.timeout, 500),
        )

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_removes_stale_container_before_creating(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        """If a stale container with the same name exists, it should be removed first."""
        from src.backend.vllm import setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)

        # First get() call is for stale removal, second is for getting the new container
        stale_container = MagicMock()
        stale_container.name = "nsl-vllm-8000"
        new_container = MagicMock()
        new_container.name = "nsl-vllm-8000"
        new_container.id = "new123"

        mock_docker = MagicMock()
        mock_docker.containers.get.side_effect = [stale_container, new_container]
        mock_docker_cls.from_env.return_value = mock_docker

        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with setup_vllm(config):
            stale_container.remove.assert_called_once_with(force=True)
            mock_start.assert_called_once()

        new_container.stop.assert_called_once()

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_setup_vllm_with_local_model_path(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        from src.backend.vllm import (
            COMPILE_CACHE_CONTAINER_PATH,
            DEFAULT_COMPILE_CACHE_HOST,
            LOCAL_ADAPTER_CONTAINER_PATH,
            LOCAL_MODEL_CONTAINER_PATH,
            setup_vllm,
        )

        mock_get_genner.return_value = MagicMock(spec=Genner)
        mock_container = MagicMock()
        mock_container.name = "nsl-vllm-8000"
        mock_docker = MagicMock()
        mock_docker.containers.get.return_value = mock_container
        mock_docker_cls.from_env.return_value = mock_docker

        with (
            tempfile.TemporaryDirectory() as model_dir,
            tempfile.TemporaryDirectory() as adapter_dir,
        ):
            write_mock_lora_adapter(adapter_dir)
            config = make_app_config(
                "vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
                local_model_path=model_dir,
                lora_adapter_path=adapter_dir,
                served_model_name="nsl-qwen2.5-v1",
            )

            with setup_vllm(config) as session:
                self.assertEqual(session.config.model, "nsl-qwen2.5-v1")

            start_call = mock_start.call_args
            vllm_command = start_call.args[2]
            extra_volumes = start_call.kwargs["extra_volumes"]

            self.assertIn(LOCAL_MODEL_CONTAINER_PATH, vllm_command)
            self.assertIn("--enable-lora", vllm_command)
            self.assertIn(f"adapter={LOCAL_ADAPTER_CONTAINER_PATH}", vllm_command)
            self.assertIn("--served-model-name", vllm_command)
            self.assertIn("nsl-qwen2.5-v1", vllm_command)
            self.assertNotIn("--quantization", vllm_command)
            self.assertIn(
                (str(Path(model_dir).resolve()), LOCAL_MODEL_CONTAINER_PATH),
                extra_volumes,
            )
            self.assertIn(
                (str(Path(adapter_dir).resolve()), LOCAL_ADAPTER_CONTAINER_PATH),
                extra_volumes,
            )
            self.assertIn(
                (
                    str(DEFAULT_COMPILE_CACHE_HOST.resolve()),
                    COMPILE_CACHE_CONTAINER_PATH,
                ),
                extra_volumes,
            )

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_cleanup_on_startup_failure(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        """If readiness check fails, container should still be cleaned up."""
        from docker.errors import NotFound
        from src.backend.vllm import setup_vllm

        mock_wait.side_effect = TimeoutError("Timed out")
        mock_container = MagicMock()
        mock_container.name = "nsl-vllm-8000"
        mock_docker = MagicMock()
        mock_docker.containers.get.side_effect = [
            NotFound("not found"),
            mock_container,
        ]
        mock_docker_cls.from_env.return_value = mock_docker

        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with self.assertRaises(TimeoutError):
            with setup_vllm(config):
                pass

        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_setup_vllm_uses_persistent_compile_cache_by_default(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        import json

        from src.backend.vllm import (
            COMPILE_CACHE_CONTAINER_PATH,
            DEFAULT_COMPILE_CACHE_HOST,
            _compile_cache_namespace,
            setup_vllm,
        )

        mock_get_genner.return_value = MagicMock(spec=Genner)
        mock_container = MagicMock()
        mock_container.name = "nsl-vllm-8000"
        mock_docker = MagicMock()
        mock_docker.containers.get.return_value = mock_container
        mock_docker_cls.from_env.return_value = mock_docker

        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with setup_vllm(config):
            start_call = mock_start.call_args
            vllm_command = start_call.args[2]
            extra_volumes = start_call.kwargs["extra_volumes"]

        self.assertIn(
            (str(DEFAULT_COMPILE_CACHE_HOST.resolve()), COMPILE_CACHE_CONTAINER_PATH),
            extra_volumes,
        )
        payload = vllm_command[vllm_command.index("--compilation-config") + 1]
        expected_namespace = _compile_cache_namespace(
            model="Qwen/Qwen2.5-7B-Instruct",
            tp=None,
            max_model_len=None,
            cudagraph_mode=None,
            enforce_eager=False,
            lora_enabled=False,
        )
        expected_cache_dir = f"{COMPILE_CACHE_CONTAINER_PATH}/{expected_namespace}"
        self.assertEqual(
            json.loads(payload),
            {
                "cache_dir": expected_cache_dir,
                "inductor_compile_config": {
                    "combo_kernels": False,
                    "benchmark_combo_kernel": False,
                },
            },
        )
        # Per-model bucket must be human-scannable: the first path segment
        # must contain the (sanitized) model identifier.
        first_segment = expected_namespace.split("/", 1)[0]
        self.assertIn("Qwen", first_segment)
        self.assertIn("Qwen2.5-7B-Instruct", first_segment)
        # Namespace subdir must be created on the host so the container mount
        # sees it on first start.
        self.assertTrue(
            (DEFAULT_COMPILE_CACHE_HOST / expected_namespace).exists()
        )

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_setup_vllm_respects_custom_compile_cache_dir(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        import json

        from src.backend.vllm import (
            COMPILE_CACHE_CONTAINER_PATH,
            _compile_cache_namespace,
            setup_vllm,
        )

        mock_get_genner.return_value = MagicMock(spec=Genner)
        mock_container = MagicMock()
        mock_container.name = "nsl-vllm-8000"
        mock_docker = MagicMock()
        mock_docker.containers.get.return_value = mock_container
        mock_docker_cls.from_env.return_value = mock_docker

        with tempfile.TemporaryDirectory() as cache_dir:
            config = make_app_config(
                "vllm:Qwen/Qwen2.5-7B-Instruct",
                compile_cache_dir=cache_dir,
            )

            with setup_vllm(config):
                start_call = mock_start.call_args
                vllm_command = start_call.args[2]
                extra_volumes = start_call.kwargs["extra_volumes"]

        self.assertIn(
            (str(Path(cache_dir).resolve()), COMPILE_CACHE_CONTAINER_PATH),
            extra_volumes,
        )
        payload = vllm_command[vllm_command.index("--compilation-config") + 1]
        expected_namespace = _compile_cache_namespace(
            model="Qwen/Qwen2.5-7B-Instruct",
            tp=None,
            max_model_len=None,
            cudagraph_mode=None,
            enforce_eager=False,
            lora_enabled=False,
        )
        expected_cache_dir = f"{COMPILE_CACHE_CONTAINER_PATH}/{expected_namespace}"
        self.assertEqual(
            json.loads(payload),
            {
                "cache_dir": expected_cache_dir,
                "inductor_compile_config": {
                    "combo_kernels": False,
                    "benchmark_combo_kernel": False,
                },
            },
        )


class TestStartVllmContainer(unittest.TestCase):
    @patch("src.backend.vllm.subprocess.run")
    def test_calls_docker_run_with_gpu_and_volume(self, mock_run: MagicMock) -> None:
        from src.backend.vllm import _start_vllm_container

        mock_run.return_value = MagicMock(returncode=0)
        _start_vllm_container("nsl-vllm-8000", 8000, ["--model", "test"], [])

        call_args = mock_run.call_args[0][0]
        self.assertIn("docker", call_args)
        self.assertIn("nvidia.com/gpu=all", call_args)
        self.assertIn("--ipc=host", call_args)
        self.assertIn("--model", call_args)

    @patch("src.backend.vllm.subprocess.run")
    def test_start_vllm_container_with_extra_volumes(self, mock_run: MagicMock) -> None:
        from src.backend.vllm import _start_vllm_container

        mock_run.return_value = MagicMock(returncode=0)
        _start_vllm_container(
            "nsl-vllm-8000",
            8000,
            ["--model", "test"],
            [("/host/model", "/models/local"), ("/host/adapter", "/adapters/local")],
        )

        call_args = mock_run.call_args.args[0]
        self.assertIn("/host/model:/models/local", call_args)
        self.assertIn("/host/adapter:/adapters/local", call_args)

    @patch("src.backend.vllm.subprocess.run")
    def test_raises_on_docker_failure(self, mock_run: MagicMock) -> None:
        from src.backend.vllm import _start_vllm_container

        mock_run.return_value = MagicMock(returncode=1, stderr="no space left")
        with self.assertRaises(RuntimeError):
            _start_vllm_container("nsl-vllm-8000", 8000, ["--model", "test"], [])

    @patch("src.backend.vllm.subprocess.run")
    def test_extra_env_added_as_docker_env_flags(self, mock_run: MagicMock) -> None:
        from src.backend.vllm import _start_vllm_container

        mock_run.return_value = MagicMock(returncode=0)
        _start_vllm_container(
            "nsl-vllm-8000",
            8000,
            ["--model", "test"],
            extra_env={"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        )

        call_args = mock_run.call_args[0][0]
        self.assertIn("PYTORCH_ALLOC_CONF=expandable_segments:True", call_args)


class TestWaitForVllmReady(unittest.TestCase):
    @patch("src.backend.vllm.is_http_ready")
    @patch("src.backend.vllm.time")
    def test_returns_when_http_ready(
        self, mock_time: MagicMock, mock_http: MagicMock
    ) -> None:
        from src.backend.vllm import _wait_for_vllm_ready

        mock_time.time.side_effect = [0, 0, 0, 1]
        mock_http.return_value = True
        mock_container = MagicMock()
        mock_container.status = "running"

        _wait_for_vllm_ready("http://localhost:8000/v1/models", mock_container, 60)

    @patch("src.backend.vllm.is_http_ready", return_value=False)
    @patch("src.backend.vllm.time")
    def test_raises_on_timeout(
        self, mock_time: MagicMock, mock_http: MagicMock
    ) -> None:
        from src.backend.vllm import _wait_for_vllm_ready

        # time.time() is called multiple times per loop iteration.
        # Use a counter to return values that eventually exceed the deadline.
        call_count = 0

        def fake_time():
            nonlocal call_count
            call_count += 1
            return 0 if call_count == 1 else (30 if call_count < 5 else 91)

        mock_time.time.side_effect = fake_time
        mock_time.sleep = MagicMock()
        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.reload = MagicMock()
        mock_container.logs.return_value = b"loading..."

        with self.assertRaises(TimeoutError):
            _wait_for_vllm_ready("http://localhost:8000/v1/models", mock_container, 60)

    @patch("src.backend.vllm.is_http_ready", return_value=False)
    @patch("src.backend.vllm.time")
    def test_raises_on_container_exit(
        self, mock_time: MagicMock, mock_http: MagicMock
    ) -> None:
        from src.backend.vllm import _wait_for_vllm_ready

        mock_time.time.side_effect = [0, 0, 0, 1]
        mock_time.sleep = MagicMock()
        mock_container = MagicMock()
        mock_container.status = "exited"
        mock_container.reload = MagicMock()
        mock_container.logs.return_value = b"some error"

        with self.assertRaises(RuntimeError, msg="container exited"):
            _wait_for_vllm_ready("http://localhost:8000/v1/models", mock_container, 60)

    @patch("src.backend.vllm.is_http_ready")
    @patch("src.backend.vllm.time")
    def test_tries_multiple_alternate_urls_after_primary_timeout(
        self, mock_time: MagicMock, mock_http: MagicMock
    ) -> None:
        from src.backend.vllm import _wait_for_vllm_ready

        urls = [
            "http://127.0.0.1:8000/v1/models",
            "http://10.0.0.8:8000/v1/models",
            "http://10.0.0.9:8000/v1/models",
        ]
        mock_time.time.side_effect = [0, 0, 0, 1, 1, 11, 11, 11, 61]
        mock_time.sleep = MagicMock()
        mock_http.side_effect = lambda url: url == urls[2]
        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.reload = MagicMock()

        ready_url = _wait_for_vllm_ready(urls, mock_container, 60, primary_timeout_s=10)

        self.assertEqual(ready_url, urls[2])
        self.assertEqual(
            [call.args[0] for call in mock_http.call_args_list],
            [urls[0], urls[1], urls[2]],
        )


class TestWaitForGpuMemoryRelease(unittest.TestCase):
    @patch("src.backend.vllm._read_gpu_memory_info_mb")
    @patch("src.backend.vllm.time")
    def test_wait_for_gpu_memory_release_returns_when_threshold_met(
        self,
        mock_time: MagicMock,
        mock_read_gpu_memory_info_mb: MagicMock,
    ) -> None:
        from src.backend.vllm import wait_for_gpu_memory_release

        mock_time.monotonic.side_effect = [0, 0, 2]
        mock_time.sleep = MagicMock()
        mock_read_gpu_memory_info_mb.side_effect = [
            (2000.0, 12000.0),
            (11000.0, 12000.0),
        ]

        wait_for_gpu_memory_release(
            min_free_memory_fraction=0.9, timeout_s=60, device_indices=[0]
        )

        self.assertEqual(mock_read_gpu_memory_info_mb.call_count, 2)
        mock_time.sleep.assert_called_once_with(2.0)

    @patch("src.backend.vllm._read_gpu_memory_info_mb", return_value=(2000.0, 12000.0))
    @patch("src.backend.vllm.time")
    def test_wait_for_gpu_memory_release_times_out_when_threshold_not_met(
        self,
        mock_time: MagicMock,
        _mock_read_gpu_memory_info_mb: MagicMock,
    ) -> None:
        from src.backend.vllm import wait_for_gpu_memory_release

        mock_time.monotonic.side_effect = [0, 0, 61]
        mock_time.sleep = MagicMock()

        with self.assertRaises(TimeoutError):
            wait_for_gpu_memory_release(
                min_free_memory_fraction=0.9, timeout_s=60, device_indices=[0]
            )

        mock_time.sleep.assert_called_once_with(2.0)

    @patch("src.backend.vllm._read_gpu_memory_info_mb", return_value=None)
    @patch("src.backend.vllm.time")
    def test_wait_for_gpu_memory_release_noops_when_gpu_probe_unavailable(
        self,
        mock_time: MagicMock,
        _mock_read_gpu_memory_info_mb: MagicMock,
    ) -> None:
        from src.backend.vllm import wait_for_gpu_memory_release

        mock_time.monotonic.return_value = 0
        mock_time.sleep = MagicMock()

        wait_for_gpu_memory_release(
            min_free_memory_fraction=0.9, timeout_s=60, device_indices=[0]
        )

        mock_time.sleep.assert_not_called()

    @patch("src.backend.vllm._read_gpu_memory_info_mb")
    @patch("src.backend.vllm.time")
    def test_gate_passes_only_when_all_devices_meet_threshold(
        self,
        mock_time: MagicMock,
        mock_read: MagicMock,
    ) -> None:
        from src.backend.vllm import wait_for_gpu_memory_release

        mock_time.monotonic.side_effect = [0, 0, 2, 4]
        mock_time.sleep = MagicMock()
        # Round 1: device 0 free, device 1 still busy -> gate keeps waiting.
        # Round 2: both free -> gate passes.
        mock_read.side_effect = [
            (11000.0, 12000.0),
            (2000.0, 12000.0),
            (11000.0, 12000.0),
            (11000.0, 12000.0),
        ]

        wait_for_gpu_memory_release(
            min_free_memory_fraction=0.9, timeout_s=60, device_indices=[0, 1]
        )

        self.assertEqual(mock_read.call_count, 4)
        self.assertEqual(
            [c.kwargs.get("device_index", c.args[0] if c.args else None)
             for c in mock_read.call_args_list],
            [0, 1, 0, 1],
        )

    @patch("src.backend.vllm.list_visible_gpu_indices", return_value=[0, 1])
    @patch("src.backend.vllm._read_gpu_memory_info_mb")
    @patch("src.backend.vllm.time")
    def test_gate_auto_enumerates_when_device_indices_none(
        self,
        mock_time: MagicMock,
        mock_read: MagicMock,
        _mock_list: MagicMock,
    ) -> None:
        from src.backend.vllm import wait_for_gpu_memory_release

        mock_time.monotonic.side_effect = [0, 0]
        mock_time.sleep = MagicMock()
        mock_read.return_value = (11000.0, 12000.0)

        wait_for_gpu_memory_release(min_free_memory_fraction=0.9, timeout_s=60)

        self.assertEqual(mock_read.call_count, 2)

    @patch("src.backend.vllm.list_visible_gpu_indices", return_value=[])
    @patch("src.backend.vllm._read_gpu_memory_info_mb")
    @patch("src.backend.vllm.time")
    def test_gate_noops_when_enumeration_returns_empty(
        self,
        mock_time: MagicMock,
        mock_read: MagicMock,
        _mock_list: MagicMock,
    ) -> None:
        from src.backend.vllm import wait_for_gpu_memory_release

        mock_time.monotonic.return_value = 0
        mock_time.sleep = MagicMock()

        wait_for_gpu_memory_release(min_free_memory_fraction=0.9, timeout_s=60)

        mock_read.assert_not_called()
        mock_time.sleep.assert_not_called()


class TestResolveModelForContainer(unittest.TestCase):
    def test_hf_awq_model_no_bnb_flags(self) -> None:
        from src.backend.vllm import _resolve_model_for_container

        model_arg, host_path, needs_bnb = _resolve_model_for_container(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ", None
        )
        self.assertEqual(model_arg, "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
        self.assertIsNone(host_path)
        self.assertFalse(needs_bnb)


class TestSetupVllmWithLocalModel(unittest.TestCase):
    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._get_host_ip", return_value="10.0.0.8")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_local_model_mounts_and_no_bnb(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_host_ip: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        from docker.errors import NotFound
        from src.backend.vllm import LOCAL_MODEL_CONTAINER_PATH, setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)
        mock_container = MagicMock()
        mock_docker = MagicMock()
        mock_docker.containers.get.side_effect = [NotFound(""), mock_container]
        mock_docker_cls.from_env.return_value = mock_docker

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_app_config(
                "vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
                local_model_path=tmpdir,
            )
            with setup_vllm(config):
                start_call = mock_start.call_args
                vllm_command = start_call[0][2]
                extra_volumes = start_call[1].get("extra_volumes", [])

                self.assertIn(LOCAL_MODEL_CONTAINER_PATH, vllm_command)
                self.assertNotIn("--quantization", vllm_command)
                mounted_host_paths = [v[0] for v in extra_volumes]
                self.assertIn(tmpdir, mounted_host_paths)

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._get_host_ip", return_value="10.0.0.8")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_lora_adapter_path_mounts_and_enables_lora(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_host_ip: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        from docker.errors import NotFound
        from src.backend.vllm import setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)
        mock_container = MagicMock()
        mock_docker = MagicMock()
        mock_docker.containers.get.side_effect = [NotFound(""), mock_container]
        mock_docker_cls.from_env.return_value = mock_docker

        with tempfile.TemporaryDirectory() as tmpdir:
            write_mock_lora_adapter(tmpdir)
            config = make_app_config(
                "vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
                lora_adapter_path=tmpdir,
            )
            with setup_vllm(config):
                start_call = mock_start.call_args
                vllm_command = start_call[0][2]
                extra_volumes = start_call[1].get("extra_volumes", [])

                self.assertIn("--enable-lora", vllm_command)
                self.assertIn("--lora-modules", vllm_command)
                mounted_host_paths = [v[0] for v in extra_volumes]
                self.assertIn(tmpdir, mounted_host_paths)

    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_invalid_lora_adapter_path_fails_before_startup(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
    ) -> None:
        from docker.errors import NotFound
        from src.backend.vllm import setup_vllm

        mock_docker = MagicMock()
        mock_docker.containers.get.side_effect = NotFound("")
        mock_docker_cls.from_env.return_value = mock_docker

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_app_config(
                "vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
                lora_adapter_path=tmpdir,
            )

            with self.assertRaisesRegex(ValueError, "adapter_config.json"):
                with setup_vllm(config):
                    pass

        mock_start.assert_not_called()

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._get_host_ip", return_value="10.0.0.8")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_served_model_name_updates_api_model(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_host_ip: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        from docker.errors import NotFound
        from src.backend.vllm import setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)
        mock_container = MagicMock()
        mock_docker = MagicMock()
        mock_docker.containers.get.side_effect = [NotFound(""), mock_container]
        mock_docker_cls.from_env.return_value = mock_docker

        config = make_app_config(
            "vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            served_model_name="nsl-qwen-v1",
        )
        with setup_vllm(config) as session:
            self.assertEqual(session.config.model, "nsl-qwen-v1")


if __name__ == "__main__":
    unittest.main()
