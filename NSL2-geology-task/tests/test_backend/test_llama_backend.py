import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.typing.config import AppConfig


def make_app_config(
    model_name: str, *, lora_adapter_path: str | None = None
) -> AppConfig:
    return AppConfig(
        model_name=model_name,
        code_host_cache_path="/tmp/code-host-cache",
        container_ids=[],
        main_container_idx=0,
        dynamic_container=False,
        docker_compose_dir="",
        train_data_save_folder="/tmp/train-data",
        llama={"lora_adapter_path": lora_adapter_path} if lora_adapter_path else None,
    )


class LlamaBackendTests(unittest.TestCase):
    @patch("src.backend.llama.terminate_process")
    @patch("src.backend.llama.get_genner")
    @patch("src.backend.llama.OpenAI")
    @patch("src.backend.llama._wait_for_llama_server_ready")
    @patch("src.backend.llama.subprocess.Popen")
    @patch("src.backend.llama._ensure_llama_server")
    @patch("src.backend.llama.is_http_ready", return_value=False)
    def test_setup_llama_passes_lora_adapter_to_server(
        self,
        _mock_is_http_ready: MagicMock,
        mock_ensure_llama_server: MagicMock,
        mock_popen: MagicMock,
        _mock_wait_for_ready: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
        mock_terminate_process: MagicMock,
    ) -> None:
        from src.backend.llama import setup_llama

        mock_ensure_llama_server.return_value = Path("/usr/bin/llama-server")
        mock_popen.return_value = MagicMock()
        mock_openai_cls.return_value = MagicMock()
        mock_get_genner.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            model_path = base_dir / "model.gguf"
            adapter_path = base_dir / "adapter.gguf"
            model_path.write_text("model", encoding="utf-8")
            adapter_path.write_text("adapter", encoding="utf-8")
            app_config = make_app_config(
                f"llama:{model_path}",
                lora_adapter_path=str(adapter_path),
            )

            with setup_llama(app_config):
                pass

        command = mock_popen.call_args.args[0]
        self.assertIn("--lora", command)
        self.assertEqual(
            command[command.index("--lora") + 1],
            str(adapter_path.resolve()),
        )
        mock_terminate_process.assert_called_once()


if __name__ == "__main__":
    unittest.main()
