import os
import unittest
from unittest.mock import MagicMock, patch

from src.genner import BackendException, get_genner
from src.genner.Base import Genner
from src.typing.config import AppConfig
from src.backend.resolver import get_backend_context_factory, resolve_backend_context


def make_app_config(model_name: str) -> AppConfig:
    return AppConfig(
        model_name=model_name,
        code_host_cache_path="/tmp/code-host-cache",
        container_ids=[],
        main_container_idx=0,
        dynamic_container=False,
        docker_compose_dir="",
        train_data_save_folder="/tmp/train-data",
    )


class BackendResolverTests(unittest.TestCase):
    def test_retired_qwen_backend_is_unsupported(self) -> None:
        self.assertIsNone(get_backend_context_factory("qwen-cleanup-merged"))
        self.assertIsNone(resolve_backend_context(make_app_config("qwen-cleanup-merged")))

    def test_retired_qwen_peft_backend_is_unsupported(self) -> None:
        self.assertIsNone(get_backend_context_factory("qwen-peft"))
        self.assertIsNone(resolve_backend_context(make_app_config("qwen-peft")))

    def test_retired_genner_backends_are_unsupported(self) -> None:
        for backend in ("qwen", "qwen-finetuned", "qwen-cleanup-merged", "dream"):
            with self.subTest(backend=backend):
                with self.assertRaises(BackendException):
                    get_genner(backend)


class BackendContextBehaviorTests(unittest.TestCase):
    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm.is_http_ready", return_value=True)
    def test_vllm_context_uses_existing_server_when_ready(
        self,
        _is_http_ready: MagicMock,
        openai_cls: MagicMock,
        get_genner: MagicMock,
    ) -> None:
        fake_client = MagicMock()
        fake_genner = MagicMock(spec=Genner)
        openai_cls.return_value = fake_client
        get_genner.return_value = fake_genner

        context = resolve_backend_context(make_app_config("vllm:demo-model"))
        self.assertIsNotNone(context)

        with context as session:
            self.assertIs(session.client, fake_client)
            self.assertIs(session.genner, fake_genner)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False)
    @patch("src.backend.claude.get_genner")
    @patch("src.backend.claude.anthropic.Anthropic")
    def test_claude_context_builds_genner(
        self,
        anthropic_cls: MagicMock,
        get_genner: MagicMock,
    ) -> None:
        fake_client = MagicMock()
        fake_genner = MagicMock(spec=Genner)
        anthropic_cls.return_value = fake_client
        get_genner.return_value = fake_genner

        context = resolve_backend_context(make_app_config("claude"))
        self.assertIsNotNone(context)

        with context as session:
            self.assertIs(session.client, fake_client)
            self.assertIs(session.genner, fake_genner)


if __name__ == "__main__":
    unittest.main()
