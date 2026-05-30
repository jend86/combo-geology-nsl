import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestQloraExports(unittest.TestCase):
    def test_resolve_training_export_format_auto_for_vllm(self) -> None:
        from src.train.qlora import resolve_training_export_format

        self.assertEqual(
            resolve_training_export_format("vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"),
            "lora",
        )

    def test_resolve_training_export_format_auto_for_llama(self) -> None:
        from src.train.qlora import resolve_training_export_format

        self.assertEqual(
            resolve_training_export_format(
                "llama:unsloth/Qwen2.5-Coder-7B-Instruct-GGUF"
            ),
            "lora",
        )

    def test_resolve_training_export_format_respects_llama_gguf_override(self) -> None:
        from src.train.qlora import resolve_training_export_format

        self.assertEqual(
            resolve_training_export_format(
                "llama:unsloth/Qwen2.5-Coder-7B-Instruct-GGUF",
                configured_format="gguf",
            ),
            "gguf",
        )

    @patch("src.train.qlora.logger.warning")
    def test_resolve_training_export_format_warns_for_vllm_merged_override(
        self,
        mock_warning: MagicMock,
    ) -> None:
        from src.train.qlora import resolve_training_export_format

        self.assertEqual(
            resolve_training_export_format(
                "vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
                configured_format="merged_16bit",
            ),
            "merged_16bit",
        )
        mock_warning.assert_called_once()


class TestLoadSftDataset(unittest.TestCase):
    def _make_jsonl(self, rows: list[dict], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def _make_tokenizer(self):
        tokenizer = MagicMock()

        def apply_chat_template(messages, tokenize, add_generation_prompt):
            user_content = messages[0]["content"]
            if len(messages) == 1:
                return f"<user>{user_content}</user><asst>"
            asst_content = messages[1]["content"]
            return f"<user>{user_content}</user><asst>{asst_content}</asst>"

        tokenizer.apply_chat_template.side_effect = apply_chat_template
        return tokenizer

    def _make_row(self, *, success: bool = True, **kwargs) -> dict:
        base = {
            "prompt": "Do something",
            "raw_response": "Done",
            "timestamp": "2026-04-06T00:00:00",
            "interaction_type": "orchestrator",
            "success": success,
            "error_message": None,
            "episode_id": "ep_gen0_0001_123",
            "generation_id": 0,
            "episode_score": 100.0,
        }
        base.update(kwargs)
        return base

    def test_load_sft_dataset_from_jsonl(self):
        from src.train.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row(), self._make_row()], path)
            dataset = _load_sft_dataset([path], tokenizer)
        self.assertGreater(len(dataset), 0)
        self.assertIn("prompt", dataset.column_names)
        self.assertIn("completion", dataset.column_names)
        self.assertNotIn("text", dataset.column_names)

    def test_load_sft_dataset_splits_prompt_and_completion(self):
        from src.train.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row(prompt="Question", raw_response="Answer")], path)
            dataset = _load_sft_dataset([path], tokenizer)

        self.assertEqual(dataset[0]["prompt"], "<user>Question</user><asst>")
        self.assertEqual(dataset[0]["completion"], "Answer</asst>")

    def test_load_sft_dataset_multiple_files(self):
        from src.train.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path0 = Path(tmpdir) / "gen0" / "sft_training_rows.jsonl"
            path1 = Path(tmpdir) / "gen1" / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row(generation_id=0)], path0)
            self._make_jsonl([self._make_row(generation_id=1)], path1)
            dataset = _load_sft_dataset([path0, path1], tokenizer)
        self.assertEqual(len(dataset), 2)

    def test_load_sft_dataset_empty_input(self):
        from src.train.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([], path)
            with self.assertRaises(ValueError, msg="No training rows found"):
                _load_sft_dataset([path], tokenizer)

    def test_load_sft_dataset_filters_unsuccessful(self):
        from src.train.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl(
                [self._make_row(success=True), self._make_row(success=False)], path
            )
            dataset = _load_sft_dataset([path], tokenizer)
        self.assertEqual(len(dataset), 1)


class TestQloraGpuCleanup(unittest.TestCase):
    @patch("src.train.qlora.FastLanguageModel.from_pretrained")
    @patch("src.train.qlora.gc.collect")
    def test_load_base_model_clears_torch_cuda_state_before_loading(
        self,
        mock_collect: MagicMock,
        mock_from_pretrained: MagicMock,
    ) -> None:
        from src.train.qlora import _load_base_model

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        expected_model = MagicMock()
        expected_tokenizer = MagicMock()
        mock_from_pretrained.return_value = (expected_model, expected_tokenizer)

        with patch.dict(sys.modules, {"torch": fake_torch}):
            model, tokenizer = _load_base_model("Qwen/Qwen2.5-Coder-7B-Instruct")

        mock_collect.assert_called_once()
        fake_torch.cuda.empty_cache.assert_called_once()
        self.assertIs(model, expected_model)
        self.assertIs(tokenizer, expected_tokenizer)

    @patch("src.train.qlora.FastLanguageModel.from_pretrained")
    @patch("src.train.qlora.gc.collect")
    def test_cleanup_torch_cuda_state_skips_when_cuda_unavailable(
        self,
        mock_collect: MagicMock,
        mock_from_pretrained: MagicMock,
    ) -> None:
        from src.train.qlora import _load_base_model

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        mock_from_pretrained.return_value = (MagicMock(), MagicMock())

        with patch.dict(sys.modules, {"torch": fake_torch}):
            _load_base_model("Qwen/Qwen2.5-Coder-7B-Instruct")

        mock_collect.assert_called_once()
        fake_torch.cuda.empty_cache.assert_not_called()


class TestTrainSft(unittest.TestCase):
    def _make_jsonl(self, rows: list[dict], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def _make_row(self) -> dict:
        return {
            "prompt": "Do something",
            "raw_response": "Done",
            "timestamp": "2026-04-06T00:00:00",
            "interaction_type": "orchestrator",
            "success": True,
            "error_message": None,
            "episode_id": "ep_gen0_0001_123",
            "generation_id": 0,
            "episode_score": 100.0,
        }

    @patch("src.train.qlora.SFTTrainer")
    @patch("src.train.qlora._attach_lora_adapter")
    @patch("src.train.qlora._load_base_model")
    def test_train_sft_runs_trainer(
        self, mock_load_base_model, mock_attach_lora_adapter, mock_sft_trainer_cls
    ):
        from src.train.qlora import train_sft

        model = MagicMock()
        tokenizer = MagicMock()

        def apply_chat_template(messages, tokenize, add_generation_prompt):
            return f"<text>{messages[0]['content']}</text>"

        tokenizer.apply_chat_template.side_effect = apply_chat_template
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model

        trainer_instance = MagicMock()
        mock_sft_trainer_cls.return_value = trainer_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"

            result = train_sft(
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                training_data_paths=[str(data_path)],
                output_dir=str(output_dir),
                max_steps=1,
            )

        trainer_instance.train.assert_called_once()
        trainer_args = mock_sft_trainer_cls.call_args.kwargs["args"]
        self.assertIs(trainer_args.completion_only_loss, True)
        model.save_pretrained.assert_called_once()
        tokenizer.save_pretrained.assert_called_once()
        model.save_pretrained_merged.assert_not_called()
        self.assertEqual(result, output_dir.resolve())

    @patch("src.train.qlora.SFTTrainer")
    @patch("src.train.qlora._attach_lora_adapter")
    @patch("src.train.qlora._load_base_model")
    def test_train_sft_loads_wandb_api_key_from_dotenv(
        self,
        mock_load_base_model,
        mock_attach_lora_adapter,
        mock_sft_trainer_cls,
    ):
        from src.train.qlora import train_sft

        model = MagicMock()
        tokenizer = MagicMock()

        def apply_chat_template(messages, tokenize, add_generation_prompt):
            return f"<text>{messages[0]['content']}</text>"

        tokenizer.apply_chat_template.side_effect = apply_chat_template
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        trainer_instance = MagicMock()
        mock_sft_trainer_cls.return_value = trainer_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"
            env_path = Path(tmpdir) / ".env"
            env_path.write_text('WANDB_API_KEY="dotenv-key"\n', encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                with patch("src.train.qlora.find_dotenv", return_value=str(env_path)):
                    result = train_sft(
                        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                        training_data_paths=[str(data_path)],
                        output_dir=str(output_dir),
                        max_steps=1,
                        wandb_project="demo-project",
                    )

                self.assertEqual("dotenv-key", os.environ["WANDB_API_KEY"])
                self.assertEqual("demo-project", os.environ["WANDB_PROJECT"])

        trainer_instance.train.assert_called_once()
        self.assertEqual(result, output_dir.resolve())

    def test_train_sft_wandb_project_still_errors_without_env_or_dotenv(self):
        from src.train.qlora import train_sft

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"

            with patch.dict(os.environ, {}, clear=True):
                with patch("src.train.qlora.find_dotenv", return_value=""):
                    with self.assertRaises(RuntimeError):
                        train_sft(
                            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                            training_data_paths=[str(data_path)],
                            output_dir=str(output_dir),
                            max_steps=1,
                            wandb_project="demo-project",
                        )

    @patch("src.train.qlora.SFTTrainer")
    @patch("src.train.qlora._attach_lora_adapter")
    @patch("src.train.qlora._load_base_model")
    def test_train_sft_merged_calls_save_pretrained_merged(
        self, mock_load_base_model, mock_attach_lora_adapter, mock_sft_trainer_cls
    ):
        from src.train.qlora import train_sft

        model = MagicMock()
        tokenizer = MagicMock()

        def apply_chat_template(messages, tokenize, add_generation_prompt):
            return f"<text>{messages[0]['content']}</text>"

        tokenizer.apply_chat_template.side_effect = apply_chat_template
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model

        trainer_instance = MagicMock()
        mock_sft_trainer_cls.return_value = trainer_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"

            result = train_sft(
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                training_data_paths=[str(data_path)],
                output_dir=str(output_dir),
                max_steps=1,
                export_format="merged_16bit",
            )

        trainer_instance.train.assert_called_once()
        model.save_pretrained_merged.assert_called_once()
        model.save_pretrained.assert_not_called()
        self.assertEqual(result, output_dir.resolve())

    @patch("src.train.qlora.SFTTrainer")
    @patch("src.train.qlora._save_training_artifact")
    @patch("src.train.qlora._attach_lora_adapter")
    @patch("src.train.qlora._load_base_model")
    def test_train_sft_passes_export_format_to_artifact_saver(
        self,
        mock_load_base_model,
        mock_attach_lora_adapter,
        mock_save_training_artifact,
        mock_sft_trainer_cls,
    ):
        from src.train.qlora import train_sft

        model = MagicMock()
        tokenizer = MagicMock()

        def apply_chat_template(messages, tokenize, add_generation_prompt):
            return f"<text>{messages[0]['content']}</text>"

        tokenizer.apply_chat_template.side_effect = apply_chat_template
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        mock_sft_trainer_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "artifact"
            mock_save_training_artifact.return_value = output_dir.resolve()

            train_sft(
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                training_data_paths=[str(data_path)],
                output_dir=str(output_dir),
                max_steps=1,
                export_format="merged_16bit",
            )

        mock_save_training_artifact.assert_called_once()
        self.assertEqual(
            mock_save_training_artifact.call_args.kwargs["export_format"],
            "merged_16bit",
        )

    @patch("src.train.qlora.SFTTrainer")
    @patch("src.train.qlora._save_training_artifact")
    @patch("src.train.qlora._attach_lora_adapter")
    @patch("src.train.qlora._load_base_model")
    def test_train_sft_writes_metadata(
        self,
        mock_load_base_model,
        mock_attach_lora_adapter,
        mock_save_training_artifact,
        mock_sft_trainer_cls,
    ):
        from src.train.qlora import train_sft

        model = MagicMock()
        tokenizer = MagicMock()

        def apply_chat_template(messages, tokenize, add_generation_prompt):
            return f"<text>{messages[0]['content']}</text>"

        tokenizer.apply_chat_template.side_effect = apply_chat_template
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        mock_sft_trainer_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row(), self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"
            mock_save_training_artifact.return_value = output_dir.resolve()

            train_sft(
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                training_data_paths=[str(data_path)],
                output_dir=str(output_dir),
                max_steps=1,
                learning_rate=1e-4,
            )

            metadata = json.loads(
                (output_dir.resolve() / "training_info.json").read_text()
            )

        self.assertEqual(metadata["base_model"], "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(metadata["learning_rate"], 1e-4)
        self.assertEqual(metadata["row_count"], 2)
        self.assertIn("training_data_paths", metadata)
        self.assertIn("exported_at", metadata)
        self.assertIn("max_steps", metadata)

    def test_train_sft_cli_multiple_training_data(self):
        """CLI should accept multiple --training-data arguments."""
        from src.train.qlora import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--base-model",
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                "--training-data",
                "./data/gen0/sft.jsonl",
                "--training-data",
                "./data/gen1/sft.jsonl",
                "--output",
                "./models/adapters/test",
            ]
        )
        self.assertEqual(len(args.training_data), 2)
        self.assertIn("./data/gen0/sft.jsonl", args.training_data)
        self.assertIn("./data/gen1/sft.jsonl", args.training_data)


if __name__ == "__main__":
    unittest.main()
