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
        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                user_content = messages[0]["content"]
                if len(messages) == 1:
                    return f"<user>{user_content}</user><asst>"
                asst_content = messages[1]["content"]
                return f"<user>{user_content}</user><asst>{asst_content}</asst>"

            def __call__(self, text=None, **_kwargs):
                assert isinstance(text, str)
                return {
                    "input_ids": [ord(char) for char in text],
                    "attention_mask": [1] * len(text),
                }

        return FakeTokenizer()

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
        self.assertIn("input_ids", dataset.column_names)
        self.assertIn("completion_mask", dataset.column_names)
        self.assertNotIn("text", dataset.column_names)
        self.assertNotIn("prompt", dataset.column_names)
        self.assertNotIn("completion", dataset.column_names)

    def test_load_sft_dataset_templates_and_masks_response(self):
        from src.train.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row(prompt="Question", raw_response="Answer")], path)
            dataset = _load_sft_dataset([path], tokenizer)

        full_text = "<user>Question</user><asst>Answer</asst>"
        prompt_text = "<user>Question</user><asst>"
        self.assertEqual(dataset[0]["input_ids"], [ord(char) for char in full_text])
        self.assertEqual(
            dataset[0]["completion_mask"][: len(prompt_text)],
            [0] * len(prompt_text),
        )
        self.assertEqual(
            dataset[0]["completion_mask"][len(prompt_text) :],
            [1] * (len(full_text) - len(prompt_text)),
        )

    def test_build_query_response_sft_row_rejects_non_prefix_template(self):
        from src.train.qlora import _build_query_response_sft_row

        tokenizer = self._make_tokenizer()
        original = tokenizer.apply_chat_template

        def bad_template(messages, tokenize, add_generation_prompt):
            if len(messages) == 1:
                return "<other-prefix>"
            return original(messages, tokenize, add_generation_prompt)

        tokenizer.apply_chat_template = bad_template

        with self.assertRaises(ValueError):
            _build_query_response_sft_row(
                tokenizer,
                prompt="Question",
                raw_response="Answer",
            )

    def test_completion_only_labels_mask_query_tokens(self):
        from src.train.qlora import IGNORE_INDEX, _build_completion_only_labels

        labels = _build_completion_only_labels(
            input_ids=[10, 11, 12, 13],
            completion_mask=[0, 0, 1, 1],
        )

        self.assertEqual(labels, [IGNORE_INDEX, IGNORE_INDEX, 12, 13])

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

    def test_load_sft_dataset_expands_virtual_epochs_with_rehearsal(self):
        from src.train.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl(
                [self._make_row(prompt="Kazakhstan task", raw_response="Hypothesis")],
                path,
            )
            rehearsal = [
                {
                    "text": (
                        f"Geology rehearsal passage {i}. "
                        "Sediment, structure, basin, and mineral systems interact. "
                        "This sentence provides enough continuation text."
                    )
                }
                for i in range(8)
            ]

            with patch("src.train.qlora.load_dataset", return_value=rehearsal) as mock_load:
                dataset = _load_sft_dataset(
                    [path],
                    tokenizer,
                    virtual_epochs=2,
                    rehearsal_dataset="ClickNoow/5k-dataset-geogpt-fineweb",
                    rehearsal_split="train",
                    rehearsal_text_field="text",
                    rehearsal_rows_per_epoch=2,
                    rehearsal_seed=11,
                    rehearsal_prompt_chars=32,
                    rehearsal_max_chars=160,
                )

        mock_load.assert_called_once_with(
            "ClickNoow/5k-dataset-geogpt-fineweb",
            split="train",
        )
        self.assertEqual(len(dataset), 6)
        texts = ["".join(chr(token_id) for token_id in row["input_ids"]) for row in dataset]
        self.assertEqual(
            sum(text == "<user>Kazakhstan task</user><asst>Hypothesis</asst>" for text in texts),
            2,
        )
        rehearsal_texts = [text for text in texts if "Continue the following" in text]
        self.assertEqual(len(rehearsal_texts), 4)
        self.assertTrue(
            all("Geology rehearsal passage" in text for text in rehearsal_texts)
        )
        self.assertTrue(all(1 in row["completion_mask"] for row in dataset))

    @patch("src.train.qlora.logger.warning")
    def test_warn_if_dataset_would_truncate_suggests_preserving_length(
        self,
        mock_warning: MagicMock,
    ) -> None:
        from src.train.qlora import _warn_if_dataset_would_truncate

        dataset = [
            {"input_ids": [1, 2]},
            {"input_ids": [1, 2, 3, 4, 5, 6]},
        ]

        _warn_if_dataset_would_truncate(dataset, MagicMock(), max_seq_length=4)

        mock_warning.assert_called_once()
        warning = mock_warning.call_args.args[0]
        self.assertIn("would truncate", warning)
        self.assertIn("max_seq_length=4", warning)
        self.assertIn("at least 6", warning)

    @patch("src.train.qlora.logger.warning")
    def test_warn_if_dataset_would_truncate_stays_quiet_when_preserved(
        self,
        mock_warning: MagicMock,
    ) -> None:
        from src.train.qlora import _warn_if_dataset_would_truncate

        dataset = [{"input_ids": [1, 2, 3]}]

        _warn_if_dataset_would_truncate(dataset, MagicMock(), max_seq_length=3)

        mock_warning.assert_not_called()


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

    def _make_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2

        def apply_chat_template(messages, tokenize, add_generation_prompt):
            user_content = messages[0]["content"]
            if len(messages) == 1:
                return f"<user>{user_content}</user><asst>"
            asst_content = messages[1]["content"]
            return f"<user>{user_content}</user><asst>{asst_content}</asst>"

        def tokenize(text=None, **_kwargs):
            assert isinstance(text, str)
            return {
                "input_ids": [ord(char) for char in text],
                "attention_mask": [1] * len(text),
            }

        tokenizer.apply_chat_template.side_effect = apply_chat_template
        tokenizer.side_effect = tokenize
        return tokenizer

    def _run_train_sft(self, train_sft, **kwargs):
        with patch.dict(sys.modules, {"unsloth": MagicMock()}):
            return train_sft(**kwargs)

    @patch("src.train.qlora.SFTTrainer")
    @patch("src.train.qlora._attach_lora_adapter")
    @patch("src.train.qlora._load_base_model")
    def test_train_sft_runs_trainer(
        self, mock_load_base_model, mock_attach_lora_adapter, mock_sft_trainer_cls
    ):
        from src.train.qlora import train_sft

        model = MagicMock()
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model

        trainer_instance = MagicMock()
        mock_sft_trainer_cls.return_value = trainer_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"

            result = self._run_train_sft(
                train_sft,
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                training_data_paths=[str(data_path)],
                output_dir=str(output_dir),
                max_steps=1,
            )

        trainer_instance.train.assert_called_once()
        trainer_args = mock_sft_trainer_cls.call_args.kwargs["args"]
        self.assertTrue(trainer_args.completion_only_loss)
        self.assertEqual(trainer_args.dataset_kwargs, {"skip_prepare_dataset": True})
        self.assertFalse(trainer_args.remove_unused_columns)
        trainer_dataset = mock_sft_trainer_cls.call_args.kwargs["train_dataset"]
        self.assertIn("input_ids", trainer_dataset.column_names)
        self.assertIn("completion_mask", trainer_dataset.column_names)
        self.assertIn("data_collator", mock_sft_trainer_cls.call_args.kwargs)
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
        tokenizer = self._make_tokenizer()
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
                    result = self._run_train_sft(
                        train_sft,
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
                        self._run_train_sft(
                            train_sft,
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
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model

        trainer_instance = MagicMock()
        mock_sft_trainer_cls.return_value = trainer_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"

            result = self._run_train_sft(
                train_sft,
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
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        mock_sft_trainer_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "artifact"
            mock_save_training_artifact.return_value = output_dir.resolve()

            self._run_train_sft(
                train_sft,
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
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        mock_sft_trainer_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row(), self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"
            mock_save_training_artifact.return_value = output_dir.resolve()

            self._run_train_sft(
                train_sft,
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

    @patch("src.train.qlora.SFTTrainer")
    @patch("src.train.qlora._save_training_artifact")
    @patch("src.train.qlora._attach_lora_adapter")
    @patch("src.train.qlora._load_base_model")
    def test_train_sft_uses_virtual_epoch_rehearsal_and_sft_knobs(
        self,
        mock_load_base_model,
        mock_attach_lora_adapter,
        mock_save_training_artifact,
        mock_sft_trainer_cls,
    ):
        from src.train.qlora import train_sft

        model = MagicMock()
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        mock_sft_trainer_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"
            mock_save_training_artifact.return_value = output_dir.resolve()
            rehearsal = [
                {
                    "text": (
                        f"Rehearsal passage {i}. "
                        "Geological context and environmental process text continue here."
                    )
                }
                for i in range(8)
            ]

            with patch("src.train.qlora.load_dataset", return_value=rehearsal):
                self._run_train_sft(
                    train_sft,
                    base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                    training_data_paths=[str(data_path)],
                    output_dir=str(output_dir),
                    max_steps=-1,
                    num_train_epochs=2,
                    gradient_accumulation_steps=16,
                    learning_rate=1e-4,
                    warmup_ratio=0.03,
                    lr_scheduler_type="linear",
                    weight_decay=0.001,
                    lora_rank=32,
                    lora_alpha=32,
                    lora_dropout=0.0,
                    seed=3407,
                    rehearsal_dataset="ClickNoow/5k-dataset-geogpt-fineweb",
                    rehearsal_rows_per_epoch=2,
                    rehearsal_seed=123,
                )

            metadata = json.loads(
                (output_dir.resolve() / "training_info.json").read_text()
            )

        mock_attach_lora_adapter.assert_called_once()
        self.assertEqual(mock_attach_lora_adapter.call_args.args[1]["rank"], 32)
        self.assertEqual(mock_attach_lora_adapter.call_args.args[1]["alpha"], 32)
        trainer_dataset = mock_sft_trainer_cls.call_args.kwargs["train_dataset"]
        self.assertEqual(len(trainer_dataset), 6)
        trainer_args = mock_sft_trainer_cls.call_args.kwargs["args"]
        self.assertEqual(trainer_args.max_steps, -1)
        self.assertEqual(trainer_args.num_train_epochs, 1)
        self.assertEqual(trainer_args.gradient_accumulation_steps, 16)
        self.assertEqual(trainer_args.learning_rate, 1e-4)
        self.assertEqual(trainer_args.warmup_ratio, 0.03)
        self.assertEqual(trainer_args.lr_scheduler_type.value, "linear")
        self.assertEqual(trainer_args.weight_decay, 0.001)
        self.assertEqual(metadata["configured_num_train_epochs"], 2)
        self.assertEqual(metadata["virtual_epochs"], 2)
        self.assertEqual(metadata["row_count"], 6)
        self.assertEqual(metadata["rehearsal_rows_per_epoch"], 2)

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
