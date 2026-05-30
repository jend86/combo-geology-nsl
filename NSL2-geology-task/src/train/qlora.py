from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence, cast

from datasets import Dataset
from dotenv import dotenv_values, find_dotenv, load_dotenv
from loguru import logger
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer


DEFAULT_MAX_SEQ_LENGTH = 2048
TrainingExportFormat = Literal["lora", "merged_16bit", "gguf"]
ConfiguredTrainingExportFormat = Literal["auto", "lora", "merged_16bit", "gguf"]
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Fully-qualified regex that scopes LoRA to the language model only.
# VLMs like Gemma 4 (Gemma4ForConditionalGeneration) expose vision_tower.* modules
# with the same projection suffixes; matching by suffix attaches LoRA to those too,
# and vLLM rejects such adapters (vision tower keys are not in its expected set).
LORA_TARGET_MODULES_LM_ONLY_REGEX = (
    r"^(?!.*vision_tower).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)

SUPPORTED_TRAINING_BASE_MODEL_EXAMPLES = (
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
)


class _FastLanguageModelProxy:
    """Import Unsloth only when training actually loads or patches the model."""

    def from_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        from unsloth import FastLanguageModel as _FastLanguageModel

        return _FastLanguageModel.from_pretrained(*args, **kwargs)

    def get_peft_model(self, *args: Any, **kwargs: Any) -> Any:
        from unsloth import FastLanguageModel as _FastLanguageModel

        return _FastLanguageModel.get_peft_model(*args, **kwargs)


FastLanguageModel = _FastLanguageModelProxy()


def _export_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_training_base_model(base_model: str) -> None:
    lower_model = base_model.lower()
    invalid_format = None
    if lower_model.endswith(".gguf") or "gguf" in lower_model:
        invalid_format = "GGUF"
    elif "awq" in lower_model:
        invalid_format = "AWQ"
    elif "gptq" in lower_model:
        invalid_format = "GPTQ"

    if invalid_format is None:
        return

    examples = ", ".join(
        repr(example) for example in SUPPORTED_TRAINING_BASE_MODEL_EXAMPLES
    )
    raise ValueError(
        f"Unsupported training base model '{base_model}': {invalid_format} is an inference/export format, not a Transformers training checkpoint. "
        f"Use a Transformers or Unsloth BnB model instead, for example {examples}."
    )


def _load_wandb_api_key_from_dotenv() -> str | None:
    dotenv_path = find_dotenv(usecwd=True)
    if not dotenv_path:
        try:
            load_dotenv()
        except Exception:
            return None

    value = dotenv_values(dotenv_path).get("WANDB_API_KEY")
    if not isinstance(value, str) or not value:
        return None
    return value


def _cleanup_torch_cuda_state() -> None:
    gc.collect()

    import torch

    if not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()


def _mixed_precision_kwargs() -> dict[str, bool]:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return {"bf16": False, "fp16": False}

    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return {"bf16": bf16, "fp16": not bf16}


def _load_base_model(
    base_model: str,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
):
    _validate_training_base_model(base_model)
    _cleanup_torch_cuda_state()

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            device_map="balanced",
        )
    except RuntimeError as exc:
        if "No config file found" in str(exc):
            examples = ", ".join(
                repr(example) for example in SUPPORTED_TRAINING_BASE_MODEL_EXAMPLES
            )
            raise RuntimeError(
                f"Failed to load training base model '{base_model}'. "
                f"Use a Transformers-compatible training checkpoint or an Unsloth BnB checkpoint, for example {examples}."
            ) from exc
        raise
    return model, tokenizer


def _attach_lora_adapter(model: Any, config: dict[str, Any]) -> Any:
    return FastLanguageModel.get_peft_model(
        model,
        r=config.get("rank", 32),
        target_modules=config.get("target_modules", LORA_TARGET_MODULES_LM_ONLY_REGEX),
        lora_alpha=config.get("alpha", 32),
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_training_export_format(
    model_name: str,
    configured_format: ConfiguredTrainingExportFormat = "auto",
) -> TrainingExportFormat:
    backend = model_name.strip().split(":", 1)[0]
    if configured_format == "auto":
        return "lora"

    if backend == "llama" and configured_format == "merged_16bit":
        raise ValueError(
            "llama backend does not consume merged_16bit training artifacts. "
            "Use export_format='lora' for adapters or export_format='gguf' "
            "for a merged GGUF override."
        )

    if backend == "vllm" and configured_format == "gguf":
        raise ValueError(
            "vLLM does not consume GGUF training artifacts. Use export_format='lora' "
            "or export_format='merged_16bit'."
        )

    if backend == "vllm" and configured_format == "merged_16bit":
        logger.warning(
            "vLLM merged_16bit override selected. This will usually be slower than "
            "serving the AWQ base model with a LoRA adapter."
        )

    return cast(TrainingExportFormat, configured_format)


def _save_training_artifact(
    model: Any,
    tokenizer: Any,
    output_path: Path,
    *,
    export_format: TrainingExportFormat,
    gguf_quantize: str = "f16",
) -> Path:

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if export_format == "gguf":
        model.save_pretrained_gguf(
            output_path, tokenizer, quantization_method=gguf_quantize
        )
        return output_path

    assert export_format in ["lora", "merged_16bit", "merged_4bit"], (
        f"Unexpected export format {export_format} for finetuned model"
    )

    if export_format == "lora":
        # NOTE: unsloth/FastLanguageModel's save_pretrained_merged is bugged.
        # Use PEFT's save_pretrained directly — save_pretrained_merged always
        # merges LoRA into the base weights (downloading full safetensors),
        # even when save_method="lora".
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
    else:
        model.save_pretrained_merged(output_path, tokenizer, save_method=export_format)
    return output_path


def _training_metadata_path(artifact_path: Path) -> Path:
    if artifact_path.suffix == ".gguf":
        return artifact_path.with_suffix(".training_info.json")
    return artifact_path / "training_info.json"


def _load_sft_dataset(training_data_paths: Sequence[Path], tokenizer: Any) -> Any:
    """Load one or more generation JSONL files into a chat-formatted Dataset."""

    rows: list[dict[str, str]] = []
    for training_data_path in training_data_paths:
        resolved_path = Path(training_data_path).expanduser().resolve()
        with open(resolved_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                payload = json.loads(line)
                if not payload.get("success"):
                    continue

                prompt = payload.get("prompt")
                raw_response = payload.get("raw_response")
                if not isinstance(prompt, str) or not isinstance(raw_response, str):
                    continue

                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                completion_text = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": raw_response},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )[len(prompt_text):]
                rows.append({"prompt": prompt_text, "completion": completion_text})

    if not rows:
        raise ValueError("No training rows found")

    return Dataset.from_list(rows)


def train_sft(
    base_model: str,
    training_data_paths: Sequence[str],
    output_dir: str,
    *,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    max_steps: int = 50,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 5,
    logging_steps: int = 1,
    wandb_project: str | None = None,
    export_format: TrainingExportFormat = "lora",
    gguf_quantize: str = "f16",
) -> Path:
    """Run SFT on the provided generation window and save a trained LoRA adapter."""

    if wandb_project:
        if not os.environ.get("WANDB_API_KEY"):
            dotenv_api_key = _load_wandb_api_key_from_dotenv()
            if dotenv_api_key:
                os.environ["WANDB_API_KEY"] = dotenv_api_key
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError(
                "wandb_project is set but WANDB_API_KEY is not in the environment. "
                "Export WANDB_API_KEY, or unset wandb_project to disable W&B tracking."
            )
        os.environ["WANDB_PROJECT"] = wandb_project
        report_to = "wandb"
    else:
        report_to = "none"

    model, tokenizer = _load_base_model(
        base_model,
        max_seq_length=max_seq_length,
    )
    model = _attach_lora_adapter(model, {})

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    if export_format != "gguf":
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_training_paths = [
        str(Path(path).expanduser().resolve()) for path in training_data_paths
    ]
    dataset = _load_sft_dataset(
        [Path(path) for path in resolved_training_paths],
        tokenizer,
    )
    row_count = len(dataset)

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(resolved_output_dir),
            max_steps=max_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_strategy="no",
            report_to=report_to,
            max_length=max_seq_length,
            completion_only_loss=True,
            **_mixed_precision_kwargs(),
            optim="paged_adamw_8bit",
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    artifact_path = _save_training_artifact(
        model,
        tokenizer,
        resolved_output_dir,
        export_format=export_format,
        gguf_quantize=gguf_quantize,
    )
    _write_metadata(
        _training_metadata_path(artifact_path),
        {
            "base_model": base_model,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "row_count": row_count,
            "training_data_paths": resolved_training_paths,
            "export_format": export_format,
            "wandb_project": wandb_project,
            "exported_at": _export_timestamp(),
        },
    )
    return artifact_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QLoRA export helpers for the NSL finetuning pipeline."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--training-data", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument(
        "--export-format",
        choices=("lora", "merged_16bit", "gguf"),
        default="lora",
    )
    parser.add_argument("--quantize", default="f16")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if not args.training_data:
        parser.error("--training-data is required")

    output_path = train_sft(
        base_model=args.base_model,
        training_data_paths=args.training_data,
        output_dir=args.output,
        max_seq_length=args.max_seq_length,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        wandb_project=args.wandb_project,
        export_format=args.export_format,
        gguf_quantize=args.quantize,
    )

    print(output_path)
    return 0


__all__ = [
    "resolve_training_export_format",
    "train_sft",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
