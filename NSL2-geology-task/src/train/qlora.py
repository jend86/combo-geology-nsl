from __future__ import annotations

import argparse
import gc
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence, cast

from datasets import Dataset, load_dataset
from dotenv import dotenv_values, find_dotenv
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
        lora_dropout=config.get("dropout", 0),
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


def _load_self_generated_sft_rows(
    training_data_paths: Sequence[Path], tokenizer: Any
) -> list[dict[str, str]]:
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
    return rows


def _load_rehearsal_sft_rows(
    *,
    dataset_name: str,
    split: str,
    text_field: str,
    rows_per_epoch: int,
    virtual_epochs: int,
    seed: int,
    prompt_chars: int,
    max_chars: int,
) -> list[list[dict[str, str]]]:
    if rows_per_epoch <= 0:
        return [[] for _ in range(virtual_epochs)]

    rehearsal_dataset = load_dataset(dataset_name, split=split)
    valid_texts: list[str] = []
    for row in rehearsal_dataset:
        value = row.get(text_field)
        if not isinstance(value, str):
            continue
        text = " ".join(value.split())
        if text:
            valid_texts.append(text[:max_chars])

    if not valid_texts:
        raise ValueError(
            f"No rehearsal text found in dataset '{dataset_name}' field '{text_field}'"
        )

    rows_by_epoch: list[list[dict[str, str]]] = []
    for epoch in range(virtual_epochs):
        rng = random.Random(seed + epoch)
        if rows_per_epoch <= len(valid_texts):
            selected_texts = rng.sample(valid_texts, rows_per_epoch)
        else:
            selected_texts = [rng.choice(valid_texts) for _ in range(rows_per_epoch)]

        epoch_rows: list[dict[str, str]] = []
        for text in selected_texts:
            excerpt = text[:prompt_chars].rstrip()
            prompt = "Continue the following geoscience passage"
            if excerpt:
                prompt += f":\n\n{excerpt}"
            else:
                prompt += "."
            epoch_rows.append({"prompt": prompt, "completion": text})
        rows_by_epoch.append(epoch_rows)

    return rows_by_epoch


def _load_sft_dataset(
    training_data_paths: Sequence[Path],
    tokenizer: Any,
    *,
    virtual_epochs: int = 1,
    rehearsal_dataset: str | None = None,
    rehearsal_split: str = "train",
    rehearsal_text_field: str = "text",
    rehearsal_rows_per_epoch: int = 0,
    rehearsal_seed: int = 3407,
    rehearsal_prompt_chars: int = 256,
    rehearsal_max_chars: int = 2048,
) -> Any:
    """Load generation JSONL files plus optional rehearsal rows into an SFT Dataset."""

    if virtual_epochs < 1:
        raise ValueError("virtual_epochs must be at least 1")

    rows = _load_self_generated_sft_rows(training_data_paths, tokenizer)

    if not rows:
        raise ValueError("No training rows found")

    rehearsal_rows_by_epoch: list[list[dict[str, str]]]
    if rehearsal_dataset:
        rehearsal_rows_by_epoch = _load_rehearsal_sft_rows(
            dataset_name=rehearsal_dataset,
            split=rehearsal_split,
            text_field=rehearsal_text_field,
            rows_per_epoch=rehearsal_rows_per_epoch,
            virtual_epochs=virtual_epochs,
            seed=rehearsal_seed,
            prompt_chars=rehearsal_prompt_chars,
            max_chars=rehearsal_max_chars,
        )
    else:
        rehearsal_rows_by_epoch = [[] for _ in range(virtual_epochs)]

    expanded_rows: list[dict[str, str]] = []
    for epoch in range(virtual_epochs):
        expanded_rows.extend(rows)
        expanded_rows.extend(rehearsal_rows_by_epoch[epoch])

    return Dataset.from_list(expanded_rows)


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
    warmup_ratio: float = 0.0,
    lr_scheduler_type: str = "linear",
    weight_decay: float = 0.0,
    logging_steps: int = 1,
    num_train_epochs: int = 1,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    seed: int = 42,
    rehearsal_dataset: str | None = None,
    rehearsal_split: str = "train",
    rehearsal_text_field: str = "text",
    rehearsal_rows_per_epoch: int = 0,
    rehearsal_seed: int | None = None,
    rehearsal_prompt_chars: int = 256,
    rehearsal_max_chars: int = 2048,
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
    model = _attach_lora_adapter(
        model,
        {"rank": lora_rank, "alpha": lora_alpha, "dropout": lora_dropout},
    )

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    if export_format != "gguf":
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_training_paths = [
        str(Path(path).expanduser().resolve()) for path in training_data_paths
    ]
    dataset = _load_sft_dataset(
        [Path(path) for path in resolved_training_paths],
        tokenizer,
        virtual_epochs=num_train_epochs,
        rehearsal_dataset=rehearsal_dataset,
        rehearsal_split=rehearsal_split,
        rehearsal_text_field=rehearsal_text_field,
        rehearsal_rows_per_epoch=rehearsal_rows_per_epoch,
        rehearsal_seed=rehearsal_seed if rehearsal_seed is not None else seed,
        rehearsal_prompt_chars=rehearsal_prompt_chars,
        rehearsal_max_chars=rehearsal_max_chars,
    )
    row_count = len(dataset)
    effective_warmup_steps = 0 if warmup_ratio > 0 else warmup_steps

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(resolved_output_dir),
            max_steps=max_steps,
            num_train_epochs=1,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=effective_warmup_steps,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            save_strategy="no",
            report_to=report_to,
            max_length=max_seq_length,
            completion_only_loss=True,
            seed=seed,
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
            "configured_num_train_epochs": num_train_epochs,
            "virtual_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "warmup_steps": effective_warmup_steps,
            "warmup_ratio": warmup_ratio,
            "lr_scheduler_type": lr_scheduler_type,
            "weight_decay": weight_decay,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "seed": seed,
            "row_count": row_count,
            "training_data_paths": resolved_training_paths,
            "rehearsal_dataset": rehearsal_dataset,
            "rehearsal_split": rehearsal_split,
            "rehearsal_text_field": rehearsal_text_field,
            "rehearsal_rows_per_epoch": rehearsal_rows_per_epoch,
            "rehearsal_seed": rehearsal_seed if rehearsal_seed is not None else seed,
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
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rehearsal-dataset", default=None)
    parser.add_argument("--rehearsal-split", default="train")
    parser.add_argument("--rehearsal-text-field", default="text")
    parser.add_argument("--rehearsal-rows-per-epoch", type=int, default=0)
    parser.add_argument("--rehearsal-seed", type=int, default=None)
    parser.add_argument("--rehearsal-prompt-chars", type=int, default=256)
    parser.add_argument("--rehearsal-max-chars", type=int, default=2048)
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
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        rehearsal_dataset=args.rehearsal_dataset,
        rehearsal_split=args.rehearsal_split,
        rehearsal_text_field=args.rehearsal_text_field,
        rehearsal_rows_per_epoch=args.rehearsal_rows_per_epoch,
        rehearsal_seed=args.rehearsal_seed,
        rehearsal_prompt_chars=args.rehearsal_prompt_chars,
        rehearsal_max_chars=args.rehearsal_max_chars,
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
