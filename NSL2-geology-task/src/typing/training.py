import datetime
import json
import os
import uuid
from pathlib import Path
from typing import Any, Literal, TypeAlias


TrainData: TypeAlias = list[dict[str, Any]]
TrainDataType: TypeAlias = Literal["episode_v2"]


def save_train_data(
    train_data_type: TrainDataType | str,
    train_data: TrainData,
    folder_path: str | Path,
) -> None:
    folder_path = Path(folder_path)

    for train_data_item in train_data:
        cur_datetime = datetime.datetime.now()
        formatted_datetime = cur_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        run_id = train_data_item["run_id"]
        version = train_data_item["version"]

        filename = f"{version}_{run_id}_{train_data_type}_{formatted_datetime}.json"
        file_path = folder_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(train_data_item, handle, indent=4)


def save_generation_training_data(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def append_episode_jsonl(
    episode: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(episode, default=str) + "\n")


def save_generation_checkpoint(
    checkpoint: dict[str, Any],
    output_path: Path,
) -> None:
    # Atomic write: a partial checkpoint.json must never be observable on disk
    # (the live process reloads it on resume, and the 60s remote->local rsync
    # mirror can capture it mid-write). Write a temp sibling then os.replace —
    # the same pattern as run_train_loop.py's _write_run_doc.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(checkpoint, handle, indent=2, default=str)
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_generation_checkpoint(output_path: Path) -> dict[str, Any] | None:
    if not output_path.exists():
        return None
    with open(output_path, "r", encoding="utf-8") as handle:
        return json.load(handle)
