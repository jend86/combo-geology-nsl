from __future__ import annotations

import argparse

import pydantic
import toml
from dotenv import load_dotenv

# override=True so the project's .env is authoritative. Without it, a stale
# OPENROUTER_API_KEY already exported in the shell shadows .env — e.g. an old,
# spend-capped key silently wins and every inference call 403s.
load_dotenv(override=True)

from src.execution import open_backend_runtime, run_generation_sequential
from src.harness.provisioning import ensure_configured_harness
from src.helper import unflatten_toml_dict
from src.typing.config import AppConfig


def _load_config(config_file: str) -> AppConfig:
    with open(config_file, "r", encoding="utf-8") as handle:
        config_dict = toml.load(handle)
    try:
        return AppConfig(**unflatten_toml_dict(config_dict))
    except pydantic.ValidationError as exc:
        raise RuntimeError(f"Config validation error: {exc}") from exc


def main(
    config_file: str = "./config/config-container.toml",
    *,
    rebuild_harness: bool = False,
) -> None:
    config = _load_config(config_file)
    ensure_configured_harness(config, rebuild=rebuild_harness)
    with open_backend_runtime(config) as rt:
        run_generation_sequential(rt, generation_id=0, target_rows=1, max_episodes=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="./config/config-container.toml",
        help="Path to the config TOML.",
    )
    parser.add_argument(
        "--rebuild-harness",
        action="store_true",
        help="Force a no-cache rebuild of the harness image, even if an image with the same tag already exists locally. **Required** after editing docker/<harness>/Dockerfile or anything inside its build context - otherwise the cached image is reused silently and your changes do not run. No-op for configs without a [harness.container.build] block (image is pulled, not built). Escape hatch: `docker rmi <image-tag>` then re-run. See docs/design/harness-image-provisioning.md.",
    )
    cli_args = parser.parse_args()
    main(config_file=cli_args.config, rebuild_harness=cli_args.rebuild_harness)
