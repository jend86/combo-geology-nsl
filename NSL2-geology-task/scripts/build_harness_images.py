"""Build local harness images referenced by configs in config/*.toml.

Bridge to Approach B (see docs/design/harness-image-provisioning.md).
Once B lands, entry points provision automatically and this script
becomes optional.

Two discovery paths, in priority order:

1. **Config-driven** (forward-compat with B): if a config declares
   `[harness.container.build]` with a context path, build
   `harness.container.image` from that context.
2. **Known-image fallback**: today's configs carry only
   `harness.container.image` (no build block). Match that tag against
   KNOWN_LOCAL_IMAGES below to recover the build context.

Once Approach B ships and configs grow proper build sections, path 1
covers everything and path 2 can be dropped.

Usage:
    uv run python scripts/build_harness_images.py
    uv run python scripts/build_harness_images.py --config config/config-forked-exploit-msagent.toml
    uv run python scripts/build_harness_images.py --rebuild   # forces --no-cache
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

from src.harness.provisioning import (
    _BUILD_CONTEXT_SHA_BUILD_ARG,
    _hash_build_context,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fallback for pre-B configs that reference an image but have no build
# block. Key = image tag as it appears in the TOML; value = (context
# dir relative to repo root, build_args dict). Drop entries here as the
# configs migrate to Approach B.
KNOWN_LOCAL_IMAGES: dict[str, tuple[str, dict[str, str]]] = {
    "ghcr.io/nsl2/ms-agent:0.1.0": ("docker/ms-agent", {}),
    "nsl/ms-agent:0.1.0": ("docker/ms-agent", {}),
}


def _harness_builds(
    config_path: Path,
) -> list[tuple[str, Path, dict[str, str], str]]:
    """Return a list of (image, context_dir, build_args, source) tuples
    to build for this config. `source` is "config" or "fallback" for
    logging."""
    with config_path.open("rb") as f:
        doc = tomllib.load(f)
    harness = doc.get("harness", {})
    if harness.get("name") != "container":
        return []
    container = harness.get("container")
    if not container:
        return []
    image = container.get("image")
    if not image:
        return []

    build = container.get("build")
    if build is not None:
        ctx = PROJECT_ROOT / build["context"]
        args = build.get("build_args", {}) or {}
        return [(image, ctx, args, "config")]

    if image in KNOWN_LOCAL_IMAGES:
        ctx_rel, args = KNOWN_LOCAL_IMAGES[image]
        return [(image, PROJECT_ROOT / ctx_rel, args, "fallback")]

    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        action="append",
        default=None,
        help="Config file to parse (repeatable). Defaults to all config/*.toml.",
    )
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="Pass --no-cache to docker build to force a layer-level rebuild.",
    )
    args = ap.parse_args()

    paths = (
        [Path(p).resolve() for p in args.config]
        if args.config
        else sorted((PROJECT_ROOT / "config").glob("*.toml"))
    )
    if not paths:
        print("no config files found", file=sys.stderr)
        return 1

    seen: set[str] = set()
    for cfg in paths:
        try:
            builds = _harness_builds(cfg)
        except Exception as exc:  # malformed TOML, etc.
            print(f"skipping {cfg.name}: {exc}", file=sys.stderr)
            continue
        for image, ctx, build_args, source in builds:
            if image in seen:
                continue
            seen.add(image)
            if not ctx.is_dir():
                print(
                    f"ERROR: context {ctx} for {image} "
                    f"(from {cfg.name}, source={source}) is not a directory",
                    file=sys.stderr,
                )
                return 1
            cmd = ["docker", "build", "-t", image]
            if args.rebuild:
                cmd.append("--no-cache")
            # Auto-inject the build-context content hash so the resulting
            # image's nsl.build-context-sha label matches what
            # ensure_harness_image's drift detector will compute on the
            # next run. Without this, a fresh build via this script would
            # leave the label as the Dockerfile default ("unknown") and
            # the next runtime invocation would think the image is stale
            # and trigger a spurious rebuild.
            try:
                build_args = {
                    **build_args,
                    _BUILD_CONTEXT_SHA_BUILD_ARG: _hash_build_context(ctx),
                }
            except OSError as exc:
                print(
                    f"warning: could not hash build context {ctx}: {exc}",
                    file=sys.stderr,
                )
            for k, v in build_args.items():
                cmd += ["--build-arg", f"{k}={v}"]
            cmd.append(str(ctx))
            print(
                f"[{cfg.name} / {source}] building {image} from {ctx}"
                + (" (no-cache)" if args.rebuild else "")
            )
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(
                    f"docker build failed for {image} (exit {result.returncode})",
                    file=sys.stderr,
                )
                return result.returncode

    if not seen:
        print(
            "no harness images found to build. Configs either don't use the "
            "container harness or reference images not in KNOWN_LOCAL_IMAGES.",
            file=sys.stderr,
        )
        return 0

    print(f"\nbuilt {len(seen)} image(s): {sorted(seen)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
