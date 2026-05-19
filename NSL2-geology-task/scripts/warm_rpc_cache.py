"""Pre-warm Foundry's RPC cache for a variation by running its DeFiHackLabs PoC.

Foundry caches fork reads under ~/.foundry/cache/rpc/<chain>/<block>/. After
running the original exploit under `forge test`, most state the agent will
later touch is already on disk. This script copies the populated cache into
the variation's `rpc_cache/` for shipping.

The PoC is compiled inside a *temporary per-variation foundry project*
assembled from DeFiHackLabs sources (the shared siblings — basetest.sol,
interface.sol, tokenhelper.sol — plus the PoC's year-month directory), with
a local foundry.toml wiring RPC endpoints to our paid Alchemy keys. This
avoids DeFiHackLabs' workspace-wide compilation, where unrelated broken
PoCs would otherwise fail the build.

Usage:
    RPC_URL_BSC=... uv run python scripts/warm_rpc_cache.py \\
        --variation pdz_flashloan_price_manip_2025_08 \\
        --defihacklabs-root /home/elijah/DeFiHackLabs
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


SHARED_SIBLINGS = ("basetest.sol", "interface.sol", "tokenhelper.sol")


def assemble_project(
    proj: Path,
    defihacklabs_root: Path,
    source_rel: str,
    solc_version: str,
    evm_version: str,
    chain_name: str,
    rpc_env: str,
) -> Path:
    """Lay out a minimal forge project containing just the target PoC.

    Returns the project-relative path to the PoC (for --match-path).
    """
    (proj / "lib").mkdir(parents=True)
    (proj / "src" / "test").mkdir(parents=True)

    # forge-std — symlink to avoid copying ~20 MB of library.
    (proj / "lib" / "forge-std").symlink_to(defihacklabs_root / "lib" / "forge-std")

    for f in SHARED_SIBLINGS:
        shutil.copy2(defihacklabs_root / "src" / "test" / f, proj / "src" / "test" / f)

    poc_rel = Path(source_rel)  # e.g. src/test/2025-08/PDZ_exp.sol
    year_month = poc_rel.parent.name  # 2025-08
    src_year_month = defihacklabs_root / poc_rel.parent
    dst_year_month = proj / "src" / "test" / year_month
    dst_year_month.mkdir(parents=True, exist_ok=True)
    # Copy only the target PoC. Other files in the same year-month dir are
    # siblings from unrelated exploits and may have broken imports.
    shutil.copy2(src_year_month / poc_rel.name, dst_year_month / poc_rel.name)

    (proj / "remappings.txt").write_text("forge-std/=lib/forge-std/src/\n")

    # rpc_endpoints uses env-var expansion — forge reads ${VAR} at test time.
    foundry_toml = f"""[profile.default]
src = 'src'
out = 'out'
libs = ['lib']
solc = '{solc_version}'
evm_version = '{evm_version}'
fs_permissions = [{{ access = "read", path = "./" }}]

[rpc_endpoints]
{chain_name} = "${{{rpc_env}}}"
"""
    (proj / "foundry.toml").write_text(foundry_toml)

    return dst_year_month / poc_rel.name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variation", required=True)
    ap.add_argument("--defihacklabs-root", required=True)
    ap.add_argument(
        "--variations-root",
        default=str(Path(__file__).parent.parent / "tasks" / "forked_exploit_variations"),
    )
    ap.add_argument("--keep-tmp", action="store_true", help="Don't delete the temp project (for debugging)")
    args = ap.parse_args()

    vdir = Path(args.variations_root) / args.variation
    toml_path = vdir / "variation.toml"
    if not toml_path.exists():
        print(f"No variation.toml at {toml_path}", file=sys.stderr)
        return 2
    with toml_path.open("rb") as f:
        doc = tomllib.load(f)

    chain = doc["chain"]
    var = doc["variation"]
    compiler = doc.get("compiler", {})
    chain_name = chain["name"]
    fork_block = int(chain["fork_block"])
    rpc_env = chain["rpc_env_var"]
    source_rel = var.get("source_exploit_relpath", "")
    if not source_rel:
        print("variation has no source_exploit_relpath — nothing to warm", file=sys.stderr)
        return 2

    rpc_url = os.environ.get(rpc_env)
    if not rpc_url:
        print(f"{rpc_env} env var not set", file=sys.stderr)
        return 2

    defihacklabs_root = Path(args.defihacklabs_root)
    pod_sol = defihacklabs_root / source_rel
    if not pod_sol.exists():
        print(f"PoC not found: {pod_sol}", file=sys.stderr)
        return 2

    foundry_cache = Path.home() / ".foundry" / "cache" / "rpc" / chain_name / str(fork_block)
    if foundry_cache.exists():
        if foundry_cache.is_dir():
            shutil.rmtree(foundry_cache)
        else:
            foundry_cache.unlink()

    tmp = Path(tempfile.mkdtemp(prefix=f"warm_{args.variation}_"))
    try:
        print(f"Assembling temp project at {tmp}")
        match_path = assemble_project(
            tmp,
            defihacklabs_root,
            source_rel,
            solc_version=compiler.get("solc_version", "0.8.20"),
            evm_version=compiler.get("evm_version", "cancun"),
            chain_name=chain_name,
            rpc_env=rpc_env,
        )

        print(f"Running forge test against {pod_sol.name} @ {chain_name}:{fork_block}")
        env = os.environ.copy()
        env[rpc_env] = rpc_url

        proc = subprocess.run(
            [
                "forge", "test",
                "--match-path", str(match_path.relative_to(tmp)),
                "--fork-url", rpc_url,
                "--fork-block-number", str(fork_block),
                "-vv",
            ],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            # Original PoC may fail for reasons unrelated to caching (solc
            # mismatch, revert behaviour). We only need the cache populated,
            # so keep going if something hit chain storage.
            print("forge test returned non-zero — proceeding with partial cache:")
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:], file=sys.stderr)

        if not foundry_cache.exists():
            print(
                f"No cache produced at {foundry_cache}. Forge may not have "
                f"reached a createSelectFork or --fork-url read. Check that "
                f"the PoC actually forks {chain_name} at {fork_block}.",
                file=sys.stderr,
            )
            return 3

        # Foundry's cache layout depends on version: older releases wrote a
        # <block>/ directory (storage.json inside), newer ones (≥~1.5) write
        # a single <block> JSON file. Preserve whichever Foundry produced so
        # the anvil image, which runs its own foundryup, matches.
        target = vdir / "rpc_cache" / chain_name / str(fork_block)
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        if foundry_cache.is_dir():
            shutil.copytree(foundry_cache, target)
            sizes = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        else:
            shutil.copy2(foundry_cache, target)
            sizes = target.stat().st_size
        print(f"Wrote {target} ({sizes / 1024:.1f} KB)")
        return 0
    finally:
        if args.keep_tmp:
            print(f"Kept temp project at {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
