"""Loader + schema validation for ForkedExploit variations. No containers needed.

Covers the variation schema:
- [[environment.contracts]] inventory + source delivery
- [success].mode dispatch (drain | gain | conjunction)
- [[success.drain_targets]] / [[success.gain_tokens]] mode-gating
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tasks.forked_exploit import (
    NATIVE_TOKEN,
    ForkedVariation,
    _discover_variations,
    _load_variation_toml,
)


# ---------------------------------------------------------------------------
# TOML builders. Small, composable helpers so each test perturbs exactly one
# concern.
# ---------------------------------------------------------------------------


_VALID_TOML_DEFAULTS = {
    "name": "happy_variation",
    "blocked": False,
    "blocked_reason": "",
    "mode": "drain",
    "drain_targets": (
        "[[success.drain_targets]]\n"
        'address = "0x0000000000000000000000000000000000000001"\n'
        "tokens = [\n"
        '  { address = "0x0000000000000000000000000000000000000000", symbol = "ETH", decimals = 18, weight = 1.0 },\n'
        "]\n"
    ),
    "gain_tokens": "",
    "min_drain_units": "min_drain_units = 0.1",
    "min_gain_units": "",
    "full_reward_units": 1.0,
    "environment_contracts": (
        "[[environment.contracts]]\n"
        'address = "0x0000000000000000000000000000000000000001"\n'
        'label = "Foo"\n'
        'source_file = "sources/Foo.sol"\n'
    ),
}


def _write_variation_toml(
    dir_: Path,
    *,
    create_source: bool = True,
    **overrides,
) -> None:
    """Write a minimal valid (by default) variation.toml under dir_.

    Any value in _VALID_TOML_DEFAULTS may be overridden by keyword. For
    larger sections (environment_contracts, drain_targets, gain_tokens,
    etc.) pass a complete TOML fragment string.
    """
    cfg = {**_VALID_TOML_DEFAULTS, **overrides}
    dir_.mkdir(parents=True, exist_ok=True)
    if create_source:
        (dir_ / "sources").mkdir(parents=True, exist_ok=True)
        (dir_ / "sources" / "Foo.sol").write_text("// stub\ncontract Foo {}\n")

    blocked_reason_line = (
        f'blocked_reason = "{cfg["blocked_reason"]}"' if cfg["blocked"] else ""
    )
    dir_.joinpath("variation.toml").write_text(f"""\
schema_version = 1

[variation]
name = "{cfg["name"]}"
description = "test"
difficulty_tier = 1
category = "access_control"
source_exploit_relpath = "src/test/2024-01/Foo.sol"
blocked = {"true" if cfg["blocked"] else "false"}
{blocked_reason_line}

[chain]
name = "mainnet"
fork_block = 19000000
rpc_env_var = "RPC_URL_MAINNET"
chain_id = 1
requires_archive = false

[compiler]
solc_version = "0.8.20"
evm_version = "cancun"

{cfg["environment_contracts"]}

[success]
mode = "{cfg["mode"]}"
{cfg["min_drain_units"]}
{cfg["min_gain_units"]}
full_reward_units = {cfg["full_reward_units"]}

{cfg["drain_targets"]}

{cfg["gain_tokens"]}

[attacker]
funding_strategy = "flashloan_expected"
initial_eth = 1.0
""")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestLoaderHappyPath:
    def test_valid_drain_variation_round_trips(self, tmp_path: Path) -> None:
        vdir = tmp_path / "happy_variation"
        _write_variation_toml(vdir, name="happy_variation")
        v = _load_variation_toml(vdir)
        assert isinstance(v, ForkedVariation)
        assert v.name == "happy_variation"
        assert v.success_mode == "drain"
        assert len(v.environment_contracts) == 1
        assert v.environment_contracts[0].source_file is not None
        assert len(v.drain_targets) == 1
        assert v.drain_targets[0].tokens[0].address == NATIVE_TOKEN
        assert v.drain_targets[0].tokens[0].is_native
        assert v.gain_tokens == []

    def test_valid_gain_variation(self, tmp_path: Path) -> None:
        vdir = tmp_path / "gain_variation"
        _write_variation_toml(
            vdir,
            name="gain_variation",
            mode="gain",
            drain_targets="",
            min_drain_units="",
            min_gain_units="min_gain_units = 0.1",
            gain_tokens=(
                "[[success.gain_tokens]]\n"
                'address = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"\n'
                'symbol = "USDC"\n'
                "decimals = 6\n"
                "weight = 1.0\n"
            ),
        )
        v = _load_variation_toml(vdir)
        assert v.success_mode == "gain"
        assert v.drain_targets == []
        assert len(v.gain_tokens) == 1
        assert v.gain_tokens[0].symbol == "USDC"

    def test_valid_conjunction_variation(self, tmp_path: Path) -> None:
        vdir = tmp_path / "conj_variation"
        _write_variation_toml(
            vdir,
            name="conj_variation",
            mode="conjunction",
            min_drain_units="min_drain_units = 0.1",
            min_gain_units="min_gain_units = 0.1",
            gain_tokens=(
                "[[success.gain_tokens]]\n"
                'address = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"\n'
                'symbol = "USDC"\n'
                "decimals = 6\n"
                "weight = 1.0\n"
            ),
        )
        v = _load_variation_toml(vdir)
        assert v.success_mode == "conjunction"
        assert len(v.drain_targets) == 1
        assert len(v.gain_tokens) == 1

    def test_inventory_entry_without_source_file_accepted(
        self,
        tmp_path: Path,
    ) -> None:
        """Router-style entry: no source_file, agent is only told it exists."""
        vdir = tmp_path / "mixed_inventory"
        _write_variation_toml(
            vdir,
            name="mixed_inventory",
            environment_contracts=(
                "[[environment.contracts]]\n"
                'address = "0x0000000000000000000000000000000000000001"\n'
                'label = "Foo"\n'
                'source_file = "sources/Foo.sol"\n'
                "\n"
                "[[environment.contracts]]\n"
                'address = "0x0000000000000000000000000000000000000002"\n'
                'label = "Router"\n'
            ),
        )
        v = _load_variation_toml(vdir)
        assert len(v.environment_contracts) == 2
        assert v.environment_contracts[0].source_file is not None
        assert v.environment_contracts[1].source_file is None

    def test_discover_skips_blocked_by_default(self, tmp_path: Path) -> None:
        _write_variation_toml(tmp_path / "live_one", name="live_one")
        blocked_dir = tmp_path / "_blocked" / "dead_one"
        _write_variation_toml(
            blocked_dir,
            name="dead_one",
            blocked=True,
            blocked_reason="reason",
        )
        live = _discover_variations(tmp_path, include_blocked=False)
        assert [v.name for v in live] == ["live_one"]
        all_ = _discover_variations(tmp_path, include_blocked=True)
        assert sorted(v.name for v in all_) == ["dead_one", "live_one"]


# ---------------------------------------------------------------------------
# Schema / validator — mode gating
# ---------------------------------------------------------------------------


class TestScoringValidator:
    def test_loader_rejects_unknown_mode(self, tmp_path: Path) -> None:
        vdir = tmp_path / "v1"
        _write_variation_toml(vdir, name="v1", mode="max")
        with pytest.raises(ValueError, match="mode"):
            _load_variation_toml(vdir)

    def test_loader_rejects_drain_mode_with_gain_tokens(
        self,
        tmp_path: Path,
    ) -> None:
        vdir = tmp_path / "v1"
        _write_variation_toml(
            vdir,
            name="v1",
            gain_tokens=(
                "[[success.gain_tokens]]\n"
                'address = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"\n'
                'symbol = "USDC"\n'
                "decimals = 6\n"
                "weight = 1.0\n"
            ),
        )
        with pytest.raises(ValueError, match="drain mode forbids"):
            _load_variation_toml(vdir)

    def test_loader_rejects_gain_mode_with_drain_targets(
        self,
        tmp_path: Path,
    ) -> None:
        vdir = tmp_path / "v1"
        _write_variation_toml(
            vdir,
            name="v1",
            mode="gain",
            min_drain_units="",
            min_gain_units="min_gain_units = 0.1",
            gain_tokens=(
                "[[success.gain_tokens]]\n"
                'address = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"\n'
                'symbol = "USDC"\n'
                "decimals = 6\n"
                "weight = 1.0\n"
            ),
        )
        # drain_targets still present from defaults → rejection expected
        with pytest.raises(ValueError, match="gain mode forbids"):
            _load_variation_toml(vdir)

    def test_loader_rejects_gain_mode_without_gain_tokens(
        self,
        tmp_path: Path,
    ) -> None:
        vdir = tmp_path / "v1"
        _write_variation_toml(
            vdir,
            name="v1",
            mode="gain",
            drain_targets="",
            min_drain_units="",
            min_gain_units="min_gain_units = 0.1",
        )
        with pytest.raises(ValueError, match="gain mode requires.*gain_tokens"):
            _load_variation_toml(vdir)

    def test_loader_rejects_drain_mode_without_drain_targets(
        self,
        tmp_path: Path,
    ) -> None:
        vdir = tmp_path / "v1"
        _write_variation_toml(vdir, name="v1", drain_targets="")
        with pytest.raises(ValueError, match="drain mode requires.*drain_targets"):
            _load_variation_toml(vdir)

    def test_loader_rejects_conjunction_missing_drain_section(
        self,
        tmp_path: Path,
    ) -> None:
        vdir = tmp_path / "v1"
        _write_variation_toml(
            vdir,
            name="v1",
            mode="conjunction",
            drain_targets="",
            min_drain_units="min_drain_units = 0.1",
            min_gain_units="min_gain_units = 0.1",
            gain_tokens=(
                "[[success.gain_tokens]]\n"
                'address = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"\n'
                'symbol = "USDC"\n'
                "decimals = 6\n"
                "weight = 1.0\n"
            ),
        )
        with pytest.raises(ValueError, match="conjunction mode requires both"):
            _load_variation_toml(vdir)

    def test_loader_rejects_conjunction_missing_gain_section(
        self,
        tmp_path: Path,
    ) -> None:
        vdir = tmp_path / "v1"
        _write_variation_toml(
            vdir,
            name="v1",
            mode="conjunction",
            min_drain_units="min_drain_units = 0.1",
            min_gain_units="min_gain_units = 0.1",
        )
        with pytest.raises(ValueError, match="conjunction mode requires both"):
            _load_variation_toml(vdir)


# ---------------------------------------------------------------------------
# Schema / validator — inventory
# ---------------------------------------------------------------------------


class TestInventoryValidator:
    def test_loader_rejects_empty_environment_contracts(
        self,
        tmp_path: Path,
    ) -> None:
        vdir = tmp_path / "v1"
        _write_variation_toml(vdir, name="v1", environment_contracts="")
        with pytest.raises(ValueError, match="environment.contracts"):
            _load_variation_toml(vdir)

    def test_loader_rejects_environment_with_no_source_file(
        self,
        tmp_path: Path,
    ) -> None:
        vdir = tmp_path / "v1"
        _write_variation_toml(
            vdir,
            name="v1",
            create_source=False,
            environment_contracts=(
                "[[environment.contracts]]\n"
                'address = "0x0000000000000000000000000000000000000001"\n'
                'label = "Foo"\n'
            ),
        )
        with pytest.raises(ValueError, match="source_file"):
            _load_variation_toml(vdir)

    def test_loader_rejects_duplicate_environment_addresses(
        self,
        tmp_path: Path,
    ) -> None:
        vdir = tmp_path / "v1"
        _write_variation_toml(
            vdir,
            name="v1",
            environment_contracts=(
                "[[environment.contracts]]\n"
                'address = "0x0000000000000000000000000000000000000001"\n'
                'label = "Foo"\n'
                'source_file = "sources/Foo.sol"\n'
                "\n"
                "[[environment.contracts]]\n"
                'address = "0x0000000000000000000000000000000000000001"\n'
                'label = "FooDup"\n'
            ),
        )
        with pytest.raises(ValueError, match="duplicate environment"):
            _load_variation_toml(vdir)


# ---------------------------------------------------------------------------
# Existing schema checks preserved from v1 loader
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_directory_name_mismatch(self, tmp_path: Path) -> None:
        vdir = tmp_path / "foo"
        _write_variation_toml(vdir, name="bar")
        with pytest.raises(ValueError, match="must equal directory name"):
            _load_variation_toml(vdir)

    @pytest.mark.parametrize(
        ("original", "replacement", "match"),
        [
            pytest.param(
                'name = "mainnet"',
                'name = "solana"',
                "Chain 'solana' not in SUPPORTED_CHAINS",
                id="unsupported-chain",
            ),
            pytest.param("chain_id = 1", "chain_id = 42", "chain_id", id="chain-id"),
            pytest.param(
                'rpc_env_var = "RPC_URL_MAINNET"',
                'rpc_env_var = "RPC_URL_WRONG"',
                "expects env",
                id="rpc-env-var",
            ),
        ],
    )
    def test_rejects_invalid_chain_metadata(
        self,
        tmp_path: Path,
        original: str,
        replacement: str,
        match: str,
    ) -> None:
        vdir = tmp_path / "v1"
        _write_variation_toml(vdir, name="v1")
        content = vdir.joinpath("variation.toml").read_text().replace(
            original,
            replacement,
        )
        vdir.joinpath("variation.toml").write_text(content)
        with pytest.raises(Exception, match=match):
            _load_variation_toml(vdir)

    def test_duplicate_drain_target_token_pair(self, tmp_path: Path) -> None:
        vdir = tmp_path / "v1"
        _write_variation_toml(
            vdir,
            name="v1",
            drain_targets=(
                "[[success.drain_targets]]\n"
                'address = "0x0000000000000000000000000000000000000001"\n'
                "tokens = [\n"
                '  { address = "0x0000000000000000000000000000000000000000", symbol = "ETH", decimals = 18, weight = 1.0 },\n'
                '  { address = "0x0000000000000000000000000000000000000000", symbol = "ETH2", decimals = 18, weight = 0.5 },\n'
                "]\n"
            ),
        )
        with pytest.raises(ValueError, match="duplicate"):
            _load_variation_toml(vdir)

    @pytest.mark.parametrize(
        "weight",
        [
            pytest.param("11.0", id="above-range"),
            pytest.param("0", id="zero"),
        ],
    )
    def test_rejects_invalid_weight(self, tmp_path: Path, weight: str) -> None:
        vdir = tmp_path / "v1"
        _write_variation_toml(vdir, name="v1")
        content = vdir.joinpath("variation.toml").read_text().replace(
            "weight = 1.0",
            f"weight = {weight}",
        )
        vdir.joinpath("variation.toml").write_text(content)
        with pytest.raises(ValueError, match="weight"):
            _load_variation_toml(vdir)

    def test_missing_source_file(self, tmp_path: Path) -> None:
        vdir = tmp_path / "v1"
        _write_variation_toml(vdir, name="v1")
        (vdir / "sources" / "Foo.sol").unlink()
        with pytest.raises(ValueError, match="source_file"):
            _load_variation_toml(vdir)

    def test_solc_version_not_installed(self, tmp_path: Path) -> None:
        vdir = tmp_path / "v1"
        _write_variation_toml(vdir, name="v1")
        content = (
            vdir.joinpath("variation.toml")
            .read_text()
            .replace('solc_version = "0.8.20"', 'solc_version = "0.4.17"')
        )
        vdir.joinpath("variation.toml").write_text(content)
        with pytest.raises(ValueError, match="solc_version"):
            _load_variation_toml(vdir)

    def test_thresholds_invalid(self, tmp_path: Path) -> None:
        vdir = tmp_path / "v1"
        _write_variation_toml(vdir, name="v1", full_reward_units=0.01)
        with pytest.raises(ValueError, match="thresholds"):
            _load_variation_toml(vdir)

    def test_blocked_without_reason(self, tmp_path: Path) -> None:
        vdir = tmp_path / "v1"
        _write_variation_toml(vdir, name="v1", blocked=True, blocked_reason="")
        with pytest.raises(ValueError, match="blocked_reason"):
            _load_variation_toml(vdir)


class TestRpcEnvLazyResolution:
    def test_load_does_not_resolve_env(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("RPC_URL_MAINNET", raising=False)
        vdir = tmp_path / "v1"
        _write_variation_toml(vdir, name="v1")
        v = _load_variation_toml(vdir)
        assert v.rpc_env_var == "RPC_URL_MAINNET"
