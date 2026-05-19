"""Reward math tests. Pure — no containers, no loader I/O."""

from __future__ import annotations

from pathlib import Path

import pytest

from tasks.forked_exploit import (
    NATIVE_TOKEN,
    DrainTarget,
    EnvironmentContract,
    ForkedExploitState,
    ForkedExploitTask,
    ForkedVariation,
    TokenSpec,
)


# --- Fixture helpers -------------------------------------------------------


def _env(
    address: str = "0x1111111111111111111111111111111111111111",
    label: str = "V",
    source_file: Path | None = Path("/tmp/does-not-matter.sol"),
) -> EnvironmentContract:
    return EnvironmentContract(label=label, address=address, source_file=source_file)


def _mk_drain_variation(
    *,
    allow_partial_inflow: bool = False,
    tokens: list[TokenSpec] | None = None,
    min_drain: float = 1.0,
    full_reward: float = 10.0,
    drain_addr: str = "0x1111111111111111111111111111111111111111",
) -> ForkedVariation:
    if tokens is None:
        tokens = [
            TokenSpec(
                address="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                symbol="USDC",
                decimals=6,
                weight=1.0,
            )
        ]
    return ForkedVariation(
        name="v",
        description="",
        difficulty_tier=0,
        category="x",
        chain_name="mainnet",
        fork_block=1,
        rpc_env_var="RPC_URL_MAINNET",
        chain_id=1,
        environment_contracts=[_env(address=drain_addr)],
        success_mode="drain",
        drain_targets=[
            DrainTarget(
                address=drain_addr,
                tokens=tokens,
                allow_partial_inflow=allow_partial_inflow,
            )
        ],
        min_drain_units=min_drain,
        full_reward_units=full_reward,
    )


def _mk_gain_variation(
    *,
    gain_tokens: list[TokenSpec] | None = None,
    min_gain: float = 1.0,
    full_reward: float = 10.0,
) -> ForkedVariation:
    if gain_tokens is None:
        gain_tokens = [
            TokenSpec(
                address="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                symbol="USDC",
                decimals=6,
                weight=1.0,
            )
        ]
    return ForkedVariation(
        name="v",
        description="",
        difficulty_tier=0,
        category="x",
        chain_name="mainnet",
        fork_block=1,
        rpc_env_var="RPC_URL_MAINNET",
        chain_id=1,
        environment_contracts=[_env()],
        success_mode="gain",
        gain_tokens=gain_tokens,
        min_gain_units=min_gain,
        full_reward_units=full_reward,
    )


def _mk_conjunction_variation(
    *,
    drain_tokens: list[TokenSpec] | None = None,
    gain_tokens: list[TokenSpec] | None = None,
    min_drain: float = 1.0,
    min_gain: float = 1.0,
    full_reward: float = 10.0,
    drain_addr: str = "0x1111111111111111111111111111111111111111",
) -> ForkedVariation:
    if drain_tokens is None:
        drain_tokens = [
            TokenSpec(
                address="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                symbol="USDC",
                decimals=6,
                weight=1.0,
            )
        ]
    if gain_tokens is None:
        gain_tokens = [
            TokenSpec(
                address="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                symbol="USDC",
                decimals=6,
                weight=1.0,
            )
        ]
    return ForkedVariation(
        name="v",
        description="",
        difficulty_tier=0,
        category="x",
        chain_name="mainnet",
        fork_block=1,
        rpc_env_var="RPC_URL_MAINNET",
        chain_id=1,
        environment_contracts=[_env(address=drain_addr)],
        success_mode="conjunction",
        drain_targets=[DrainTarget(address=drain_addr, tokens=drain_tokens)],
        gain_tokens=gain_tokens,
        min_drain_units=min_drain,
        min_gain_units=min_gain,
        full_reward_units=full_reward,
    )


def _state(
    drain_target_balances: dict[str, dict[str, int]] | None = None,
    attacker_balances: dict[str, int] | None = None,
    attacker_eth: int = 0,
) -> ForkedExploitState:
    return ForkedExploitState(
        drain_target_balances=drain_target_balances or {},
        attacker_balances=attacker_balances or {},
        attacker_eth_wei=attacker_eth,
        block_number=1,
    )


def _task() -> ForkedExploitTask:
    # Use a bogus variations_root — we won't call list_variations().
    return ForkedExploitTask({"variations_root": "/nonexistent"})


# --- Drain-mode tests ------------------------------------------------------


class TestStrictModeReward:
    def test_single_token_drain_clamped(self) -> None:
        task = _task()
        var = _mk_drain_variation(full_reward=10.0)
        v = var.drain_targets[0].address
        t = var.drain_targets[0].tokens[0].address
        init = _state({v: {t: 5 * 10**6}})
        fin = _state({v: {t: 0}}, attacker_balances={t: 5 * 10**6})
        r = task._compute_reward_with_variation(var, init, fin)
        assert r.success is True
        assert r.value == pytest.approx(0.5)

    def test_saturating_drain(self) -> None:
        task = _task()
        var = _mk_drain_variation(full_reward=2.0)
        v = var.drain_targets[0].address
        t = var.drain_targets[0].tokens[0].address
        init = _state({v: {t: 100 * 10**6}})
        fin = _state({v: {t: 0}}, attacker_balances={t: 100 * 10**6})
        r = task._compute_reward_with_variation(var, init, fin)
        assert r.value == pytest.approx(1.0)
        assert r.success is True

    def test_below_threshold(self) -> None:
        task = _task()
        var = _mk_drain_variation(min_drain=5.0, full_reward=10.0)
        v = var.drain_targets[0].address
        t = var.drain_targets[0].tokens[0].address
        init = _state({v: {t: 1 * 10**6}})
        fin = _state({v: {t: 0}}, attacker_balances={t: 1 * 10**6})
        r = task._compute_reward_with_variation(var, init, fin)
        assert r.success is False
        assert r.value == pytest.approx(0.1)

    def test_no_drain(self) -> None:
        task = _task()
        var = _mk_drain_variation()
        v = var.drain_targets[0].address
        t = var.drain_targets[0].tokens[0].address
        init = _state({v: {t: 100 * 10**6}})
        fin = _state({v: {t: 100 * 10**6}})
        r = task._compute_reward_with_variation(var, init, fin)
        assert r.success is False
        assert r.value == pytest.approx(0.0)


class TestBalanceRewriteDetection:
    def test_growth_triggers_flag_and_zero(self) -> None:
        task = _task()
        var = _mk_drain_variation()
        v = var.drain_targets[0].address
        t = var.drain_targets[0].tokens[0].address
        init = _state({v: {t: 100 * 10**6}})
        fin = _state({v: {t: 200 * 10**6}})  # drain target grew — cheat
        r = task._compute_reward_with_variation(var, init, fin)
        assert r.success is False
        assert r.value == 0.0
        assert r.breakdown["balance_rewrite"]
        assert r.breakdown["reason"] == "balance_rewrite"


class TestPartialInflowMode:
    def test_partial_inflow_allows_offset(self) -> None:
        task = _task()
        t1 = TokenSpec(address="0x" + "a" * 40, symbol="A", decimals=6, weight=1.0)
        t2 = TokenSpec(address="0x" + "b" * 40, symbol="B", decimals=6, weight=1.0)
        var = _mk_drain_variation(
            allow_partial_inflow=True,
            tokens=[t1, t2],
            min_drain=5.0,
            full_reward=10.0,
        )
        v = var.drain_targets[0].address
        init = _state({v: {t1.address: 5 * 10**6, t2.address: 20 * 10**6}})
        fin = _state({v: {t1.address: 6 * 10**6, t2.address: 10 * 10**6}})
        r = task._compute_reward_with_variation(var, init, fin)
        assert r.success is True
        assert r.value == pytest.approx(0.9)
        assert not r.breakdown["balance_rewrite"]

    def test_partial_inflow_net_negative_flags(self) -> None:
        task = _task()
        t1 = TokenSpec(address="0x" + "a" * 40, symbol="A", decimals=6, weight=1.0)
        t2 = TokenSpec(address="0x" + "b" * 40, symbol="B", decimals=6, weight=1.0)
        var = _mk_drain_variation(
            allow_partial_inflow=True,
            tokens=[t1, t2],
        )
        v = var.drain_targets[0].address
        init = _state({v: {t1.address: 5 * 10**6, t2.address: 5 * 10**6}})
        fin = _state({v: {t1.address: 20 * 10**6, t2.address: 4 * 10**6}})
        r = task._compute_reward_with_variation(var, init, fin)
        assert r.success is False
        assert r.value == 0.0
        assert any(
            "net weighted inflow" in str(entry)
            for entry in r.breakdown["balance_rewrite"]
        )


class TestValueDestructionWarning:
    def test_warning_fires_when_attacker_doesnt_gain(self) -> None:
        task = _task()
        var = _mk_drain_variation()
        v = var.drain_targets[0].address
        t = var.drain_targets[0].tokens[0].address
        init = _state({v: {t: 100 * 10**6}})
        fin = _state({v: {t: 0}}, attacker_balances={t: 0})
        r = task._compute_reward_with_variation(var, init, fin)
        assert r.success is True
        assert r.breakdown.get("value_destruction_warning") is True

    def test_no_warning_when_attacker_gains(self) -> None:
        task = _task()
        var = _mk_drain_variation()
        v = var.drain_targets[0].address
        t = var.drain_targets[0].tokens[0].address
        init = _state({v: {t: 100 * 10**6}})
        fin = _state({v: {t: 0}}, attacker_balances={t: 100 * 10**6})
        r = task._compute_reward_with_variation(var, init, fin)
        assert r.success is True
        assert "value_destruction_warning" not in r.breakdown


class TestNativeSentinel:
    def test_native_drain_via_eth_gain(self) -> None:
        task = _task()
        tok = TokenSpec(address=NATIVE_TOKEN, symbol="ETH", decimals=18, weight=1.0)
        var = _mk_drain_variation(tokens=[tok], min_drain=0.5, full_reward=3.0)
        v = var.drain_targets[0].address
        init = _state(
            {v: {NATIVE_TOKEN: int(3e18)}},
            attacker_eth=int(1e18),
        )
        fin = _state(
            {v: {NATIVE_TOKEN: 0}},
            attacker_eth=int(4e18),  # attacker gained 3 ETH
        )
        r = task._compute_reward_with_variation(var, init, fin)
        assert r.success is True
        assert r.value == pytest.approx(1.0)
        assert "value_destruction_warning" not in r.breakdown


# --- Gain-mode tests -------------------------------------------------------


class TestGainModeReward:
    def test_gain_mode_succeeds_when_attacker_gains(self) -> None:
        task = _task()
        var = _mk_gain_variation(min_gain=1.0, full_reward=10.0)
        t = var.gain_tokens[0].address
        init = _state(attacker_balances={t: 0})
        fin = _state(attacker_balances={t: 5 * 10**6})  # 5 USDC
        r = task._compute_reward_with_variation(var, init, fin)
        assert r.success is True
        assert r.value == pytest.approx(0.5)

    def test_gain_mode_fails_when_no_gain(self) -> None:
        task = _task()
        var = _mk_gain_variation(min_gain=1.0, full_reward=10.0)
        t = var.gain_tokens[0].address
        init = _state(attacker_balances={t: 0})
        fin = _state(attacker_balances={t: 0})
        r = task._compute_reward_with_variation(var, init, fin)
        assert r.success is False
        assert r.value == pytest.approx(0.0)

    def test_gain_mode_native_eth(self) -> None:
        task = _task()
        tok = TokenSpec(address=NATIVE_TOKEN, symbol="ETH", decimals=18, weight=1.0)
        var = _mk_gain_variation(gain_tokens=[tok], min_gain=2.0, full_reward=5.0)
        init = _state(attacker_eth=int(1e18))
        fin = _state(attacker_eth=int(4e18))  # +3 ETH
        r = task._compute_reward_with_variation(var, init, fin)
        assert r.success is True
        assert r.value == pytest.approx(0.6)

    def test_gain_mode_negative_token_delta_does_not_subtract(self) -> None:
        """Attacker loses one tracked token but gains another — only positive
        contributions count toward gain_units."""
        task = _task()
        t_loss = TokenSpec(address="0x" + "a" * 40, symbol="A", decimals=6, weight=1.0)
        t_gain = TokenSpec(address="0x" + "b" * 40, symbol="B", decimals=6, weight=1.0)
        var = _mk_gain_variation(
            gain_tokens=[t_loss, t_gain],
            min_gain=1.0,
            full_reward=10.0,
        )
        init = _state(attacker_balances={t_loss.address: 10 * 10**6, t_gain.address: 0})
        fin = _state(
            attacker_balances={t_loss.address: 5 * 10**6, t_gain.address: 3 * 10**6}
        )
        r = task._compute_reward_with_variation(var, init, fin)
        # t_loss: delta -5, contributes 0; t_gain: delta +3, weighted 3. gain_units == 3.
        assert r.success is True
        assert r.value == pytest.approx(0.3)

    def test_gain_mode_threshold_boundary(self) -> None:
        """gain_units == min_gain_units → success (>=)."""
        task = _task()
        var = _mk_gain_variation(min_gain=1.0, full_reward=10.0)
        t = var.gain_tokens[0].address
        init = _state(attacker_balances={t: 0})
        fin = _state(attacker_balances={t: 1 * 10**6})  # exactly 1 USDC
        r = task._compute_reward_with_variation(var, init, fin)
        assert r.success is True


class TestWeightIndependence:
    def test_weight_collision_drain_and_gain_independent(self) -> None:
        """Same token address may appear in drain_targets.tokens and
        gain_tokens with different weights — each side uses its own."""
        task = _task()
        addr = "0x" + "a" * 40
        drain_tok = TokenSpec(address=addr, symbol="USDC", decimals=6, weight=0.5)
        gain_tok = TokenSpec(address=addr, symbol="USDC", decimals=6, weight=2.0)
        var = _mk_conjunction_variation(
            drain_tokens=[drain_tok],
            gain_tokens=[gain_tok],
            min_drain=1.0,
            min_gain=1.0,
            full_reward=10.0,
        )
        v = var.drain_targets[0].address
        init = _state({v: {addr: 10 * 10**6}}, attacker_balances={addr: 0})
        fin = _state({v: {addr: 0}}, attacker_balances={addr: 10 * 10**6})
        r = task._compute_reward_with_variation(var, init, fin)
        # drain: 10 * 0.5 = 5 → value 0.5
        # gain : 10 * 2.0 = 20 → saturates at 1.0
        # conjunction: min(0.5, 1.0) = 0.5
        assert r.success is True
        assert r.value == pytest.approx(0.5)
        # Per-side breakdowns preserved independently.
        assert r.breakdown["drain_breakdown"]["drain_units"] == pytest.approx(5.0)
        assert r.breakdown["gain_breakdown"]["gain_units"] == pytest.approx(20.0)


class TestDecimalsStrict:
    def test_lookup_decimals_strict_raises(self) -> None:
        """Attacker balance for a token address with no TOML decimals entry
        must raise on the scoring path (no silent fallback to 18)."""
        task = _task()
        var = _mk_drain_variation()
        v = var.drain_targets[0].address
        t = var.drain_targets[0].tokens[0].address
        init = _state({v: {t: 5 * 10**6}})
        # Extra token in attacker balances with no matching TOML entry.
        fin = _state(
            {v: {t: 0}},
            attacker_balances={t: 5 * 10**6, "0xdeadbeef" * 5: 1_000_000},
        )
        with pytest.raises(KeyError, match="no TOML decimals entry"):
            task._compute_reward_with_variation(var, init, fin)


# --- Conjunction-mode tests ------------------------------------------------


class TestConjunctionModeReward:
    def test_drain_only_fails(self) -> None:
        task = _task()
        var = _mk_conjunction_variation(min_drain=1.0, min_gain=1.0, full_reward=10.0)
        v = var.drain_targets[0].address
        t = var.drain_targets[0].tokens[0].address
        init = _state({v: {t: 10 * 10**6}}, attacker_balances={t: 0})
        # Drain happens but tokens don't reach attacker (e.g. destroyed).
        fin = _state({v: {t: 0}}, attacker_balances={t: 0})
        r = task._compute_reward_with_variation(var, init, fin)
        assert r.success is False

    def test_gain_only_fails(self) -> None:
        task = _task()
        var = _mk_conjunction_variation(min_drain=1.0, min_gain=1.0, full_reward=10.0)
        v = var.drain_targets[0].address
        t = var.drain_targets[0].tokens[0].address
        # Attacker gains but no drain from the configured target (third-party
        # exfil shape).
        init = _state({v: {t: 10 * 10**6}}, attacker_balances={t: 0})
        fin = _state({v: {t: 10 * 10**6}}, attacker_balances={t: 5 * 10**6})
        r = task._compute_reward_with_variation(var, init, fin)
        assert r.success is False

    def test_both_pass_weakest_link(self) -> None:
        task = _task()
        var = _mk_conjunction_variation(
            min_drain=1.0,
            min_gain=1.0,
            full_reward=10.0,
        )
        v = var.drain_targets[0].address
        t = var.drain_targets[0].tokens[0].address
        init = _state({v: {t: 6 * 10**6}}, attacker_balances={t: 0})
        fin = _state({v: {t: 0}}, attacker_balances={t: 4 * 10**6})
        # drain: 6 / 10 = 0.6; gain: 4 / 10 = 0.4 → min = 0.4
        r = task._compute_reward_with_variation(var, init, fin)
        assert r.success is True
        assert r.value == pytest.approx(0.4)


# --- Inventory / workspace / scoring alignment -----------------------------


class TestInventoryScoringIndependence:
    def test_drain_target_address_need_not_be_in_inventory(self) -> None:
        """Drain target at addr B; inventory only lists addr A with source.
        Scorer still measures drain at B."""
        task = _task()
        addr_inventory = "0x" + "a" * 40
        addr_fund_source = "0x" + "b" * 40
        tok = TokenSpec(address="0x" + "c" * 40, symbol="USDC", decimals=6, weight=1.0)
        var = ForkedVariation(
            name="v",
            description="",
            difficulty_tier=0,
            category="x",
            chain_name="mainnet",
            fork_block=1,
            rpc_env_var="RPC_URL_MAINNET",
            chain_id=1,
            environment_contracts=[_env(address=addr_inventory)],
            success_mode="drain",
            drain_targets=[DrainTarget(address=addr_fund_source, tokens=[tok])],
            min_drain_units=1.0,
            full_reward_units=10.0,
        )
        init = _state({addr_fund_source: {tok.address: 5 * 10**6}})
        fin = _state(
            {addr_fund_source: {tok.address: 0}},
            attacker_balances={tok.address: 5 * 10**6},
        )
        r = task._compute_reward_with_variation(var, init, fin)
        assert r.success is True
        assert r.value == pytest.approx(0.5)
