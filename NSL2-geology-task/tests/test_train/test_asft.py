"""CPU unit tests for the vendored ASFT loss helpers (``src/train/asft.py``).

These cover the parts that do not require the Triton CE kernel (no GPU): the
KL-gating resolution (local review fix #1), label shifting, logit transforms,
DFT weighting via the CE-losses path, and KL divergence. End-to-end loss and
trainer behaviour require a GPU smoke test (design doc step 7).
"""

import unittest

import torch

from src.train.asft import (
    _compute_dft_weights,
    _compute_kl_divergence,
    build_shift_labels,
    effective_logits,
    resolve_effective_mode,
)


class TestResolveEffectiveMode(unittest.TestCase):
    """Review fix #1: kl_weight == 0.0 must skip the KL/reference path."""

    def test_asft_with_zero_kl_collapses_to_dft(self):
        self.assertEqual(resolve_effective_mode("asft", 0.0), ("dft", False))

    def test_sft_plus_kl_with_zero_kl_collapses_to_sft(self):
        self.assertEqual(resolve_effective_mode("sft+kl", 0.0), ("sft", False))

    def test_asft_with_nonzero_kl_stays_active(self):
        self.assertEqual(resolve_effective_mode("asft", 0.5), ("asft", True))

    def test_sft_plus_kl_with_nonzero_kl_stays_active(self):
        self.assertEqual(resolve_effective_mode("sft+kl", 0.25), ("sft+kl", True))

    def test_dft_never_activates_kl(self):
        self.assertEqual(resolve_effective_mode("dft", 0.9), ("dft", False))

    def test_sft_never_activates_kl(self):
        self.assertEqual(resolve_effective_mode("sft", 0.9), ("sft", False))


class TestAsftLossHelpers(unittest.TestCase):
    def test_build_shift_labels_shifts_and_caps_last(self):
        labels = torch.tensor([[10, 11, 12, 13]])
        shifted = build_shift_labels(labels)
        self.assertTrue(torch.equal(shifted, torch.tensor([[11, 12, 13, -100]])))

    def test_effective_logits_identity_without_scale_or_cap(self):
        x = torch.tensor([[1.0, -2.0, 3.0]])
        self.assertTrue(torch.allclose(effective_logits(x), x))

    def test_effective_logits_applies_scaling(self):
        x = torch.tensor([[1.0, -2.0, 3.0]])
        self.assertTrue(torch.allclose(effective_logits(x, logit_scaling=2.0), x * 2.0))

    def test_effective_logits_applies_softcapping(self):
        x = torch.tensor([[1.0, -2.0, 3.0]])
        t = 5.0
        self.assertTrue(
            torch.allclose(
                effective_logits(x, logit_softcapping=t), t * torch.tanh(x / t)
            )
        )

    def test_dft_weights_from_ce_losses_are_exp_neg_ce_masked(self):
        # DFT weight w_t = p_theta(y_t) = exp(-ce_t), zeroed on ignored tokens.
        ce = torch.tensor([[0.0, 1.0, 2.0]])
        valid = torch.tensor([[True, True, False]])
        weights = _compute_dft_weights(None, None, ce_losses=ce, valid_mask=valid)
        expected = torch.exp(-ce) * valid
        self.assertTrue(torch.allclose(weights, expected))

    def test_kl_is_zero_for_identical_distributions(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 5, 8)
        kl = _compute_kl_divergence(logits, logits.clone())
        self.assertTrue(torch.allclose(kl, torch.zeros_like(kl), atol=1e-5))

    def test_kl_is_nonnegative_and_direction_asymmetric(self):
        torch.manual_seed(1)
        cur = torch.randn(1, 3, 6)
        ref = torch.randn(1, 3, 6)
        kl_fwd = _compute_kl_divergence(cur, ref, kl_direction="forward")
        kl_rev = _compute_kl_divergence(cur, ref, kl_direction="reverse")
        self.assertTrue(bool((kl_fwd >= -1e-6).all()))
        self.assertTrue(bool((kl_rev >= -1e-6).all()))
        self.assertFalse(torch.allclose(kl_fwd, kl_rev))


class TestAsftTrainerWiring(unittest.TestCase):
    """Prove the local trainer/config are real (not just mocked in qlora tests)."""

    def test_asft_trainer_subclasses_unsloth_trainer(self):
        from unsloth.trainer import UnslothTrainer

        from src.train.asft import ASFTStreamingConfig as LossStreamingConfig
        from src.train.asft_trainer import ASFTStreamingConfig, ASFTTrainer

        self.assertTrue(issubclass(ASFTTrainer, UnslothTrainer))
        # ASFTStreamingConfig is re-exported from the vendored loss module.
        self.assertIs(ASFTStreamingConfig, LossStreamingConfig)


if __name__ == "__main__":
    unittest.main()
