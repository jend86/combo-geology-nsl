from __future__ import annotations

import importlib
import os
from typing import Any

from loguru import logger


_DEFAULT_IGNORE_INDEX = -100
_LOGGED_ACTIVE = False
_ORIGINAL_COMPUTE_FUSED_CE_LOSS: Any | None = None


def _log_active_once(logit_softcapping: Any) -> None:
    global _LOGGED_ACTIVE
    if _LOGGED_ACTIVE:
        return
    logger.info(
        "DFT-fused active: chunked -p.detach()*log p, softcap={}",
        logit_softcapping,
    )
    _LOGGED_ACTIVE = True


def compute_fused_dft_loss(
    hidden_states: Any,
    lm_head_weight: Any,
    lm_head_bias: Any,
    labels: Any,
    n_items: Any | None = None,
    scaling: Any | None = None,
    shift_labels: bool = True,
    **kwargs: Any,
) -> tuple[Any, tuple[Any]]:
    """Unsloth fused-loss inner function for Dynamic Fine-Tuning.

    The outer Unsloth autograd function chunks hidden states and differentiates this
    function with torch.func.grad_and_value. Reweighting per-token CE by detached
    target probability therefore yields the DFT gradient without a custom kernel.
    """

    import torch

    ignore_index = int(kwargs.get("ignore_index", _DEFAULT_IGNORE_INDEX))
    label_smoothing = float(kwargs.get("label_smoothing", 0.0))
    device = lm_head_weight.device

    if shift_labels:
        shifted_labels = torch.empty_like(labels, device=device)
        shifted_labels[..., :-1] = labels[..., 1:]
        shifted_labels[..., -1] = ignore_index
        labels = shifted_labels

    logits = torch.nn.functional.linear(
        hidden_states.to(dtype=lm_head_weight.dtype, device=device),
        lm_head_weight,
        lm_head_bias,
    )
    vocab_size = lm_head_weight.shape[0]

    logit_scale_multiply = kwargs.get("logit_scale_multiply", None)
    logit_scale_divide = kwargs.get("logit_scale_divide", None)
    logit_softcapping = kwargs.get("logit_softcapping", None)
    if logit_scale_multiply != 0 and logit_scale_multiply is not None:
        logits = logits * logit_scale_multiply
    if logit_scale_divide != 0 and logit_scale_divide is not None:
        logits = logits / logit_scale_divide
    if logit_softcapping != 0 and logit_softcapping is not None:
        logits = logits / logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * logit_softcapping

    flat_labels = labels.view(-1).to(device).contiguous()
    ce = torch.nn.functional.cross_entropy(
        input=logits.view(-1, vocab_size).float().contiguous(),
        target=flat_labels,
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    loss = ((-ce).exp().detach() * ce).sum()

    divisor = n_items
    if divisor is None:
        divisor = (flat_labels != ignore_index).sum().clamp_min(1)
    if not torch.is_tensor(divisor):
        divisor = torch.tensor(divisor, dtype=torch.float32, device=device)
    if divisor.numel() != 1:
        divisor = divisor.ravel()[0]
    loss = loss / divisor.to(dtype=loss.dtype, device=device)

    scaled_loss = loss * scaling if scaling is not None else loss
    _log_active_once(logit_softcapping)
    return scaled_loss, (loss.detach(),)


def install_fused_dft(ignore_index: int = -100) -> None:
    """Patch Unsloth's chunked CE seam to compute fused DFT.

    This is intentionally small and version-fragile: if Unsloth moves the symbol,
    import/attribute errors should fail the run before it silently trains NLL.
    """

    global _DEFAULT_IGNORE_INDEX, _ORIGINAL_COMPUTE_FUSED_CE_LOSS
    _DEFAULT_IGNORE_INDEX = int(ignore_index)
    os.environ["UNSLOTH_ENABLE_CCE"] = "0"

    ce_mod = importlib.import_module("unsloth_zoo.fused_losses.cross_entropy_loss")
    current = getattr(ce_mod, "compute_fused_ce_loss")
    if current is compute_fused_dft_loss:
        return
    if _ORIGINAL_COMPUTE_FUSED_CE_LOSS is None:
        _ORIGINAL_COMPUTE_FUSED_CE_LOSS = current
    setattr(ce_mod, "compute_fused_ce_loss", compute_fused_dft_loss)
    logger.info(
        "Installed fused DFT loss patch into unsloth_zoo.fused_losses.cross_entropy_loss"
    )


__all__ = ["compute_fused_dft_loss", "install_fused_dft"]
