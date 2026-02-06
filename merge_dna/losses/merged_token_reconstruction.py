import torch
import torch.nn as nn
from typing import Optional

_ce = nn.CrossEntropyLoss(reduction="sum")

def merged_token_reconstruction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
):
    """
    Compute cross-entropy reconstruction loss at base resolution.

    Args:
        logits: (B, L_trunc, V) float tensor (pre-softmax)
        targets: (B, L_trunc) long tensor with token ids
        mask: Optional (B, L_trunc) bool/byte tensor indicating which positions to include
        reduction: "mean" or "sum"
a
    Returns:
      scalar loss (torch.Tensor)
    """
    V = logits.shape[2]
    logits_flat = logits.view(-1, V)              # (B*L, V)
    targets_flat = targets.view(-1)               # (B*L,)

    if mask is not None:
        mask_flat = mask.view(-1).to(dtype=torch.bool)
        if mask_flat.numel() != targets_flat.numel():
            raise ValueError("mask must be same #elements as targets")
        if mask_flat.sum().item() == 0:
            # nothing to predict
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        logits_flat = logits_flat[mask_flat]
        targets_flat = targets_flat[mask_flat]

    loss_sum = _ce(logits_flat, targets_flat)  # sum of CE over selected positions

    if reduction == "sum":
        return loss_sum
    elif reduction == "mean":
        return loss_sum / targets_flat.numel()
    else:
        raise ValueError("reduction must be 'sum' or 'mean'")