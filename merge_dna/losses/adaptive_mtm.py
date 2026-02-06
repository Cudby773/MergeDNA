import torch

def select_merged_tokens_by_importance(
    merged_feats: torch.Tensor,
    mask_ratio: float = 0.15,
    min_mask: int = 1,
) -> torch.BoolTensor:
    """
    Choose top-k most 'important' merged tokens per sample to be masked,
    where importance = ||feat||_2 (or any other heuristic).
    Args:
      merged_feats: (B, M, D)
      mask_ratio: fraction of merged tokens to mask (0..1)
      min_mask: minimum masked tokens per sample

    Returns:
      mask_merged: (B, M) bool tensor (True => selected/masked)
    """
    B, M, D = merged_feats.shape
    importance = merged_feats.norm(dim=-1)  # (B, M)
    k = max(min_mask, int(round(M * mask_ratio)))
    if k >= M:
        # mask all except at least 1 unmasked? We'll just mask up to M-1
        k = max(1, M - 1)
    # topk returns (values, indices)
    _, topk_idx = torch.topk(importance, k=k, dim=-1, largest=True, sorted=False)  # (B, k)
    mask = torch.zeros((B, M), dtype=torch.bool, device=merged_feats.device)
    batch_idx = torch.arange(B, device=merged_feats.device).unsqueeze(1).expand(-1, topk_idx.shape[1])
    mask[batch_idx, topk_idx] = True
    return mask


def merged_mask_to_base_mask(owner_idx: torch.LongTensor, merged_mask: torch.BoolTensor) -> torch.BoolTensor:
    """
    Convert a mask over merged tokens into a mask over original base positions.

    Args:
      owner_idx: (B, L_orig) long mapping each base pos j -> merged index in [0..M-1]
      merged_mask: (B, M) bool indicating which merged tokens are masked

    Returns:
      base_mask: (B, L_orig) bool where base_mask[b,j] = merged_mask[b, owner_idx[b,j]]
    """
    B, L = owner_idx.shape
    # owner_idx is long in [0..M-1]
    # gather merged_mask for each base position
    # expand owner_idx to (B, L, 1) to gather per-batch
    gather_idx = owner_idx.unsqueeze(-1)  # (B, L, 1)
    base_mask = torch.gather(merged_mask.unsqueeze(1).expand(-1, L, -1), dim=2, index=gather_idx)  # (B, L, 1)
    base_mask = base_mask.squeeze(-1)
    return base_mask


def adaptive_mtm_loss(
    logits_base: torch.Tensor,
    targets_base: torch.LongTensor,
    owner_idx: torch.LongTensor,
    merged_feats: torch.Tensor,
    mask_ratio: float = 0.15,
    reduction: str = "mean",
):
    """
    High-level helper: choose merged tokens to mask (by importance),
    map them to base positions, and compute CE on the base-level logits for those positions.

    Args:
      logits_base: (B, L, V) model logits at base resolution (the model must predict masked positions)
      targets_base: (B, L) ground-truth tokens
      owner_idx: (B, L) map base pos -> merged index
      merged_feats: (B, M, D)
      mask_ratio: fraction of merged tokens to mask
      reduction: "mean" or "sum"

    Returns:
      scalar loss
    """
    merged_mask = select_merged_tokens_by_importance(merged_feats, mask_ratio=mask_ratio)
    base_mask = merged_mask_to_base_mask(owner_idx, merged_mask)
    # compute CE only on base positions where base_mask == True
    from .merged_token_reconstruction import merged_token_reconstruction_loss
    return merged_token_reconstruction_loss(logits_base, targets_base, mask=base_mask, reduction=reduction)
