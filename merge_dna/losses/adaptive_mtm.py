import torch

def sample_topk_from_scores(scores: torch.Tensor, K: int, deterministic: bool = False) -> torch.Tensor:
    """
    scores: (B, M) positive scores (not necessarily normalized)
    returns: mask (B, M) boolean with exactly K True per row
    deterministic -> top-K, otherwise sample without replacement proportional to scores.
    """
    B, M = scores.shape
    if K <= 0:
        return torch.zeros((B, M), dtype=torch.bool, device=scores.device)
    if K >= M:
        return torch.ones((B, M), dtype=torch.bool, device=scores.device)

    if deterministic:
        topk_idx = torch.topk(scores, K, dim=1).indices  # (B, K)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        bs = torch.arange(B, device=scores.device).unsqueeze(1).expand(-1, K)
        mask[bs, topk_idx] = True
        return mask
    else:
        probs = scores / (scores.sum(dim=1, keepdim=True) + 1e-9)
        # torch.multinomial wants non-negative numbers, sampling without replacement
        idx = torch.multinomial(probs, num_samples=K, replacement=False)  # (B, K)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        bs = torch.arange(B, device=scores.device).unsqueeze(1).expand(-1, K)
        mask[bs, idx] = True
        return mask


def mask_base_positions_from_merged_selection(
    x: torch.LongTensor,
    final_source_map: torch.LongTensor,
    merged_mask: torch.BoolTensor,
    mask_token_id: int
):
    """
    Robust mapping: if final_source_map shorter than x, only mask positions that have a mapping.
    Also supports final_source_map containing -1 for unmapped positions.
    Returns x_masked (B, L_x) and base_mask (B, L_x) boolean.
    """
    B, L_x = x.shape
    Bm, L_map = final_source_map.shape
    assert B == Bm, f"batch size mismatch between x {B} and final_source_map {Bm}"

    # merged_mask: (B, M)
    if merged_mask.ndim != 2:
        raise ValueError(f"merged_mask must be 2D (B, M) but got {merged_mask.shape}")

    base_mask = torch.zeros((B, L_x), dtype=torch.bool, device=x.device)

    # Only fill the prefix that final_source_map covers
    L_common = min(L_x, L_map)
    if L_common == 0:
        return x.clone(), base_mask

    # Use gather, but ensure indices are valid: clamp negative indices to 0, and then mask them out later
    gather_idx = final_source_map[:, :L_common].clamp(min=0)   # (B, L_common)
    merged_mask_int = merged_mask.long()                       # (B, M)
    gathered = torch.gather(merged_mask_int, dim=1, index=gather_idx)  # (B, L_common)

    # Where final_source_map had -1, we should set gathered to 0
    invalid_pos = final_source_map[:, :L_common] < 0
    gathered[invalid_pos] = 0

    base_mask[:, :L_common] = gathered.bool()
    x_masked = x.clone()
    x_masked[base_mask] = mask_token_id
    return x_masked, base_mask