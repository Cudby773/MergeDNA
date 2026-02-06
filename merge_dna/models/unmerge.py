import torch
import torch.nn as nn
      

def compose_source_maps(source_maps: list[torch.LongTensor]) -> torch.Tensor:
    """
    Compose a list of source maps to map original positions to final merged-token indices.
    """
    if not source_maps:
        raise ValueError("source_maps list is empty")
    source = source_maps[0] 
    for next_map in source_maps[1:]:
        source = torch.gather(next_map, dim=1, index=source)  
    return source

   
class Unmerge(nn.Module):
    """
    Inverse of TokenMerge.
    Expects:
        merged: (B, M, D)
        source_index: (B, L) 
    Returns:
        unmerged: (B, L, D)
    """
    def __init__(self, normalize: bool = True, eps: float = 1e-6):
        super().__init__()
        self.normalize = normalize
        self.eps = eps


    def forward(self, merged_feats: torch.Tensor, source_maps: list[torch.LongTensor]) -> torch.Tensor:
        """
        Args:
            merged_feats: (B, M, D)
            source_idx:    (B, L) long mapping each original position to merged index in [0..M-1]

        Returns:
            unmerged_feats: (B, L, D) where position j receives merged_feats[..., source_idx[..., j], :]
        """
        B, M, D = merged_feats.shape
        source_idx = compose_source_maps(source_maps)
        B2, L = source_idx.shape
        assert B == B2, "batch mismatch between merged_feats and source_idx"

        device = merged_feats.device
        # Optionally normalize merged_feats by counts per merged token
        if self.normalize:
            # compute counts per merged token: (B, M)
            ones = torch.ones((B, L), dtype=merged_feats.dtype, device=device)
            counts = torch.zeros((B, M), dtype=merged_feats.dtype, device=device)
            counts = counts.scatter_add_(1, source_idx, ones)  # (B, M)
            # clamp to avoid div by zero if some merged tokens are unused (shouldn't happen)
            counts = counts.clamp_min(self.eps)
            # divide merged_feats by counts per merged token
            merged_feats = merged_feats / counts.unsqueeze(-1)  # (B, M, 1) -> broadcast

        # gather: expand source_idx to (B, L, D) index tensor and gather along dim=1
        source_idx_exp = source_idx.unsqueeze(-1).expand(-1, -1, D)  # (B, L, D)
        unmerged = torch.gather(merged_feats, dim=1, index=source_idx_exp)  # (B, L, D)
        return unmerged

