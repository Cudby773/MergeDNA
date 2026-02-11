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

   
class TokenUnmerge(nn.Module):
    """
    Reverses a series of token merges represented by a list of source maps
    """
    def __init__(self, normalize: bool = True, eps: float = 1e-6):
        super().__init__()
        self.normalize = normalize
        self.eps = eps


    def forward(self, merged_feats: torch.Tensor, source_maps: list[torch.LongTensor]) -> torch.Tensor:
        """
        Args:
            merged_feats: (B, M, D)
        Returns:
            unmerged_feats: (B, N, D)
        """
        x = merged_feats
        for source_map in source_maps[::-1]:
            B, M, D = x.shape
            B2, N = source_map.shape
            if not B == B2:
                raise Exception(f"Batch mismatch between x {B} and source_map {B2}")

            device = merged_feats.device
            if self.normalize:
                ones = torch.ones((B, N), dtype=x.dtype, device=device)
                counts = torch.zeros((B, M), dtype=x.dtype, device=device)
                counts = counts.scatter_add_(1, source_map, ones)  # (B, M)
                counts = counts.clamp_min(self.eps)
                x = x / counts.unsqueeze(-1)  # (B, M, 1)

            source_map_expanded = source_map.unsqueeze(-1).expand(-1, -1, D)  # (B, N, D)
            x = torch.gather(x, dim=1, index=source_map_expanded)  # (B, N, D)
        return x

