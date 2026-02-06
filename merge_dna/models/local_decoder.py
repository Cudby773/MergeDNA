import torch
import torch.nn as nn
import torch.nn.functional as F
from .unmerge import Unmerge
    
    
class LocalDecoder(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, conv_kernel: int = 3, normalize_unmerge: bool = True):
        super().__init__()
        assert conv_kernel % 2 == 1, "conv_kernel should be odd"
        self.unmerge = Unmerge(normalize=normalize_unmerge)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, padding=(conv_kernel // 2))
        self.ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)


    def forward(self, merged_feats: torch.Tensor, source_maps: list[torch.LongTensor]) -> torch.Tensor:
        """
        Args:
          merged_feats: (B, M, D)
          owner_idx:    (B, L) mapping original positions -> merged index (final mapping)
        Returns:
          logits: (B, L, V)
        """
        # 1) expand to per-base features
        unmerged = self.unmerge.forward(merged_feats, source_maps)  # (B, L, D)

        # 2) local conv expects (B, D, L)
        x = unmerged.transpose(1, 2)  # (B, D, L)
        x = self.conv.forward(x)
        x = x.transpose(1, 2)  # (B, L, D)

        # 3) small residual + layernorm
        x = self.ln(x + unmerged)

        # 4) project to vocab logits
        logits = self.out_proj(x)  # (B, L, V)
        return logits