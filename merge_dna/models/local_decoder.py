import torch
import torch.nn as nn
import torch.nn.functional as F
from .local_attention import LocalAttention
from .token_unmerge import TokenUnmerge


class LocalAttentionBlock(nn.Module):
    """
    Single local decoder layer with local-attention.
    """
    def __init__(self, d_model: int, nhead: int, window_size: int):
        super().__init__()
        self.attn = LocalAttention(d_model, nhead, window_size)
        self.window_size = window_size
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) token indices
        B, L, D = x.shape
        W = self.window_size

        pad_len = (W - (L % W)) % W
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len)) 
        L_pad = x.shape[1]

        x_windows_attn = self.attn.forward(x)  # (L_pad, B, D)
        x_attn = x_windows_attn.view(B, L_pad, D)
        if pad_len > 0:
            x_attn = x_attn[:, :L, :]

        return x_attn


class LocalDecoder(nn.Module):
    """
    Symmetric local decoder that reverses LocalEncoder's token merges.
    """
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        layer_configs: list[dict],
        nhead: int = 4,
        normalize_unmerge: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.layer_configs = layer_configs
        self.unmerge = TokenUnmerge(normalize=normalize_unmerge)
        self.layers = nn.ModuleList()
        for cfg in layer_configs:
            W = cfg['window_size']            
            block = LocalAttentionBlock(d_model=d_model, nhead=nhead, window_size=W)
            self.layers.append(block)

        # final projection to vocab
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.final_ln = nn.LayerNorm(d_model)


    def forward(self, merged_feats: torch.Tensor, source_maps: list[torch.LongTensor]) -> torch.Tensor:
        """
        merged_feats: (B, L_last, D)   -- features at the most-compressed level
        source maps: list of returned token merge source_maps
        Returns:
            logits: (B, L_orig, V)
        """
        x = merged_feats
        x = self.unmerge(x, source_maps)
        for layer in self.layers:
            x = layer(x) 
        
        x = self.final_ln(x)
        logits = self.out_proj(x)
        return logits
