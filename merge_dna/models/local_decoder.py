import torch
import torch.nn as nn
import torch.nn.functional as F
from .local_attention import LocalAttention
from .unmerge import Unmerge


class LocalUnmergeBlock(nn.Module):
    """
    One unmerge block: use source_map to expand merged_feats -> expanded tokens,
    then run LocalAttention.

    - r is implicit (derived from source_map mapping)
    - window_size must match the corresponding encoder block
    """
    def __init__(
        self,
        d_model: int,
        window_size: int,
        nhead: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.attn = LocalAttention(d_model=d_model, nhead=nhead, window_size=window_size)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, expanded: torch.Tensor) -> torch.Tensor:
        """
        expanded: (B, L_expanded, D)
        returns refined: (B, L_expanded, D)
        """
        L = expanded.shape[1]
        W = self.window_size
        pad_len = (W - (L % W)) % W
        if pad_len > 0:
            expanded = F.pad(expanded, (0, 0, 0, pad_len))  # pads seq dim on the right

        x_attn_unfolded, _, _ = self.attn.forward(expanded)

        # trim any padded positions
        if pad_len > 0:
            x_attn_unfolded = x_attn_unfolded[:, :L, :]
            expanded = expanded[:, :L, :]
        return self.ln(expanded + x_attn_unfolded)


class LocalDecoder(nn.Module):
    """
    Symmetric local decoder that reverses LocalEncoder's layer_configs.
    Usage:
        - merged_feats: final features from encoder/latent decoder (B, M_last, D)
        - source_maps: list of source_index tensors returned by LocalEncoder, one per layer,
                       in the same order encoder applied layers. Each source_map is (B, L_layer)
                       representing mapping from original positions -> merged index for that layer.
    NOTE: layer_configs must be the same list passed to LocalEncoder so the decoder knows window sizes.
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
        # Store configs in encoder order; decoder will iterate reversed(...)
        self.layer_configs = layer_configs
        self.unmerge = Unmerge(normalize=normalize_unmerge)
        # build a LocalUnmergeBlock per layer (in encoder order)
        self.blocks = nn.ModuleList()
        for cfg in self.layer_configs:
            W = cfg["window_size"]
            blk = LocalUnmergeBlock(d_model=d_model, window_size=W, nhead=nhead)
            self.blocks.append(blk)

        # final projection to vocab
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.final_ln = nn.LayerNorm(d_model)


    def forward(self, merged_feats: torch.Tensor, source_maps: list[torch.LongTensor]) -> torch.Tensor:
        """
        merged_feats: (B, M_last, D)   -- features at the most-compressed level
        source_maps: list of tensors (one per layer) returned by LocalEncoder.forward, same order encoder used.
                     Each tensor has shape (B, L_layer) mapping absolute positions -> merged index for that layer
        Returns:
            logits: (B, L_orig, V)
        """
        # verify lengths match
        if len(source_maps) != len(self.layer_configs):
            raise ValueError("source_maps length must equal number of local encoder layers (layer_configs)")

        x = merged_feats  # start with most-compressed features
        for idx in range(len(source_maps)):
            src_map = source_maps[-(idx+1)]
            block = self.blocks[(-idx+1)]
            x = self.unmerge.forward(x, src_map)  # (B, L_next, D)
            x = block(x)  # (B, L_next, D)
        
        # final normalization + projection
        x = self.final_ln(x)
        logits = self.out_proj(x)  # (B, L_orig, V)
        return logits
