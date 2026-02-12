import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding
from .token_merge import TokenMerge
from .token_unmerge import TokenUnmerge
from typing import Optional

class LatentEncoder(nn.Module):
    """
    Transformer-based latent encoder.

    Args:
        input_dim: dimension of input embeddings (if different from d_model).
        d_model: transformer model dim.
        nhead: number of attention heads.
        num_layers: number of transformer encoder layers.
        dim_feedforward: hidden size in feedforward layers.
        dropout: dropout probability.
        max_len: max positional encoding length.
        activation: activation function
        norm_first: whether to apply LayerNorm first in TransformerEncoderLayer.
    Input:
        x: (batch, seq_len, input_dim)
    Output:
        memory: (batch, seq_len, d_model)
    """
    def __init__(
        self,
        input_dim: int = 512,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        activation: str = "relu",
        norm_first: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.token_merge = TokenMerge(r=0, protect_cls=True)
        self.token_unmerge = TokenUnmerge()

        if input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(
        self, 
        x: torch.Tensor,
        src_mask=None, 
        src_key_padding_mask=None, 
        token_merge=False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: (B, L, input_dim)
        returns latent: (B, L, d_model) and source_map if token_merge is True
        """
        x = self.input_proj(x)               # (B, L, d_model)
        x = self.pos_enc(x)                  # (B, L, d_model)
        
        if token_merge:
            L = x.shape[1]
            desired_r = L // 2
            self.token_merge.r = desired_r
            merged, source_map = self.token_merge(x, x)
            merged = self.pos_enc(merged)
        
            latent = self.transformer_encoder(merged, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            latent = self.token_unmerge(latent, [source_map])
            return latent, source_map
        else:
            latent = self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            return latent, None
