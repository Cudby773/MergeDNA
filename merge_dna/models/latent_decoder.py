import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding


class LatentDecoder(nn.Module):
    """
    Transformer-based latent decoder. Symmetric to latent_encoder. 

    Args:
        input_dim: dimension of input embeddings (if different from d_model).
        d_model: transformer model dim.
        nhead: number of attention heads.
        num_layers: number of transformer encoder layers.
        dim_feedforward: hidden size in feedforward layers.
        dropout: dropout probability.
        max_len: max positional encoding length.
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
            batch_first=False,
            norm_first=norm_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._reset_parameters()

    def _reset_parameters(self):
        # follow PyTorch transformer default initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, src_mask=None, src_key_padding_mask=None):
        """
        x: (B, S, input_dim)
        returns memory: (B, S, d_model)
        """
        x = self.input_proj(x)               # (B, S, d_model)
        x = x.transpose(0, 1)                # (S, B, d_model) for PyTorch transformer
        x = self.pos_enc(x)                  # (S, B, d_model)
        memory = self.transformer_encoder.forward(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        memory = memory.transpose(0, 1)      # (B, S, d_model)
        return memory

