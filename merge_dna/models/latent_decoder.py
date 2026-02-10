import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding


class LatentDecoder(nn.Module):
    """
    Transformer-based latent decoder (symmetric). Accepts a single latent tensor
    and decodes it back to a token-level representation.

    Args:
        input_dim: dimension of incoming latent vectors (e.g., encoder d_model).
        d_model: transformer hidden dimension (attention & FFN width).
        output_dim: final output dimension produced by the decoder.
        nhead: number of attention heads.
        num_layers: number of transformer encoder layers.
        dim_feedforward: feedforward inner dim.
        dropout: dropout prob.
        max_len: max positional encoding length.
        activation: activation in FFN.
        norm_first: whether to apply LayerNorm first in TransformerEncoderLayer.
    Input:
        latent: (B, S, input_dim)
    Output:
        decoded: (B, S, output_dim)
    """

    def __init__(
        self,
        input_dim: int | None = None,
        d_model: int = 512,
        output_dim: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        activation: str = "relu",
        norm_first: bool = False,
    ):
        super().__init__()

        if input_dim is None:
            input_dim = d_model

        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim

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

        if d_model != output_dim:
            self.output_proj = nn.Linear(d_model, output_dim)
        else:
            self.output_proj = nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, latent: torch.Tensor, src_mask=None, src_key_padding_mask=None):
        """
        latent: (B, S, input_dim)
        returns decoded: (B, S, output_dim)
        """
        x = self.input_proj(latent)                 # (B, S, d_model)
        x = self.pos_enc(x)                         # (S, B, d_model)
        x = self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        decoded = self.output_proj(x)               # (B, S, output_dim)
        return decoded
