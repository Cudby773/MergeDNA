import torch
import torch.nn as nn


class LocalAttention(nn.Module):
    """
    Local-attention implemented by folding windows into the batch dimension
    and applying nn.MultiheadAttention per window (batch_first=True).
    """
    def __init__(self, d_model: int, nhead: int, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model))
        self.ln2 = nn.LayerNorm(d_model)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Args:
            x: (B, L, D)
        Returns:
            x_attn_unfolded: (B, L_trunc, D)  -- attended, unfolded (with padding kept)
            x_windows_attn: (B * num_windows, W, D) -- folded windows after attn
            num_windows: int
        """
        B, L, D = x.shape
        W = self.window_size
        if L % W != 0:
            raise ValueError("LocalAttention expects L divisible by window_size (pad before calling)")

        num_windows = L // W
        # fold windows into batch
        x_windows = x.view(B, num_windows, W, D).reshape(B * num_windows, W, D)  # folded

        # apply MHA per folded window
        attn_out, _ = self.mha(x_windows, x_windows, x_windows, need_weights=False)
        x_windows_attn = x_windows + attn_out
        x_windows_attn = self.ln(x_windows_attn)
        ff = self.ff(x_windows_attn)
        x_windows_attn = self.ln2(x_windows_attn + ff)

        # unfold back to (B, L_trunc, D)
        x_attn_unfolded = x_windows_attn.view(B, num_windows * W, D)

        return x_attn_unfolded, x_windows_attn, num_windows
