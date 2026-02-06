import torch.nn as nn

class LatentEncoder(nn.Module):
    """
    Tiny latent encoder: a single Transformer-like block (self-attention via nn.MultiheadAttention)
    """
    def __init__(self, d_model: int, nhead: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model*4), nn.ReLU(), nn.Linear(d_model*4, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, M, D)
        attn_out, _ = self.attn(x, x, x, need_weights=False)  # (B, M, D)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x