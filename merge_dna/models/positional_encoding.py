import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # odd case: last column remains zero for cos
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)


    def forward(self, x):
        # x: (seq_len, batch, d_model) or (batch, seq_len, d_model)
        pe: torch.Tensor = self.pe
        if pe.device != x.device or pe.dtype != x.dtype:
            pe = pe.to(device=x.device, dtype=x.dtype)
            
        if x.dim() == 3 and x.shape[0] == pe.shape[1]:
            # (seq_len, batch, d)
            x = x + pe.squeeze(0).transpose(0, 1)[: x.size(0), :].unsqueeze(1)
            return x
        if x.dim() == 3 and x.shape[1] == pe.shape[1]:
            # (batch, seq_len, d)
            return x + pe[:, : x.size(1), :]
        else:
            # x is (seq_len, batch, d) with seq_len < max_len
            S, _, _ = x.shape
            if S > pe.shape[1]:
                raise ValueError(f"Requested len={S} exceeds max_len={pe.shape[1]}")
            return x + pe[:, :S, :].transpose(0, 1)
