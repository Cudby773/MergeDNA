import torch.nn as nn

class LatentDecoder(nn.Module):
    """
    Tiny symmetric decoder; here just a linear projection for demonstration.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.proj(x)