import torch.nn as nn

class Heads(nn.Module):
    """
    Separate heads for reconstruction tasks.
    TODO: further implementations
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, features):
        # features: (B, L, D)
        return self.head(features)