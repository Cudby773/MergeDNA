import torch.nn as nn

class Heads(nn.Module):
    """
    Separate heads for reconstruction / MLM as needed.
    Here we provide a single reconstruction head (maps features back to logits).
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, features):
        # features: (B, L, D)
        return self.head(features)