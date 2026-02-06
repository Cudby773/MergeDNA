import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Tuple

def make_optimizer_and_scheduler(
    model: torch.nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    max_steps: int = 100000,
) -> Tuple[torch.optim.Optimizer, object]:
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=max_steps, eta_min=1e-6)
    return opt, scheduler
