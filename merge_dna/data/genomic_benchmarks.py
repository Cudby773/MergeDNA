from genomic_benchmarks.dataset_getters.pytorch_datasets import HumanEnhancersCohn

import torch
from torch.utils.data import DataLoader

from .constants import DNA_MAP


def collate(batch):
    xs, ys = [], []

    for seq, label in batch:
        ys.append(torch.tensor([label], dtype=torch.float32))
        x = torch.tensor([DNA_MAP[c] for c in seq], dtype=torch.long)
        xs.append(x)

    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs, ys

train_dset = HumanEnhancersCohn(split='train', version=0)
test_dset = HumanEnhancersCohn(split='test', version=0)

human_enhancers_cohn_train_loader = DataLoader(train_dset, batch_size=32, shuffle=False, collate_fn=collate)
human_enhancers_cohn_test_loader = DataLoader(test_dset, batch_size=32, shuffle=False, collate_fn=collate)