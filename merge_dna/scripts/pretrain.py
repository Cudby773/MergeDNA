# merge_dna/scripts/pretrain.py
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from merge_dna.models import (
    LocalEncoder, LatentEncoder, LatentDecoder, LocalDecoder, MergeDNAModel, TokenMerge
)
from merge_dna.training.optim import make_optimizer_and_scheduler
from merge_dna.training.trainer import Trainer

# --- synthetic dataset for testing ---
class RandomDNADataset(Dataset):
    def __init__(self, vocab_size=4, seq_len=256, n_samples=1000, seed=42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n = n_samples
        rng = np.random.RandomState(seed)
        self.data = rng.randint(0, vocab_size, size=(n_samples, seq_len)).astype(np.int64)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).long()
        y = x.clone()
        return x, y


def make_model_small(vocab_size=4, d_model=64, nhead=4):
    layer_configs = [{"window_size": 8, "r": 2}, {"window_size": 8, "r": 2}]
    def tm_factory(r):
        return TokenMerge(r=r)
    local_enc = LocalEncoder(vocab_size=vocab_size, d_model=d_model, nhead=nhead, layer_configs=layer_configs, token_merge_factory=tm_factory)
    latent_enc = LatentEncoder(d_model=d_model, nhead=nhead)
    latent_dec = LatentDecoder(d_model=d_model)
    local_dec = LocalDecoder(d_model=d_model, vocab_size=vocab_size)
    model = MergeDNAModel(local_enc, latent_enc, latent_dec, local_dec)
    return model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    return p.parse_args()


def main():
    seed = 42
    args = parse_args()
    device = args.device
    print("device:", device)
    torch.manual_seed(seed)
    model = make_model_small().to(device)
    opt, sched = make_optimizer_and_scheduler(model, lr=args.lr, max_steps=args.steps)
    trainer = Trainer(model=model, optimizer=opt, scheduler=sched, device=device, log_interval=10, save_dir=args.save_dir)
    # tiny dataset and dataloader
    ds = RandomDNADataset(seq_len=args.seq_len, n_samples=1000, seed=seed)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0)

    step = 0
    for epoch in range(args.epochs):
        step, avg_loss = trainer.train_epoch(dl, step_start=step, max_steps=args.steps, amtm_cfg={"enabled": False})
        print(f"epoch {epoch} finished, avg_loss={avg_loss:.4f}")
        trainer.save(step)


if __name__ == "__main__":
    main()
