import argparse
import torch

from merge_dna.models import (
    LocalEncoder, LatentEncoder, LatentDecoder, LocalDecoder, MergeDNAModel, TokenMerge
)
from merge_dna.training.optim import make_optimizer_and_scheduler
from merge_dna.training.trainer import Trainer
from merge_dna.data import make_multi_species_genome_data_loader


def make_model_small(vocab_size=6, d_model=64, nhead=4):
    layer_configs = [{"window_size": 8, "r": 2}, {"window_size": 8, "r": 2}]
    def tm_factory(r):
        return TokenMerge(r=r)
    local_enc = LocalEncoder(vocab_size=vocab_size, d_model=d_model, nhead=nhead, layer_configs=layer_configs, token_merge_factory=tm_factory)
    latent_enc = LatentEncoder(d_model=d_model, nhead=nhead, num_layers=8)
    latent_dec = LatentDecoder(d_model=d_model, num_layers=4)
    local_dec = LocalDecoder(d_model=d_model, vocab_size=vocab_size)
    model = MergeDNAModel(local_enc, latent_enc, latent_dec, local_dec)
    return model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=16)
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
    step = 0
    for epoch in range(args.epochs):
        loader = make_multi_species_genome_data_loader(args.batch, infinite=False, num_workers=2)
        step, avg_loss = trainer.train_epoch(loader, step_start=step, max_steps=args.steps, amtm_cfg={"enabled": False})
        print(f"epoch {epoch} finished, avg_loss={avg_loss:.4f}")
        trainer.save(step)


if __name__ == "__main__":
    main()
