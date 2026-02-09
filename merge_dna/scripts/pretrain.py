import argparse
import torch

from merge_dna.models import (
    LocalEncoder, LatentEncoder, LatentDecoder, LocalDecoder, MergeDNAModel
)
from merge_dna.training.optim import make_optimizer_and_scheduler
from merge_dna.training.trainer import Trainer
from merge_dna.data import make_multi_species_genome_data_loader


def make_model_small(vocab_size=6, embed_dim = 32, latent_d_model=64, nhead=4):
    # Choose layer config to achieve ~50% token length
    layer_configs = [{"window_size": 16, "r": 2}, {"window_size": 16, "r": 2}, {"window_size": 16, "r": 3}, {"window_size": 16, "r": 3}]
    local_enc = LocalEncoder(vocab_size=vocab_size, d_model=embed_dim, nhead=nhead, layer_configs=layer_configs)
    latent_enc = LatentEncoder(d_model=latent_d_model, input_dim=embed_dim, nhead=nhead, num_layers=2)
    latent_dec = LatentDecoder(d_model=latent_d_model, input_dim=latent_enc.d_model, output_dim=embed_dim, num_layers=1)
    local_dec = LocalDecoder(d_model=embed_dim, vocab_size=vocab_size, layer_configs=layer_configs)
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
    args = parse_args()
    device = args.device
    print("device:", device)
    model = make_model_small().to(device)
    opt, _ = make_optimizer_and_scheduler(model, lr=args.lr, max_steps=args.steps)
    dataloader = make_multi_species_genome_data_loader(args.batch, infinite=False, num_workers=2)
    trainer = Trainer(
        model=model, dataloader=dataloader,optimizer=opt,device=device,mask_token_id=None,
    )
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(epoch, log_every=10, max_batches=args.steps)
        print(f"epoch {epoch} finished, metrics={metrics}")
        trainer.save_checkpoint(epoch)


if __name__ == "__main__":
    main()
