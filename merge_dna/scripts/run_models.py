import torch
import torch.nn as nn
import argparse
from merge_dna.models import (
    LocalEncoder, LatentEncoder, LatentDecoder, LocalDecoder, MergeDNAModel, TokenMerge
)

def make_fake_batch(batch_size=2, seq_len=32, vocab_size=4, device="cpu"):
    # random DNA tokens 0..3
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

def make_model_small(vocab_size=6, embed_dim = 32, latent_d_model=64, nhead=4):
    layer_configs = [{"window_size": 8, "r": 2}, {"window_size": 8, "r": 2}]
    local_enc = LocalEncoder(vocab_size=vocab_size, d_model=embed_dim, nhead=nhead, layer_configs=layer_configs)
    latent_enc = LatentEncoder(d_model=latent_d_model, input_dim=embed_dim, nhead=nhead, num_layers=2)
    latent_dec = LatentDecoder(d_model=latent_d_model, input_dim=latent_enc.d_model, output_dim=embed_dim, num_layers=1)
    local_dec = LocalDecoder(d_model=embed_dim, vocab_size=vocab_size, layer_configs=layer_configs)
    model = MergeDNAModel(local_enc, latent_enc, latent_dec, local_dec)
    return model


def main(device: str):
    device = torch.device(device)
    # hyperparams
    vocab_size = 4
    embed_dim = 32
    latent_d_model = 64
    merge_dna_model = make_model_small(embed_dim, latent_d_model)

    # toy optimizer
    params = merge_dna_model.parameters()
    opt = torch.optim.Adam(params, lr=1e-3)

    for i in range(20):
        # make fake data
        x = make_fake_batch(batch_size=2, seq_len=32, vocab_size=vocab_size, device=device)
        target = x.clone()  # reconstruction target at base resolution
        
        if i == 0:
            try:
                from torchinfo import summary
                summary(
                    merge_dna_model,
                    input_size=x.shape,
                    dtypes=[torch.long],
                    device=x.device
                )
            except:
                pass
            
        # forward
        merged, source_index, scores = merge_dna_model.local_encoder(x)         # (B, S, D)
        latent = merge_dna_model.latent_encoder(merged)                 # (B, S, input_dim) -> (B, S, d_model)
        decoded_latent = merge_dna_model.latent_decoder(latent)         # (B, S, d_model) -> (B, S, output_dim)
        logits = merge_dna_model.local_decoder(decoded_latent, source_index)  # (B, L_trunc, V)

        print(f'x.shape: {x.shape}')
        print(f'merged.shape: {merged.shape}')
        print(f'latent.shape: {latent.shape}')
        print(f'decoded_latent.shape: {decoded_latent.shape}')
        print(f'logits.shape: {logits.shape}')

        # loss: cross-entropy over flattened positions
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))
        print(f"loss = {loss.item():.6f}")

        # backward
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("one optimization step done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(args.device)