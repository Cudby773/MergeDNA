import torch
import torch.nn as nn
import argparse
from merge_dna.models import (
    LocalEncoder, LatentEncoder, LatentDecoder, LocalDecoder, MergeDNAModel, TokenMerge
)

def make_fake_batch(batch_size=2, seq_len=32, vocab_size=4, device="cpu"):
    # random DNA tokens 0..3
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

def main(device: str):
    device = torch.device(device)
    # hyperparams
    vocab_size = 4
    d_model = 32

    # instantiate modules
    local_enc = LocalEncoder(
        vocab_size, 
        d_model, 
        nhead=4, 
        layer_configs=[{'window_size': 30, 'r':3}]*2, 
        token_merge_factory=lambda r: TokenMerge(r)
    ).to(device)
    latent_enc = LatentEncoder(d_model).to(device)
    latent_dec = LatentDecoder(d_model).to(device)
    local_dec = LocalDecoder(d_model, vocab_size).to(device)
    merge_dna_model = MergeDNAModel(local_enc, latent_enc, latent_dec, local_dec)

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
        merged, source_index = local_enc(x)         # (B, M, D)
        latent = latent_enc(merged)                 # (B, M, D)
        decoded_latent = latent_dec(latent)         # (B, M, D)
        logits = local_dec(decoded_latent, source_index)  # (B, L_trunc, V)

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