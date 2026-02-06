import torch.nn as nn
from merge_dna.models import (
    LocalEncoder, LatentEncoder, LatentDecoder, LocalDecoder
)
class MergeDNAModel(nn.Module):
    def __init__(self, local_enc: LocalEncoder, latent_enc: LatentEncoder, latent_dec: LatentDecoder, local_dec: LocalDecoder):
        super().__init__()
        self.local_enc = local_enc
        self.latent_enc = latent_enc
        self.latent_dec = latent_dec
        self.local_dec = local_dec

    def forward(self, x):
        merged, source_maps = self.local_enc.forward(x)
        latent = self.latent_enc(merged)
        decoded = self.latent_dec(latent)
        logits = self.local_dec.forward(decoded, source_maps)
        return logits