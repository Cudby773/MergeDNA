import torch.nn as nn
from merge_dna.models import (
    LocalEncoder, LatentEncoder, LatentDecoder, LocalDecoder
)
class MergeDNAModel(nn.Module):
    def __init__(
        self, 
        local_enc: LocalEncoder, 
        latent_encoder: LatentEncoder, 
        latent_decoder: LatentDecoder, 
        local_dec: LocalDecoder,
        latent_enc_config: dict | None = None,
        latent_dec_config: dict | None = None,
    ):
        super().__init__()
        self.local_enc = local_enc
        self.latent_encoder = latent_encoder
        self.latent_decoder = latent_decoder
        self.local_dec = local_dec
        self.latent_enc_config = latent_enc_config
        self.latent_dec_config = latent_dec_config


    def forward(self, x):
        merged, source_maps = self.local_enc.forward(x)
        latent = self.latent_encoder.forward(merged, src_mask=None, src_key_padding_mask=None)
        decoded = self.latent_decoder.forward(latent)
        logits = self.local_dec.forward(decoded, source_maps)
        return logits