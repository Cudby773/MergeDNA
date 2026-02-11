import torch.nn as nn
from merge_dna.models import (
    LocalEncoder, LatentEncoder, LatentDecoder, LocalDecoder
)


class MergeDNAModel(nn.Module):
    def __init__(
        self, 
        local_encoder: LocalEncoder, 
        latent_encoder: LatentEncoder, 
        latent_decoder: LatentDecoder, 
        local_decoder: LocalDecoder,
    ):
        super().__init__()
        self.local_encoder = local_encoder
        self.latent_encoder = latent_encoder
        self.latent_decoder = latent_decoder
        self.local_decoder = local_decoder


    def forward(self, x):
        merged, source_maps, _ = self.local_encoder.forward(x)
        latent, _ = self.latent_encoder.forward(merged, src_mask=None, src_key_padding_mask=None)
        decoded = self.latent_decoder.forward(latent)
        logits = self.local_decoder.forward(decoded, source_maps)
        return logits
    
    
class MergeDNAEncoderModel(nn.Module):
    def __init__(
        self, 
        local_encoder: LocalEncoder, 
        latent_encoder: LatentEncoder, 
    ):
        super().__init__()
        self.local_encoder = local_encoder
        self.latent_encoder = latent_encoder


    def forward(self, x):
        merged, _, _ = self.local_encoder.forward(x)
        latent, _ = self.latent_encoder.forward(merged, src_mask=None, src_key_padding_mask=None)
        return latent