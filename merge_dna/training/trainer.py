import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path

from merge_dna.models import (
    LocalEncoder, MergeDNAModel, TokenUnmerge
)
from merge_dna.losses import sample_k_local_tokens_from_source_map


def load_checkpoint(path: str, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt.get("step", 0)
    epoch = ckpt.get("epoch", 0)
    return step, epoch, ckpt.get("config", {})


def ensure_mask_token_in_embedding(local_encoder: LocalEncoder, mask_token_id: int):
    """
    Ensure the LocalEncoder embedding matrix supports mask_token_id. If mask_token_id == vocab_size,
    create a new embedding with vocab_size+1 rows, copy weights, and initialize the mask embedding as the mean.
    This mutates local_enc.emb in-place.
    """
    emb: nn.Embedding = local_encoder.emb
    vocab_size, d_model = emb.num_embeddings, emb.embedding_dim
    if mask_token_id < vocab_size:
        return
    if mask_token_id != vocab_size:
        raise ValueError("This helper only supports adding a single mask token at index == vocab_size")
    new_vocab = vocab_size + 1
    new_emb = nn.Embedding(new_vocab, d_model)
    with torch.no_grad():
        new_emb.weight[:vocab_size].copy_(emb.weight)
        # set mask embedding to mean of previous embeddings
        mean_emb = emb.weight.mean(dim=0, keepdim=True)  # (1, d)
        new_emb.weight[vocab_size : vocab_size + 1].copy_(mean_emb)
    local_encoder.emb = new_emb
    return


def merged_padding_mask_from_source_map(source_maps: list[torch.LongTensor],
                                        orig_mask: torch.BoolTensor,
                                        merged_shapes: list[torch.Size]):
    # orig_mask:   (B, N) bool, True = masked
    if not len(source_maps) == len(merged_shapes):
        raise Exception('Should be same number of maps as shapes')
    new_mask = orig_mask
    for idx, source_map in enumerate(source_maps):
        B, _ = source_map.shape
        device = source_map.device
        merged_masks = []
        num_groups = merged_shapes[idx][1]
        for b in range(B):
            sm = source_map[b, :]    
            m = orig_mask[b].to(torch.long)   # 1 if masked, 0 if keep
            # count unmasked per group
            unmasked_per_group = torch.zeros(num_groups, device=device, dtype=torch.long)
            unmasked_per_group.scatter_add_(0, sm, (1 - m))  # adds 1 for each unmasked source token
            # group is masked iff unmasked count == 0
            group_mask = (unmasked_per_group == 0)          # True = masked
            merged_masks.append(group_mask)
        
        new_mask = torch.stack(merged_masks, dim=0)   # (B, M_new)
    return new_mask


class Trainer:
    def __init__(
        self,
        model: MergeDNAModel,
        dataloader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        mask_token_id: Optional[int] = None,
        K_for_AMTM: int = 8,
        lambda_latent: float = 0.25,
        deterministic_selection: bool = False,
        checkpoint_dir: str = "checkpoints",
        checkpoint_every: int = 100,
    ):
        """
        model: a wrapper MergeDNAModel that exposes .local_encoder, .latent_encoder, .latent_decoder, .local_decoder
        dataloader: yields batches of shape (B, L) long tensors (base tokens)
        mask_token_id: integer token id to use for masking base tokens. If None, will be set to vocab_size and
                       embedding extended automatically.
        """
        self.model = model
        self.dataloader = dataloader
        self.opt = optimizer
        self.device = device
        self.K = K_for_AMTM
        self.lambda_latent = lambda_latent
        self.deterministic_selection = deterministic_selection

        self.local_encoder = model.local_encoder
        vocab_size = self.local_encoder.emb.num_embeddings
        if mask_token_id is None:
            mask_token_id = vocab_size
        self.mask_token_id = mask_token_id
        if mask_token_id >= vocab_size:
            ensure_mask_token_in_embedding(self.local_encoder, mask_token_id)

        self.model.to(self.device)
        self.local_encoder.emb.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        
        self.checkpoint_every = checkpoint_every
        self.global_step = 0

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        
    def save_checkpoint(self, epoch: int):
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "step": self.global_step,
            "epoch": epoch,
            "config": {
                "K_for_AMTM": self.K,
                "lambda_latent": self.lambda_latent,
                "deterministic_selection": self.deterministic_selection,
                "mask_token_id": self.mask_token_id,
            },
        }

        path = self.checkpoint_dir / f"step_{self.global_step:07d}.pt"
        tmp_path = path.with_suffix(".pt.tmp")

        torch.save(ckpt, tmp_path)
        tmp_path.replace(path)

        print(f"[checkpoint] saved to {path}")


    def compute_losses(self, x: torch.LongTensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        x: (B, N) base token ids
        returns: total_loss (scalar), losses dict
        """
        model = self.model
        device = self.device
        x = x.to(device)

        B, N = x.shape

        # ---------------------------
        # 1) Full MTR (end-to-end)
        # ---------------------------
        merged, source_maps, _ = model.local_encoder.forward(x)  # merged: (B, L, D)
        latent, _ = model.latent_encoder.forward(merged)                 # (B, L, d_model)
        decoded_latent = model.latent_decoder.forward(latent)        # (B, L, out_dim)
        logits = model.local_decoder.forward(decoded_latent, source_maps)  # (B, N, V)

        # CE over all base positions (flatten)
        logits_flat = logits.view(-1, logits.size(-1))  # (B*N, V)
        targets_flat = x.view(-1)                       # (B*N,)
        perpos_loss = self.loss_fn(logits_flat, targets_flat).view(B, N)  # (B, N)
        L_MTR = perpos_loss.mean()

        # ---------------------------
        # 2) Latent-only MTR (tokenizer frozen)
        # ---------------------------
        # detach merged to prevent grads to tokenizer
        merged_detached = merged.detach()
        latent_det, latent_source_map = model.latent_encoder.forward(merged_detached, token_merge=True)
        decoded_det = model.latent_decoder(latent_det)
        logits_det = model.local_decoder(decoded_det, source_maps)
        perpos_loss2 = self.loss_fn(logits_det.view(-1, logits_det.size(-1)), targets_flat).view(B, N)
        L_MTR_latent = perpos_loss2.mean()


        # ---------------------------
        # 3) AMTM (Adaptive Masked Token Modeling)
        # ---------------------------  
        local_tokens = sample_k_local_tokens_from_source_map(latent_source_map)
        local_tokens_mask = torch.zeros((B, latent_source_map.shape[1]), dtype=torch.bool, device=local_tokens.device)
        B, _ = local_tokens.shape
        batch_indices = torch.arange(B, device=local_tokens.device).unsqueeze(1)
        local_tokens_mask[batch_indices, local_tokens] = True
        local_tokens_mask = local_tokens_mask.unsqueeze(-1)
        unmerger = TokenUnmerge(normalize=False)
        base_mask = unmerger.forward(local_tokens_mask, source_maps)
        base_mask = base_mask.squeeze()
        mask_bool = base_mask.to(torch.bool)
        x_masked = torch.where(
            mask_bool,
            self.mask_token_id,
            x
        )
        
        merged_masked, source_maps_masked, merged_shapes = model.local_encoder.forward(x_masked) 
        merged_bool_mask = merged_padding_mask_from_source_map(source_maps_masked, mask_bool, merged_shapes)
        latent_masked, _ = model.latent_encoder.forward(merged_masked, src_key_padding_mask=merged_bool_mask)
        decoded_masked = model.latent_decoder.forward(latent_masked)
        logits_masked = model.local_decoder.forward(decoded_masked, source_maps_masked)  

        # compute AMTM loss: negative log-likelihood over masked base positions
        logp = F.log_softmax(logits_masked, dim=-1)  
        # gather probability of true tokens
        true_logp = logp.gather(dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
        masked_vals = true_logp * base_mask
        denom = base_mask.sum()
        if denom == 0:
            L_AMTM = torch.tensor(0.0, device=device)
        else:
            L_AMTM = -(masked_vals.sum() / denom)

        total_loss = L_MTR + self.lambda_latent * L_MTR_latent + L_AMTM

        metrics = {
            "L_MTR": float(L_MTR.detach().cpu().item()),
            "L_MTR_latent": float(L_MTR_latent.detach().cpu().item()),
            "L_AMTM": float(L_AMTM.detach().cpu().item()),
            "total_loss": float(total_loss.detach().cpu().item()),
            "masked_tokens": int(denom.item()) if denom is not None else 0,
        }
        return total_loss, metrics


    def train_step(self, batch: torch.LongTensor, epoch: int) -> dict:
        """
        One optimization step given a batch of base tokens shape (B, L)
        """
        self.model.train()
        self.opt.zero_grad()
        total_loss, metrics = self.compute_losses(batch)
        total_loss.backward()
        self.opt.step()
        
        self.global_step += 1
        if self.global_step % self.checkpoint_every == 0:
            self.save_checkpoint(epoch)
            
        return metrics


    def train_epoch(self, epoch: int = 0, log_every: int = 10, max_batches: Optional[int] = None):
        """
        Run one epoch through dataloader (or a fraction if max_batches set)
        """
        running = {"L_MTR": 0.0, "L_MTR_latent": 0.0, "L_AMTM": 0.0, "total_loss": 0.0, "masked_tokens": 0}
        count = 0
        for i, batch in enumerate(self.dataloader):
            if max_batches is not None and i >= max_batches:
                break
            metrics = self.train_step(batch, epoch)
            for k in running:
                if k in metrics:
                    running[k] += metrics[k]
            count += 1
            if i % log_every == 0:
                print(f"[epoch {epoch}] step {i} metrics: {metrics}")
        avg = {k: (running[k] / count if count > 0 else 0.0) for k in running}
        print(f"[epoch {epoch}] avg metrics: {avg}")
        return avg
