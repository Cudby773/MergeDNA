import os
import time
import torch
from typing import Optional

from merge_dna.losses.merged_token_reconstruction import merged_token_reconstruction_loss
from merge_dna.losses.adaptive_mtm import adaptive_mtm_loss

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cpu",
        log_interval: int = 10,
        save_dir: str= "./checkpoints",
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.optim = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.GradScaler("cuda", enabled=self.use_amp)
        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, step: int):
        path = os.path.join(self.save_dir, f"checkpoint_{step}.pt")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "step": step,
            },
            path,
        )
        print(f"[trainer] saved checkpoint {path}")


    def train_epoch(self, dataloader, step_start: int = 0, max_steps: Optional[int] = None, amtm_cfg: Optional[dict] = None):
        """
        Single epoch training loop (iterates dataloader until exhausted or max_steps reached).
        dataloader yields (input_tokens, target_tokens).
        amtm_cfg: dict with keys {"enabled": bool, "mask_ratio": float}. If None, disables AMTM.
        """
        print("[trainer] train_epoch starting; step_start=", step_start, "max_steps=", max_steps)
        
        model = self.model
        model.train()
        total_loss = 0.0
        t0 = time.time()
        step = step_start
        batches_consumed = 0
        
        data_iter = iter(dataloader)
        try:
            while max_steps is None or step < max_steps:
                batch = next(data_iter)   # if loader empty, this raises StopIteration
                batches_consumed += 1
        
                inputs, targets = batch['input_ids'], batch['labels']  # expect (B, L) and (B, L)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward + compute losses:
                # We expect model(x) to return logits at base resolution.
                # If MergeDNAModel also exposes merged features and owner maps, the trainer can use them for AMTM.
                with torch.autocast("cuda", enabled=self.use_amp):
                    outputs = model(inputs)  # Either returns logits or (logits, metadata)
                    # ADAPT HERE if your model returns (logits, metadata)
                    if isinstance(outputs, tuple) or isinstance(outputs, list):
                        logits = outputs[0]
                        meta = outputs[1] if len(outputs) > 1 else {}
                    else:
                        logits = outputs
                        meta = {}

                    # ensure logits shape (B, L_trunc, V)
                    # align targets to logits length (truncate targets)
                    if logits.shape[1] < targets.shape[1]:
                        targets_trunc = targets[:, : logits.shape[1]]
                    else:
                        # If logits longer than targets (unlikely), pad targets with ignore index (-100)
                        pad_len = logits.shape[1] - targets.shape[1]
                        if pad_len > 0:
                            targets_trunc = torch.cat([targets, torch.full((targets.shape[0], pad_len), -100, dtype=targets.dtype, device=targets.device)], dim=1)
                        else:
                            targets_trunc = targets

                    # Reconstruction loss (base-level CE)
                    loss_rec = merged_token_reconstruction_loss(logits, targets_trunc, mask=None, reduction="mean")

                    # Optional AMTM: only if model provides merged features & owner_final in meta
                    loss_mtm = torch.tensor(0.0, device=self.device)
                    if amtm_cfg and amtm_cfg.get("enabled", False):
                        # meta should contain:
                        #   merged_feats: (B, M, D)
                        #   owner_final: (B, L_trunc) mapping base pos -> merged idx
                        merged_feats = meta.get("merged_feats", None)
                        owner_final = meta.get("owner_final", None)
                        if merged_feats is None or owner_final is None:
                            # Can't compute AMTM — skip but warn
                            print("[trainer] AMTM requested but model didn't return merged_feats/owner_final in meta; skipping AMTM for this batch.")
                        else:
                            loss_mtm = adaptive_mtm_loss(
                                logits_base=logits,
                                targets_base=targets_trunc,
                                owner_idx=owner_final,
                                merged_feats=merged_feats,
                                mask_ratio=amtm_cfg.get("mask_ratio", 0.15),
                                reduction="mean",
                            )

                    # combine losses with weights
                    alpha = amtm_cfg.get("alpha", 1.0) if amtm_cfg else 0.0
                    total_batch_loss = loss_rec + alpha * loss_mtm

                # backward + step
                self.optim.zero_grad()
                self.scaler.scale(total_batch_loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optim)
                self.scaler.update()
                if self.scheduler is not None:
                    try:
                        self.scheduler.step()
                    except Exception:
                        pass

                total_loss += total_batch_loss.item()
                step += 1

                if step % self.log_interval == 0:
                    avg_loss = total_loss / (step - step_start + 1e-12)
                    elapsed = time.time() - t0
                    print(f"[train] step={step} avg_loss={avg_loss:.4f} lr={self.optim.param_groups[0]['lr']:.3e} time={elapsed:.1f}s")
                if step % (self.log_interval * 10) == 0:
                    self.save(step)
        except StopIteration:
            print("[trainer] DataLoader iterator exhausted after", batches_consumed, "batches")
        
        print("[trainer] train_epoch done; consumed", batches_consumed, "batches; final step=", step)
        return step, total_loss / max(1, (step - step_start))

