import torch
import torch.nn as nn
import math
    
    
class TokenMerge(nn.Module):
    """
    Bipartite merge using a top-r selection per batch entry.
    """
    def __init__(self, r: int, protect_cls: bool = True):
        """
        Args:
            r: number of 'a' tokens to merge (per sample in batch)
            protect_cls: if True, forces a[0] not to be merged by setting its row scores to -inf
        """
        super().__init__()
        self.r = int(r)
        self.protect_cls = protect_cls

        self.register_buffer("_pos_a", None)   
        self.register_buffer("_pos_b", None)   
        self.register_buffer("_unm_idx", None)


    def _ensure_buffers(self, T: int, device: torch.device):
        """
        Create pos_a and pos_b for sequence length T and register them as buffers
        (if not already present or length changed). unm_idx depends on batch and is set later.
        """
        # require even T for this simple implementation
        # if T % 2 != 0:
        #     raise ValueError(f"TokenMerge currently expects even sequence length T, got {T}")

        pos_a = torch.arange(0, T, 2, device=device, dtype=torch.long)  # (Ta,)
        pos_b = torch.arange(1, T, 2, device=device, dtype=torch.long)  # (Tb,)

        current_pos_a = getattr(self, "_pos_a")
        if (current_pos_a is None) or (current_pos_a.numel() != pos_a.numel()):
            self._pos_a = pos_a
            self._pos_b = pos_b


    def forward(self, x: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C) input features
            k: (B, T, C) keys/scores for matching
        Returns:
            merged: (B, M_new, C)
            source_index: (B, T_trunc) long tensor mapping each original position to new merged index
        """
        B, T, C = x.shape
        device = x.device

        self._ensure_buffers(T, device)
        pos_a = self._pos_a            # (Ta,)
        pos_b = self._pos_b            # (Tb,)
        Tb = pos_b.shape[0]
        # assert Ta == Tb, "This implementation assumes Ta == Tb (even T)."

        # split into a and b sequences (interleaved)
        a = k[:, ::2, :]   # (B, Ta, C)
        b = k[:, 1::2, :]  # (B, Tb, C)

        # normalize keys for cosine similarity (stabilize)
        a = a / (a.norm(dim=-1, keepdim=True) + 1e-12)
        b = b / (b.norm(dim=-1, keepdim=True) + 1e-12)

        # similarity (B, Ta, Tb)
        scores = a @ b.transpose(-1, -2)

        if self.protect_cls:
            # prevent merging the first a token (common in CLS-style setups)
            # set its row to -inf so it never gets top-k selected
            scores[:, 0, :] = -math.inf

        # best matching b index for each a (B, Ta)
        node_max, node_idx = scores.max(dim=-1)

        # order 'a' tokens by descending match score and pick top-r as src
        edge_order = node_max.argsort(dim=-1, descending=True)  # (B, Ta)
        src_idx = edge_order[:, :self.r]       # (B, r) indices into a
        unm_idx = edge_order[:, self.r:]       # (B, Ta-r) indices into a (unsorted)
        # sort unm_idx so the kept a tokens appear in increasing order
        unm_sorted, _ = unm_idx.sort(dim=-1)    # (B, Ta-r)

        # corresponding dst indices for selected srcs (indices into b)
        dst_idx = node_idx.gather(dim=-1, index=src_idx)  # (B, r)

        src_pos = pos_a[src_idx]          # (B, r)
        unm_pos = pos_a[unm_sorted]       # (B, Ta-r)

        # store unm_sorted as a buffer (per-batch) for inspection/debugging.
        self._unm_idx = unm_sorted

        # now form the actual merged features from x
        src = x[:, ::2, :]   # (B, Ta, C)
        dst = x[:, 1::2, :]  # (B, Tb, C)

        # gather kept 'a' features
        unm = torch.gather(src, dim=1, index=unm_sorted.unsqueeze(-1).expand(-1, -1, C))  # (B, Ta-r, C)

        # gather src features to merge
        src_sel = torch.gather(src, dim=1, index=src_idx.unsqueeze(-1).expand(-1, -1, C))  # (B, r, C)

        # scatter-add src_sel into dst at positions dst_idx
        dst_updated = dst.clone()
        dst_updated = dst_updated.scatter_add(1, dst_idx.unsqueeze(-1).expand(-1, -1, C), src_sel)  # (B, Tb, C)

        # new merged sequence: [kept a tokens] ++ [all b tokens]
        merged = torch.cat([unm, dst_updated], dim=1)  # (B, (Ta-r)+Tb, C)

        # Build source_index: map each original absolute position j in [0..T-1] to merged index m in [0..M_new-1]
        source_index = torch.full((B, T), -1, dtype=torch.long, device=device)

        # indexes for kept 'a' tokens: new indices 0..(Ta-r-1)
        a_new_indices = torch.arange(unm.shape[1], device=device).unsqueeze(0).expand(B, -1)  # (B, Ta-r)
        # place into source_index at absolute positions unm_pos
        source_index.scatter_(dim=1, index=unm_pos, src=a_new_indices)

        # indexes for b tokens: base offset + 0..Tb-1
        base_b = unm.shape[1]
        b_new_indices = base_b + torch.arange(Tb, device=device).unsqueeze(0).expand(B, -1)  # (B, Tb)
        source_index.scatter_(dim=1, index=pos_b.unsqueeze(0).expand(B, -1), src=b_new_indices)

        # src tokens (merged a tokens) map to the new index of their destination b token:
        src_to_new = base_b + dst_idx  # (B, r)
        source_index.scatter_(dim=1, index=src_pos, src=src_to_new)

        # sanity check: no -1 left (all positions assigned)
        if (source_index < 0).any():
            raise RuntimeError("Some original positions were not assigned an owner index. Check matching shapes.")

        return merged, source_index

    