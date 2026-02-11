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
            x: (B, T_prev, C) input features (the sequence passed to this TokenMerge)
            k: (B, T_prev, C) keys used for matching (may be same as x or projected)
        Returns:
            merged: (B, M_new, C)
            source_map: (B, T_prev) long mapping from indices in this layer's input -> new merged idx (or -1)
            merge_scores: (B, M_new) float scores per merged token (norm or learned)
        """
        B, T_prev, C = x.shape
        device = x.device
        
        self._ensure_buffers(T_prev, device)
        pos_a = self._pos_a  
        pos_b = self._pos_b 
        Tb = pos_b.shape[0]

        # split into a and b
        a = k[:, ::2, :]   # (B, Ta, C)
        b = k[:, 1::2, :]  # (B, Tb, C)

        a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-12)
        b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-12)
        sim = a_norm @ b_norm.transpose(-1, -2)

        if self.protect_cls:
            sim[:, 0, :] = -math.inf

        node_max, node_idx = sim.max(dim=-1)  # (B, Ta)  , node_idx indexes into 0..Tb-1
        edge_order = node_max.argsort(dim=-1, descending=True)  # (B, Ta)
        src_idx = edge_order[:, : self.r]    # (B, r) indices into a (positions in a segment)
        unm_idx = edge_order[:, self.r:]    # (B, Ta-r)
        unm_sorted, _ = unm_idx.sort(dim=-1)  # (B, Ta-r)

        src = x[:, ::2, :]  # (B, Ta, C)
        dst = x[:, 1::2, :]  # (B, Tb, C)

        src_sel = torch.gather(src, dim=1, index=src_idx.unsqueeze(-1).expand(-1, -1, C))
        unm = torch.gather(src, dim=1, index=unm_sorted.unsqueeze(-1).expand(-1, -1, C))

        dst_updated = dst.clone()
        dst_idx = node_idx.gather(dim=1, index=src_idx)  # (B, r) - indices into dst (0..Tb-1)
        dst_updated = dst_updated.scatter_add(1, dst_idx.unsqueeze(-1).expand(-1, -1, C), src_sel)

        merged = torch.cat([unm, dst_updated], dim=1)  # (B, M_new, C) where M_new = (Ta - r) + Tb
        source_map = torch.full((B, T_prev), -1, dtype=torch.long, device=device)

        if unm_sorted.numel() > 0:
            if pos_a.dim() == 1:
                pos_a_expand = pos_a.unsqueeze(0).expand(B, -1)  # (B, Ta)
            else:
                pos_a_expand = pos_a  # (B, Ta)
            unm_prev_pos = torch.gather(pos_a_expand, 1, unm_sorted)  # (B, n_unm) absolute prev positions
            n_unm = unm_sorted.shape[1]
            new_unm_idx = torch.arange(n_unm, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
            source_map.scatter_(1, unm_prev_pos, new_unm_idx)

        n_unm = unm_sorted.shape[1] if unm_sorted.numel() > 0 else 0
        if pos_b is not None and pos_b.numel() > 0:
            if pos_b.dim() == 1:
                pos_b_expand = pos_b.unsqueeze(0).expand(B, -1)  # (B, Tb)
            else:
                pos_b_expand = pos_b
            b_new_indices = (torch.arange(Tb, dtype=torch.long, device=device).unsqueeze(0) + n_unm).expand(B, -1)
            source_map.scatter_(1, pos_b_expand.long().to(device), b_new_indices)

        if src_idx.numel() > 0:
            # src_idx are indices into a (0..Ta-1) -> map to prev positions via pos_a_expand
            if pos_a.dim() == 1:
                pos_a_expand = pos_a.unsqueeze(0).expand(B, -1)
            else:
                pos_a_expand = pos_a
            src_prev_pos = torch.gather(pos_a_expand, 1, src_idx)  # (B, r) absolute prev positions
            dst_new = (dst_idx + n_unm).long()
            source_map.scatter_(1, src_prev_pos.long().to(device), dst_new.long().to(device))


        return merged, source_map

    