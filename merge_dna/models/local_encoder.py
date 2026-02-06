import torch
import torch.nn as nn
import torch.nn.functional as F
from .token_merge import TokenMerge
from typing import Tuple, List
    

class LocalAttention(nn.Module):
    """
    Local-attention implemented by folding windows into the batch dimension
    and applying nn.MultiheadAttention per window (batch_first=True).
    """
    def __init__(self, d_model: int, nhead: int, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.to_k = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model))
        self.ln2 = nn.LayerNorm(d_model)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Args:
            x: (B, L, D)
        Returns:
            x_attn_unfolded: (B, L_trunc, D)  -- attended, unfolded (with padding kept)
            x_windows_attn: (B * num_windows, W, D) -- folded windows after attn
            num_windows: int
        """
        B, L, D = x.shape
        W = self.window_size
        if L % W != 0:
            raise ValueError("LocalAttention expects L divisible by window_size (pad before calling)")

        num_windows = L // W
        # fold windows into batch
        x_windows = x.view(B, num_windows, W, D).reshape(B * num_windows, W, D)  # folded

        # compute keys if needed (we return them folded, but here we only produce to_k for TokenMerge if used)
        # k_windows = self.to_k(x_windows)

        # apply MHA per folded window
        attn_out, _ = self.mha(x_windows, x_windows, x_windows, need_weights=False)
        x_windows_attn = x_windows + attn_out
        x_windows_attn = self.ln(x_windows_attn)
        ff = self.ff(x_windows_attn)
        x_windows_attn = self.ln2(x_windows_attn + ff)

        # unfold back to (B, L_trunc, D)
        x_attn_unfolded = x_windows_attn.view(B, num_windows * W, D)

        return x_attn_unfolded, x_windows_attn, num_windows



class LocalBlock(nn.Module):
    """
    Single local encoder layer: local-attention -> TokenMerge on each window.
    Returns:
        merged: (B, L_next, D)
        source_index: (B, L_trunc) mapping original positions to merged indices
    """
    def __init__(self, d_model: int, nhead: int, window_size: int, r: int, token_merge: TokenMerge):
        """
        token_merge: an instance of TokenMerge that expects (x_window, k_window) and returns (merged_window, source_index_window)
        r: number of merges per window (reduces window length by r)
        """
        super().__init__()
        assert window_size % 2 == 0, "window_size must be even for bipartite merging"
        self.attn = LocalAttention(d_model, nhead, window_size)
        self.window_size = window_size
        self.r = r
        self.token_merge = token_merge
        # precompute M_w (output length per window) = W - r (for bipartite pattern)
        self.M_w = window_size - r


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, D) token indices
        B, L, D = x.shape
        W = self.window_size
        device = x.device

        pad_len = (W - (L % W)) % W
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len)) 
        L_trunc = x.shape[1]
        num_windows = L_trunc // W

        # 1) local attention per window
        x_attn_unfolded, x_windows_attn, num_windows = self.attn.forward(x)  # (B, L_trunc, D)

        # 2) call TokenMerge in batch: it must accept shape (B*num_windows, W, D) and return merged and source_index (local)
        merged_windows, source_index_windows = self.token_merge.forward(x_windows_attn, x_windows_attn)
        # merged_windows: (B * num_windows, M_w, D)
        # source_index_windows: (B * num_windows, W) long, mapping local window pos -> merged index (0..M_w-1)


        # Defensive asserts to catch shape mismatch:
        expected_folded_batch = B * num_windows
        if merged_windows.dim() != 3:
            raise RuntimeError(f"token_merge returned tensor with dim {merged_windows.dim()}, expected 3 (batch, M_w, D).")
        if source_index_windows.dim() != 2:
            raise RuntimeError(f"token_merge returned source_index with dim {source_index_windows.dim()}, expected 2 (batch, W).")

        # merged_windows should have batch == expected_folded_batch
        if merged_windows.shape[0] != expected_folded_batch:
            raise RuntimeError(
                f"token_merge returned batch {merged_windows.shape[0]} but expected {expected_folded_batch} "
                f"(B * num_windows). This means token_merge is not processing folded windows correctly."
            )
        # check second dim equals M_w
        if merged_windows.shape[1] != self.M_w:
            raise RuntimeError(
                f"token_merge returned merged length {merged_windows.shape[1]} but expected M_w={self.M_w}. "
                "Check TokenMerge configuration (r vs window_size)."
            )
        # check vector dim
        if merged_windows.shape[2] != D:
            raise RuntimeError(f"token_merge returned D={merged_windows.shape[2]} but expected D={D}.")


        # 3) reshape merged windows back to (B, num_windows * M_w, D)
        merged = merged_windows.view(B, num_windows * self.M_w, D)

        # 4) build global source_index mapping original positions -> merged index
        # source_index_windows currently maps window-local pos -> merged index within that window.
        # We must convert window-local indices to global merged indices in concatenated merged sequence.
        # For window w, global_offset = w * M_w. We add those offsets and then place owner indices into the global owner map.

        source_index_windows = source_index_windows.view(B, num_windows, W)  # (B, num_windows, W)
        # offsets per window with shape (1, num_windows, 1)
        offsets_per_window = (torch.arange(num_windows, device=device, dtype=torch.long) * self.M_w).view(1, num_windows, 1)
        source_index_global_windows = source_index_windows + offsets_per_window  # (B, num_windows, W)

        # Now we need to map each original absolute position to its merged index.
        # compute absolute original positions for each window-local position
        # For window w, local positions 0..W-1 correspond to absolute positions base = w*W + local_pos
        # Let's compute a tensor abs_pos_windows of shape (num_windows, W)
        abs_pos = (torch.arange(W, device=device).unsqueeze(0) + (torch.arange(num_windows, device=device).unsqueeze(1) * W))  # (num_windows, W)
        # expand to (B, num_windows, W)
        abs_pos = abs_pos.unsqueeze(0).expand(B, -1, -1)

        # Now flatten to (B, num_windows * W) in the same order windows were folded
        source_index_per_position = source_index_global_windows.reshape(B, num_windows * W)  # merged index per position
        abs_pos_flat = abs_pos.reshape(B, num_windows * W)  # absolute positions in padded sequence

        # abs_pos_flat[b,p] is the absolute location (0..L_trunc-1) corresponding to position p in the folded order.
        # But abs_pos_flat is exactly [0..L_trunc-1] broadcast per batch; we can invert mapping easy:
        # Build source_index_by_absolute_position of shape (B, L_trunc), init -1
        source_index_by_abs = torch.full((B, L_trunc), -1, dtype=torch.long, device=device)
        # scatter owner indices into source_index_by_abs at positions abs_pos_flat using source_index_per_position as src
        source_index_by_abs.scatter_(dim=1, index=abs_pos_flat, src=source_index_per_position)

        # Finally, trim padding if any
        if pad_len > 0:
            source_index_by_abs = source_index_by_abs[:, :L]   # L is original length before padding
            # Also trim merged accordingly: merged has shape (B, num_windows*M_w, D). We do not trim merged here because merged corresponds to padded windows. 
            # But if pad created extra windows with only padded tokens, you might need to drop the corresponding merged tokens afterwards.
            # A quick safe approach: compute number of valid windows = ceil(original_L / W) and keep first valid_windows * M_w merged tokens.
            valid_windows = (L + W - 1) // W
            merged = merged[:, : valid_windows * self.M_w, :]

        return merged, source_index_by_abs


class LocalEncoder(nn.Module):
    """
    Stack multiple LocalBlock layers. Each layer takes the compressed sequence from the previous
    layer and runs local attention + merging with its own window_size and r.
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, layer_configs: List[dict], token_merge_factory):
        """
        layer_configs: list of dicts, each with keys: {'window_size': int, 'r': int}
        token_merge_factory: callable r -> TokenMerge instance constructor or a module to be cloned per layer
        """
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model) 
        self.layers = nn.ModuleList()
        for cfg in layer_configs:
            W = cfg['window_size']
            r = cfg['r']
            if r < 0 or r > W // 2:
                raise Exception(f'Require 0 leq r leq W // 2, received {r, W}')
            token_merge = token_merge_factory(r)
            block = LocalBlock(d_model=d_model, nhead=nhead, window_size=W, r=r, token_merge=token_merge)
            self.layers.append(block)


    def forward(self, x: torch.Tensor):
        """
        x: (B, L, D)
        Returns:
          x: compressed representation after last local layer
          source_index_list: list of source_index maps per layer (if you want to keep them), else only last
        """
        x = self.emb(x)
        source_maps = []
        for layer in self.layers:
            merged, source_index = layer(x)
            source_maps.append(source_index)
            x = merged  # next layer runs on compressed sequence
        return x, source_maps
