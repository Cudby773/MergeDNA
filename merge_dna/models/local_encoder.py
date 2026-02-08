import torch
import torch.nn as nn
import torch.nn.functional as F
from .token_merge import TokenMerge
from .local_attention import LocalAttention
from typing import Tuple, List
    

class LocalMergeBlock(nn.Module):
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
        
        
    def build_source_index(self, prev_to_new_windows: torch.Tensor, B: int, W: int, num_windows: int, L_orig: int, device):
        prev_to_new_windows = prev_to_new_windows.view(B, num_windows, W)  # (B, num_windows, W)
        offsets_per_window = (torch.arange(num_windows, device=device, dtype=torch.long) * self.M_w).view(1, num_windows, 1)
        source_index_global_windows = prev_to_new_windows + offsets_per_window  # (B, num_windows, W)
        
        abs_pos = (torch.arange(W, device=device).unsqueeze(0) + (torch.arange(num_windows, device=device).unsqueeze(1) * W))  # (num_windows, W)
        abs_pos = abs_pos.unsqueeze(0).expand(B, -1, -1)
        source_index_per_position = source_index_global_windows.reshape(B, num_windows * W)  # merged index per position
        abs_pos_flat = abs_pos.reshape(B, num_windows * W)  # absolute positions in padded sequence
        source_index_by_abs = torch.full((B, L_orig), -1, dtype=torch.long, device=device)
        # scatter owner indices into source_index_by_abs at positions abs_pos_flat using source_index_per_position as src
        source_index_by_abs.scatter_(dim=1, index=abs_pos_flat, src=source_index_per_position)        
        return source_index_by_abs


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, L, D) token indices
        B, L, D = x.shape
        W = self.window_size
        device = x.device

        pad_len = (W - (L % W)) % W
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len)) 
        L_trunc = x.shape[1]
        num_windows = L_trunc // W

        _, x_windows_attn, num_windows = self.attn.forward(x)  # (B, L_trunc, D)

        merged_windows, prev_to_new_windows, merge_scores_windows = self.token_merge.forward(x_windows_attn, x_windows_attn)
        # merged_windows: (B * num_windows, M_w, D)
        # prev_to_new_windows: (B * num_windows, W) long, mapping local window pos -> merged index (0..M_w-1)
        # merge_scores: (B*num_windows, D)        
        
        merged = merged_windows.view(B, num_windows * self.M_w, D)
        merge_scores = merge_scores_windows.view(B, num_windows * self.M_w)
        source_index_by_abs = self.build_source_index(prev_to_new_windows, B, W, num_windows, L_trunc, device)


        # Finally, trim padding if any
        if pad_len > 0:
            source_index_by_abs = source_index_by_abs[:, :L]   # L is original length before padding
            valid_windows = (L + W - 1) // W
            merged = merged[:, :valid_windows * self.M_w, :]
            merge_scores = merge_scores[:, :valid_windows * self.M_w]

        return merged, source_index_by_abs, merge_scores


class LocalEncoder(nn.Module):
    """
    Stack multiple LocalBlock layers. Each layer takes the compressed sequence from the previous
    layer and runs local attention + merging with its own window_size and r.
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, layer_configs: List[dict]):
        """
        layer_configs: list of dicts, each with keys: {'window_size': int, 'r': int}
        token_merge_factory: callable r -> TokenMerge instance constructor or a module to be cloned per layer
        """
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model) 
        self.layers = nn.ModuleList()
        for cfg in layer_configs:
            W = cfg['window_size']
            r = cfg['r']
            if r < 0 or r > W // 2:
                raise Exception(f'Require 0 leq r leq W // 2, received {r, W}')
            token_merge = TokenMerge(r)
            block = LocalMergeBlock(d_model=d_model, nhead=nhead, window_size=W, r=r, token_merge=token_merge)
            self.layers.append(block)


    def forward(self, x: torch.Tensor):
        """
        x: (B, L, D)
        Returns:
            x: compressed representation after last local layer
            source_maps: list of source_index maps per layer
            merge_scores_list: per layer scores
        """
        x = self.emb(x)
        source_maps = []
        merge_scores_list = []
        B, T_orig, _ = x.shape
        for layer in self.layers:
            # each layer.forward now returns (merged, source_index_by_abs, merge_scores)
            merged, source_index, merge_scores = layer(x)
            source_maps.append(source_index)
            merge_scores_list.append(merge_scores)  # (B, M_layer)
            x = merged  # next layer runs on compressed sequence
        return x, source_maps, merge_scores_list
