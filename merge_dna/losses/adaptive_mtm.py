import torch


def _token_probs_from_source_map(source_map: torch.LongTensor, eps: float = 1e-12):
    B, L = source_map.shape
    device = source_map.device
    P = torch.zeros((B, L), device=device, dtype=torch.float)

    for b in range(B):
        groups = source_map[b]                    
        group_sizes = torch.bincount(groups)      
        inv_sizes = 1.0 / (group_sizes ** 2 + eps)
        P[b] = inv_sizes[groups]                  

    P = P / P.sum(dim=1, keepdim=True)
    return P


def sample_k_local_tokens_from_source_map(source_map: torch.LongTensor):
    P = _token_probs_from_source_map(source_map)
    return torch.multinomial(P, num_samples=int(source_map.max().item()) + 1, replacement=False)

