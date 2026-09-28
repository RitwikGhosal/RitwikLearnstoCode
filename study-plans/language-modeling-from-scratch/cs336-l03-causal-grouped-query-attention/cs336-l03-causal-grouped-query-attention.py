import torch
import math

def causal_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, q_h, S, C = q.shape
    kv_h = k.shape[1]
    group_size = q_h // kv_h

    q_group = q.float().reshape(B, kv_h, group_size, S, C)
    k_reshaped = k.float().reshape(B, kv_h, 1, S, C)
    v_reshaped = v.float().reshape(B, kv_h, 1, S, C)

    wei = (q_group @ k_reshaped.transpose(-2, -1)) / math.sqrt(C)

    mask = torch.tril(
        torch.ones(S, S, device=q.device, dtype=torch.bool)
    )

    scores = wei.masked_fill(~mask, float("-inf"))
    prob = torch.softmax(scores, dim=-1)

    grouped_out = prob @ v_reshaped

    out = grouped_out.reshape(B, q_h, S, C).to(q.dtype)
    return out