import torch
import math

def attention_scores(q: torch.Tensor, k: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Returns scores of shape (batch, heads, query_length, key_length).
    """
    B, q_T, C = q.shape
    k_T = k.shape[1]
    head_size = C // num_heads

    q_ = q.reshape(B, q_T, num_heads, head_size).transpose(1, 2)
    k_ = k.reshape(B, k_T, num_heads, head_size).transpose(1, 2)
    wei = (q_ @ k_.transpose(-1, -2)) / math.sqrt(head_size)
    return wei