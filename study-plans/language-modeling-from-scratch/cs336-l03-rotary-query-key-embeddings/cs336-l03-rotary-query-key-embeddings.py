import torch

def rotary_embed(
    q: torch.Tensor, k: torch.Tensor,
    positions: torch.Tensor, inv_freq: torch.Tensor,
) -> dict:
    """
    Returns q_rotated and k_rotated tensors in a dictionary.
    """
    if positions.ndim == 1:
        position_values = positions.float().reshape(1, 1, -1, 1)
    else:
        position_values = positions.float().reshape(positions.shape[0], 1, positions.shape[1], 1)

    angles = position_values * inv_freq.float().reshape(1, 1, 1, -1)
    cosine = torch.cos(angles)
    sine = torch.sin(angles)

    def rotate(tensor):
        tensor_float = tensor.float()
        even = tensor_float[..., 0::2]
        odd = tensor_float[..., 1::2]
        rotated_even = even * cosine - odd * sine
        rotated_odd = even * sine + odd * cosine
        return torch.stack((rotated_even, rotated_odd), dim = -1).flatten(-2).to(tensor.dtype)

    return {"q_rotated": rotate(q), "k_rotated": rotate(k)}