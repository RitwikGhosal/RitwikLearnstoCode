import torch

def rmsnorm(x: torch.Tensor, g: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Returns the RMS-normalized tensor with the same shape and dtype as x.
    """
    x_f = x.float()
    scale_f = g.float()
    mean_square = x_f.square().mean(dim = -1, keepdim = True)
    normalized = x_f * torch.rsqrt(mean_square + epsilon)
    return (normalized * scale_f).to(x.dtype)
