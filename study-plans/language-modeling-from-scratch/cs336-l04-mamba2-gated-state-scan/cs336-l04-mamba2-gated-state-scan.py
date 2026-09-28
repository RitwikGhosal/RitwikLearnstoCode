import torch

def mamba2_gated_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: torch.Tensor) -> dict:
    """
    Returns a dict of tensors: outputs and final_state.
    """
    q_float = q.float()
    k_float = k.float()
    v_float = v.float()
    gamma_float = gamma.float()

    state = torch.zeros(q.shape[0], q.shape[2], v.shape[2], dtype = torch.float32, device= q.device)
    output_steps = []

    for position in range(q.shape[1]):
        decay = gamma_float[:, position].reshape(-1, 1, 1)
        write = k_float[:, position].unsqueeze(-1) * v_float[:, position].unsqueeze(-2)
        state = decay * state + write
        output_steps.append(
            torch.bmm(q_float[:, position].unsqueeze(1), state).squeeze(1)
        )

    return {
        "outputs": torch.stack(output_steps, dim=1).to(q.dtype),
        "final_state": state.to(q.dtype),
    }
