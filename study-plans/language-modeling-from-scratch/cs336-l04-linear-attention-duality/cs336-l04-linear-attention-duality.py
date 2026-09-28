import torch

def linear_attention_duality(q, k, v):
    q_float = q.float()
    k_float = k.float()
    v_float = v.float()

    writes = k_float.unsqueeze(-1) * v_float.unsqueeze(-2)
    parallel_states = torch.cumsum(writes, dim=1)
    parallel_output = torch.einsum("bsd,bsdv->bsv", q_float, parallel_states)

    state = torch.zeros(
        q.shape[0], q.shape[2], v.shape[2], dtype=torch.float32, device=q.device
    )
    recurrent_steps = []
    for position in range(q.shape[1]):
        state = state + writes[:, position]
        recurrent_steps.append(
            torch.einsum("bd,bdv->bv", q_float[:, position], state)
        )
    recurrent_output = torch.stack(recurrent_steps, dim=1)

    return {
        "parallel_output": parallel_output.to(q.dtype),
        "recurrent_output": recurrent_output.to(q.dtype),
        "final_state": state.to(q.dtype),
    }
