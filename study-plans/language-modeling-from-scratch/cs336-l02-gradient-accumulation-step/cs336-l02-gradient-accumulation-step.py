import torch

def gradient_accumulation_step(
    param: torch.Tensor, microbatch_inputs: list[torch.Tensor],
    microbatch_targets: list[torch.Tensor], lr: float,
) -> dict:
    """
    Returns new_param and full_grad tensors in a dictionary.
    """
    work_param = param.detach().clone().requires_grad_(True)
    total_examples = sum(inputs.shape[0] for inputs in microbatch_inputs)

    for inputs, target in zip(microbatch_inputs, microbatch_targets):
        predictions = inputs @ work_param
        loss = (predictions - target).square().sum() / total_examples
        loss.backward()

    full_grad = work_param.grad.detach().clone()
    new_param = (work_param - lr * full_grad).detach()
    return {"new_param": new_param, "full_grad": full_grad}