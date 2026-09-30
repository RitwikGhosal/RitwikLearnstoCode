import torch

def recursive_parameter_collection(weights: list[torch.Tensor], biases: list[torch.Tensor]) -> tuple:
    """
    Returns a list of scalar parameter views and its integer count.
    """
    op = []
    for w, b in zip(weights, biases):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                op.append(w[i, j])
            op.append(b[i])    
    return (op, len(op))
