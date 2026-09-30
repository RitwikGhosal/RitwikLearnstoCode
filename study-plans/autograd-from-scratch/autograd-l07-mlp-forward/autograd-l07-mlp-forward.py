import torch

def mlp_forward(inputs: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor]) -> tuple:
    """
    Returns the final output tensor and a list of layer-output tensors.
    """
    #act = inputs
    layer_outputs = []
    for w, b in zip(weights, biases):
        inputs = torch.tanh(w @ inputs + b)
        layer_outputs.append(inputs)
    return (inputs, layer_outputs)
