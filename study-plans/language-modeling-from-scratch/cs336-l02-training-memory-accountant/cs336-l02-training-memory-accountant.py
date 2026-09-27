def memory_accountant(
    param_shapes: list[list[int]], param_bytes_per_element: int,
    grad_bytes_per_element: int, activation_shapes: list[list[int]],
    activation_bytes_per_element: int, optimizer: str,
    optimizer_bytes_per_element: int,
) -> dict:
    """
    Returns integer byte counts for parameters, gradients, activations, optimizer_state, and total.
    """
    def elemenet_count(shapes):
        total = 0
        for shape in shapes:
            count = 1
            for dim in shape:
                count *= dim 
            total += count
        return total

    param_elemenets = elemenet_count(param_shapes)
    activation_elements = elemenet_count(activation_shapes)
    parameters = param_elemenets * param_bytes_per_element
    gradients = param_elemenets * grad_bytes_per_element
    activations = activation_elements * activation_bytes_per_element
    state_multiplier = {'sgd':0, 'adagrad': 1, 'adam': 2}[optimizer]
    optimizer_state = state_multiplier * param_elemenets * optimizer_bytes_per_element
    total = parameters + gradients + activations + optimizer_state
    return {'parameters': parameters, 'gradients': gradients, 'activations': activations, 'optimizer_state': optimizer_state, 'total': total}
