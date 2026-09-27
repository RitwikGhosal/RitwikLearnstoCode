def flop_estimator(matmuls: list[list[int]], attention_flops: int = 0) -> dict:
    """
    Returns integer forward_flops, backward_flops, and total_flops in a dictionary.
    """
    forward_flops = 0
    for entry in matmuls:
        (B, D, K) = entry
        forward_flops += 2 * int(B) * int(D) * int(K)
    forward_flops += int(attention_flops)    
    backward_flops = 2 * forward_flops
    total_flops = forward_flops + backward_flops
    return {'forward_flops': forward_flops, 'backward_flops': backward_flops, 'total_flops': total_flops}
        
