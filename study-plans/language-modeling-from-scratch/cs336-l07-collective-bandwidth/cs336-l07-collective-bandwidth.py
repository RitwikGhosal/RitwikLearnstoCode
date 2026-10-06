def collective_bandwidth(payload_bytes: int, world_size: int, duration_seconds: float, collective: str) -> dict:
    """
    Returns a dict of floats: algorithm_bytes and bandwidth_bytes_per_second.
    """
    c = 2 if collective == "all_reduce" else 1
    algo_bytes = float((c * payload_bytes * (world_size - 1) / world_size))
    return {
        "algorithm_bytes" : algo_bytes,
        "bandwidth_bytes_per_second" : float(algo_bytes / duration_seconds)
    }
