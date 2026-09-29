def gpu_occupancy(
    threads_per_block: int, registers_per_thread: int, shared_mem_per_block: int,
    max_threads_per_sm: int, max_warps_per_sm: int, max_blocks_per_sm: int,
    max_registers_per_sm: int, max_shared_mem_per_sm: int, warp_size: int = 32,
) -> dict:
    """
    Returns a dict: blocks_per_sm (int), resident_warps (int), occupancy (float).
    """
    warps_per_block = (threads_per_block + warp_size - 1) // warp_size
    effective_threads = warps_per_block * warp_size
    blocks_from_threads = max_threads_per_sm // effective_threads
    blocks_from_warps = max_warps_per_sm // warps_per_block
    blocks_from_registers = max_blocks_per_sm
    if registers_per_thread:
        blocks_from_registers = max_registers_per_sm // (
            registers_per_thread * effective_threads
        )
        
    blocks_from_shared_memory = max_blocks_per_sm
    if shared_mem_per_block:
        blocks_from_shared_memory = max_shared_mem_per_sm // shared_mem_per_block

    blocks_per_sm = min(
        max_blocks_per_sm,
        blocks_from_threads,
        blocks_from_warps,
        blocks_from_registers,
        blocks_from_shared_memory,
    )
    resident_warps = blocks_per_sm * warps_per_block
    return {
        "blocks_per_sm": blocks_per_sm,
        "resident_warps": resident_warps,
        "occupancy": resident_warps / max_warps_per_sm,
    }
