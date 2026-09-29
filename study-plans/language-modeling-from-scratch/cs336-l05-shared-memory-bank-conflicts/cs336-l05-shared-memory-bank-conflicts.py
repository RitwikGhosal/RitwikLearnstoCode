import numpy as np

def bank_conflict_analysis(byte_addresses: np.ndarray, num_banks: int = 32, bank_width_bytes: int = 4) -> dict:
    """
    Returns a dict of int64 arrays: bank_ids and conflict_degree.
    """
    addresses = np.asarray(byte_addresses, dtype = np.int64)
    bank_ids = (addresses // bank_width_bytes) % num_banks
    conflict_degree = np.empty(addresses.shape, dtype = np.int64)
    for bank in np.unique(bank_ids):
        lane_mask = bank_ids == bank
        conflict_degree[lane_mask] = np.unique(addresses[lane_mask]).size
    return {"bank_ids": bank_ids, "conflict_degree": conflict_degree}
