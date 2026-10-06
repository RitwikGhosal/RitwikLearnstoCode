import numpy as np

def temperature_data_mixture(token_counts: list, alpha: int | float, training_budget: int) -> dict:
    """
    Returns a dict: probabilities, expected_tokens, expected_epochs (float64 arrays).
    """
    counts = np.asarray(token_counts, dtype=np.float64)
    positive = counts > 0
    probabilities = np.zeros(counts.shape, dtype=np.float64)
    if np.any(positive):
        log_counts = np.log(counts[positive])
        with np.errstate(over="ignore"):
            log_weights = float(alpha) * (log_counts - np.max(log_counts))
        weights = np.exp(log_weights)
        probabilities[positive] = weights / weights.sum()
    expected_tokens = probabilities * float(training_budget)
    expected_epochs = np.zeros(counts.shape, dtype=np.float64)
    expected_epochs[positive] = expected_tokens[positive] / counts[positive]
    return {
        "probabilities": probabilities,
        "expected_tokens": expected_tokens,
        "expected_epochs": expected_epochs,
    }