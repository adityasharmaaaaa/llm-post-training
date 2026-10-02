import numpy as np

def masked_ce_loss(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0

    sub_logits = logits[mask]  # shape: (N_active, vocab_size)
    sub_targets = targets[mask]  # shape: (N_active,)

    max_logits = np.max(sub_logits, axis=-1, keepdims=True)
    shifted_logits = sub_logits - max_logits
    log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=-1, keepdims=True))

    log_probs = shifted_logits - log_sum_exp

    row_indices = np.arange(len(sub_targets))
    target_log_probs = log_probs[row_indices, sub_targets]

    loss = -np.mean(target_log_probs)

    return float(loss)