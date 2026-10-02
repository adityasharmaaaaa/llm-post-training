import numpy as np

def masked_avg_logprob(logits, labels, selection_mask):
    mask_sum = np.sum(selection_mask)
    if mask_sum == 0:
        return 0.0

    max_logits = np.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - max_logits
    log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=-1, keepdims=True))
    log_probs = shifted_logits - log_sum_exp  # shape: (B, T, V)

    gathered_log_probs = np.take_along_axis(
        log_probs, np.expand_dims(labels, axis=-1), axis=-1
    ).squeeze(
        axis=-1
    )  

    masked_log_probs = gathered_log_probs * selection_mask
    result = np.sum(masked_log_probs) / mask_sum

    return float(result)
