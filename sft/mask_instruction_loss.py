def mask_instruction_loss(batch, pad_token_id, ignore_index=-100):
    max_len = max(len(item["tokens"]) for item in batch)

    inputs = []
    targets = []

    for item in batch:
        tokens = item["tokens"]
        instruction_length = item["instruction_length"]

        padded = tokens + [pad_token_id] * (max_len - len(tokens))

        input_ids = padded[:-1]
        target_ids = padded[1:]

        for i in range(len(target_ids)):
            if i + 1 < instruction_length:
                target_ids[i] = ignore_index

        first_pad_seen = False

        for i, token_id in enumerate(target_ids):
            if token_id == pad_token_id:
                if not first_pad_seen:
                    first_pad_seen = True
                else:
                    target_ids[i] = ignore_index

        inputs.append(input_ids)
        targets.append(target_ids)

    return {
        "inputs": inputs,
        "targets": targets,
    }