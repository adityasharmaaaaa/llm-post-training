def custom_collate(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None):
    max_len=max(len(item)+1 for item in batch)
    ans={}
    ans['inputs']=[]
    ans['targets']=[]
    for item in batch:
        padded=item.copy()
        padded.append(pad_token_id)
        while len(padded) < max_len: 
            padded.append(pad_token_id)
        inputs=padded[:-1]
        targets=padded[1:]
        first=True
        for i in range(len(targets)):
            if targets[i]==pad_token_id:
                if first==True:
                    first=False
                else:
                    targets[i]=ignore_index
        if allowed_max_length is not None:
            inputs=inputs[:allowed_max_length]
            targets=targets[:allowed_max_length]
        ans['inputs'].append(inputs)
        ans['targets'].append(targets)
    return ans
