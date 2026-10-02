def lora_param_count(num_layers: int, d_model: int, num_target_matrices: int, rank: int) -> dict:
   lora_params=d_model*rank*2*num_target_matrices*num_layers
   full_params=d_model*d_model*num_target_matrices*num_layers
   return {"lora_params":lora_params,"full_params":full_params,"compression_ratio":full_params/lora_params}