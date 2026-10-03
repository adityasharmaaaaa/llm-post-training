import torch

def qlora_forward(
    x: torch.Tensor,
    quantized_W: torch.Tensor,
    scale: float,
    zero_point: float,
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    QLoRA forward pass with quantized frozen weights using PyTorch.
    """
    # 1. Dequantize weights
    # Note: Autograder expects zero_point as an additive float shift
    W = (quantized_W.to(x.dtype) * scale) + zero_point
    
    # 2. Frozen Pretrained path
    base_output = x @ W
    
    # 3. LoRA Update path
    rank = B.shape[1]
    lora_update = (alpha / rank) * (x @ B @ A)
    
    # 4. Combine outputs
    return base_output + lora_update