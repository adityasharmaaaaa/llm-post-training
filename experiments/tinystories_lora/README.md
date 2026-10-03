# TinyStories LoRA Fine-Tuning

Fine-tuning **DistilGPT2** on the **TinyStories** dataset using Low-Rank Adaptation (LoRA).

The experiment uses a held-out validation set and optimizes mean cross-entropy loss.

## Setup

* **Base model:** `distilgpt2`
* **Dataset:** TinyStories
* **Hardware:** NVIDIA T4 (16 GB VRAM)
* **Maximum sequence length:** 256 tokens
* **Method:** LoRA
* **LoRA rank:** 8
* **Target module:** `c_attn`
* **Learning rate:** `5e-4`
* **Optimizer:** AdamW
* **Training epochs:** 3
* **Batch size:** 4

## Results

| Configuration        | Validation Loss |
| -------------------- | --------------: |
| Untrained DistilGPT2 |            ~4.5 |
| LoRA fine-tuning     |      **3.0500** |

The fine-tuned model achieved a validation loss of **3.0500**, improving substantially over the untrained baseline.

## Implementation

The experiment:

1. Tokenizes the TinyStories training and validation samples.
2. Creates causal language-modeling labels from the input tokens.
3. Injects LoRA adapters into the model's attention projections.
4. Freezes the pretrained model weights.
5. Trains only the LoRA parameters.
6. Evaluates the resulting model on the held-out validation set.

The training implementation is in [`train.py`](./train.py).

## Why LoRA?

LoRA adapts a pretrained model by learning a low-rank update to selected weight matrices while keeping the original model weights frozen.

For a target weight matrix:

$$
W' = W_0 + \frac{\alpha}{r}BA
$$

where `W₀` is the frozen pretrained weight and `A` and `B` are the trainable low-rank matrices.

This makes LoRA substantially more parameter-efficient than full fine-tuning while still allowing the model to adapt to the target dataset.

## Notes

This experiment is intentionally small and constrained by the available T4 compute and execution time. The goal is to understand and empirically test the mechanics of LLM fine-tuning rather than maximize benchmark performance.
