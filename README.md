# LLM Post-Training

Implementations and experiments for understanding **LLM post-training from first principles**.

The goal of this repository is to build a deeper understanding of how modern language models are fine-tuned and optimized, rather than relying only on high-level frameworks.

## Topics

* Supervised Fine-Tuning (SFT)
* Parameter-Efficient Fine-Tuning (LoRA)
* Preference Optimization
* Reinforcement Learning for LLMs

---

## Learning Path

```text
Supervised Fine-Tuning
        ↓
      LoRA
        ↓
Preference Optimization
        ↓
Reinforcement Learning
```

---

## Implementations

### Supervised Fine-Tuning

* Chat template encoding
* Instruction fine-tuning collation
* Instruction loss masking
* Masked cross-entropy
* Masked token log-probability

### LoRA

* LoRA parameter counting
* LoRA forward pass
* Recursive LoRA layer injection
* LoRA weight merging and unmerging
* QLoRA forward pass

### Preference Optimization

Coming soon.

### Reinforcement Learning

Coming soon.

---

## Experiments

### TinyStories LoRA Fine-Tuning

Fine-tuned **DistilGPT2** on TinyStories using LoRA.

* Base model: `distilgpt2`
* LoRA rank: 8
* Target module: `c_attn`
* Learning rate: `5e-4`
* Epochs: 3
* Batch size: 4
* Hardware: NVIDIA T4
* **Validation loss: 3.0500**

[Experiment details](./experiments/tinystories_lora)

---

## Repository Structure

```text
llm-post-training/
│
├── README.md
│
├── sft/
│   ├── chat_template.py
│   ├── custom_collate.py
│   ├── mask_instruction_loss.py
│   ├── masked_cross_entropy.py
│   └── masked_logprob.py
│
├── lora/
│   ├── parameter_count.py
│   ├── forward.py
│   ├── replace_linear.py
│   ├── merge.py
│   └── qlora_forward.py
│
├── experiments/
│   └── tinystories_lora/
│       ├── README.md
│       └── train.py
│
├── preference_optimization/
├── reinforcement_learning/
│
├── tests/
│   ├── test_sft.py
│   └── test_lora.py
│
└── requirements.txt
```

---

## Philosophy

The implementations are intentionally kept small and explicit.

The focus is on understanding:

* the mathematics behind post-training algorithms,
* how training objectives are constructed,
* how token-level losses and probabilities are computed,
* how parameter-efficient fine-tuning works,
* and how these components behave in actual model-training experiments.

The repository will gradually progress from small first-principles implementations toward larger end-to-end experiments involving preference optimization and reinforcement learning.
