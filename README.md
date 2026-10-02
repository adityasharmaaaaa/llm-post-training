# LLM Post-Training

Implementations and experiments for understanding **LLM post-training**.

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

### Preference Optimization

Coming soon.

### Reinforcement Learning

Coming soon.

---

## Repository Structure

```text
llm-post-training/
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
│   └── forward.py
│
├── preference_optimization/
│
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

* the mathematics behind the algorithms,
* how the training objectives are constructed,
* how token-level losses and probabilities are computed,
* and how the individual components fit together into modern LLM post-training pipelines.

As the repository grows, larger experiments and end-to-end implementations will be added.
