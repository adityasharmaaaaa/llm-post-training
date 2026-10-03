import torch
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

def train(model, tokenizer, train_texts, val_texts):
    # 1. Data Preparation
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )
    train_encodings['labels'] = train_encodings['input_ids'].clone()

    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )
    val_encodings['labels'] = val_encodings['input_ids'].clone()

    train_dataset = TextDataset(train_encodings)
    val_dataset = TextDataset(val_encodings)

    # 2. Model Setup (LoRA)
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        target_modules=["c_attn"]
    )
    model = get_peft_model(model, lora_config)

    # 3. Training
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=5e-4,
        num_train_epochs=3,
        per_device_train_batch_size=4
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    trainer.train()
    
    return model