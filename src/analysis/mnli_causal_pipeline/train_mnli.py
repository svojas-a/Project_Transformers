"""
Fine-tune DistilBERT for GLUE MNLI.
Run: python train_mnli.py

Saves best checkpoint to ./mnli_checkpoint
"""

import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score


MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./mnli_checkpoint"
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

# GLUE MNLI has train, validation_matched, validation_mismatched, test splits.
# trust_remote_code is needed by some Datasets versions for script-based datasets.
dataset = load_dataset("glue", "mnli", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_examples(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

tokenized = dataset.map(
    tokenize_examples,
    batched=True,
    remove_columns=[
        c for c in dataset["train"].column_names
        if c not in ("label",)
    ],
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    valid = labels != -1
    labels = labels[valid]
    predictions = predictions[valid]
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label={
        0: "ENTAILMENT",
        1: "NEUTRAL",
        2: "CONTRADICTION",
    },
    label2id={
        "ENTAILMENT": 0,
        "NEUTRAL": 1,
        "CONTRADICTION": 2,
    },
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
    logging_steps=200,
    report_to="none",
    seed=SEED,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation_matched"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

print("\nMatched validation metrics:")
print(trainer.evaluate(tokenized["validation_matched"]))

print("\nMismatched validation metrics:")
print(trainer.evaluate(
    eval_dataset=tokenized["validation_mismatched"],
    metric_key_prefix="eval_mismatched",
))

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nBest checkpoint saved to: {OUTPUT_DIR}")
