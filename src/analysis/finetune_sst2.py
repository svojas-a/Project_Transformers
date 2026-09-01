"""
M4 - Prerequisite: Fine-tune DistilBERT on SST-2
==================================================

Run this FIRST, before forward_causal_intervention.py, if no fine-tuned
checkpoint exists yet. This trains a plain DistilBERT classifier on SST-2
and saves it to Drive so forward_causal_intervention.py has something real
to point at.

Run this in Colab (needs GPU: Runtime -> Change runtime type -> GPU).
"""

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
SAVE_PATH = "/content/drive/MyDrive/Project_Transformers/checkpoints/sst2_finetuned"
NUM_EPOCHS = 2          # SST-2 is small + DistilBERT converges fast; 2-3 is plenty
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

# -----------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------
print("Loading SST-2 from nyu-mll/glue...")
dataset = load_dataset("nyu-mll/glue", "sst2")
print(dataset)
# dataset has: train, validation, test
# columns: sentence, label (0=negative, 1=positive), idx

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    return tokenizer(examples["sentence"], truncation=True)

print("Tokenizing...")
tokenized = dataset.map(tokenize_fn, batched=True)
tokenized = tokenized.remove_columns(["sentence", "idx"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -----------------------------------------------------------------------
# 2. LOAD MODEL (2 labels: negative / positive)
# -----------------------------------------------------------------------
print(f"Loading {MODEL_NAME} for sequence classification...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

# -----------------------------------------------------------------------
# 3. METRICS
# -----------------------------------------------------------------------
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# -----------------------------------------------------------------------
# 4. TRAIN
# -----------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="/content/sst2_training_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nStarting training...")
trainer.train()

print("\nFinal evaluation on validation set:")
eval_results = trainer.evaluate()
print(eval_results)

# -----------------------------------------------------------------------
# 5. SAVE CHECKPOINT
# -----------------------------------------------------------------------
print(f"\nSaving fine-tuned model to: {SAVE_PATH}")
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print("Done. Point MODEL_CHECKPOINT_PATH in forward_causal_intervention.py "
      f"at:\n  {SAVE_PATH}")