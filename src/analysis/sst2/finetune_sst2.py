"""
SST-2 fine-tuning prerequisite.

Run from the sst2 directory:
    python finetune_sst2.py

The checkpoint is saved under:
    sst2/checkpoints/sst2_finetuned/
"""

from pathlib import Path
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

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "distilbert-base-uncased"
SAVE_PATH = ROOT / "checkpoints" / "sst2_finetuned"

NUM_EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
SEED = 42

print("Loading SST-2...")
dataset = load_dataset("nyu-mll/glue", "sst2")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_fn(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
    )


tokenized = dataset.map(tokenize_fn, batched=True)
tokenized = tokenized.remove_columns(["sentence", "idx"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(
        predictions=predictions,
        references=labels,
    )


training_args = TrainingArguments(
    output_dir=str(ROOT / "training_output"),
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
    seed=SEED,
    data_seed=SEED,
    report_to="none",
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

print("\nTraining...")
trainer.train()

print("\nValidation evaluation:")
results = trainer.evaluate()
print(results)

SAVE_PATH.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(SAVE_PATH))
tokenizer.save_pretrained(str(SAVE_PATH))

print(f"\nCheckpoint saved to:\n{SAVE_PATH}")
