"""
Layer-wise causal propagation for fine-tuned DistilBERT on MNLI.

Run:
    python mnli_causal_propagation.py

Uses validation_matched as the primary evaluation split.
The checkpoint is kept frozen; the intervention modifies activations only.

Outputs:
    ./mnli_causal_results/propagation_metrics.csv
    ./mnli_causal_results/propagation_summary.csv
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score


MODEL_PATH = "./mnli_checkpoint"
OUTPUT_DIR = "./mnli_causal_results"

MAX_LENGTH = 128
BATCH_SIZE = 32
BOTTLENECK_DIMS = [256, 128, 64, 32]
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
print("Device:", DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

dataset = load_dataset("glue", "mnli", trust_remote_code=True)

def tokenize_examples(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

# Keep only model inputs and labels; remove raw text and metadata.
validation_dataset = dataset["validation_matched"].map(
    tokenize_examples,
    batched=True,
    remove_columns=[
        c for c in dataset["validation_matched"].column_names
        if c != "label"
    ],
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
)

NUM_LAYERS = len(model.distilbert.transformer.layer)
HIDDEN_SIZE = model.config.dim
print(f"DistilBERT layers={NUM_LAYERS}, hidden_size={HIDDEN_SIZE}")


def create_projection(hidden_size, bottleneck_dim, seed):
    if not 1 <= bottleneck_dim <= hidden_size:
        raise ValueError("Invalid bottleneck dimension")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(
        hidden_size,
        bottleneck_dim,
        generator=generator,
        dtype=torch.float32,
    )
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.to(device=DEVICE, dtype=torch.float32)


def make_intervention_hook(projection):
    def hook(module, inputs, output):
        hidden = output[0]
        compressed = hidden.float() @ projection
        reconstructed = compressed @ projection.T
        reconstructed = reconstructed.to(dtype=hidden.dtype)
        return (reconstructed, *output[1:])
    return hook


@torch.no_grad()
def evaluate(capture_activations=True):
    model.eval()
    all_predictions = []
    all_labels = []

    # activations[0] = embedding output
    # activations[i+1] = output of Transformer block i
    activations = [[] for _ in range(NUM_LAYERS + 1)]
    active_attention_mask = None

    def make_capture_hook(index):
        def capture_hook(module, inputs, output):
            hidden = output if index == 0 else output[0]
            mask = active_attention_mask
            for b in range(hidden.shape[0]):
                valid = mask[b].bool()
                activations[index].append(
                    hidden[b, valid].detach().float().cpu()
                )
        return capture_hook

    handles = []
    if capture_activations:
        handles.append(
            model.distilbert.embeddings.register_forward_hook(
                make_capture_hook(0)
            )
        )
        for layer_idx, layer in enumerate(model.distilbert.transformer.layer):
            handles.append(
                layer.register_forward_hook(
                    make_capture_hook(layer_idx + 1)
                )
            )

    try:
        for batch in validation_loader:
            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }
            active_attention_mask = batch["attention_mask"]

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict=True,
            )

            predictions = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"]

            # GLUE test examples can have label -1, but validation_matched
            # should have labels. Exclude any missing labels defensively.
            valid = labels != -1
            all_predictions.extend(predictions[valid].cpu().tolist())
            all_labels.extend(labels[valid].cpu().tolist())

    finally:
        for handle in handles:
            handle.remove()

    metrics = {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "macro_f1": f1_score(
            all_labels, all_predictions, average="macro"
        ),
    }
    return {**metrics, "activations": activations}


def mean_cosine_by_layer(baseline, intervened):
    if len(baseline) != len(intervened):
        raise RuntimeError("Activation counts differ")

    values = []
    for base_seq, int_seq in zip(baseline, intervened):
        if base_seq.shape != int_seq.shape:
            raise RuntimeError(
                f"Activation shape mismatch: {base_seq.shape} vs {int_seq.shape}"
            )
        cos = F.cosine_similarity(base_seq, int_seq, dim=-1, eps=1e-8)
        values.append(cos)

    if not values:
        return float("nan")
    return torch.cat(values).mean().item()


print("\nEvaluating baseline on MNLI validation_matched...")
baseline = evaluate(capture_activations=True)
print(
    f"Baseline accuracy={baseline['accuracy']:.4f}, "
    f"macro-F1={baseline['macro_f1']:.4f}"
)

records = []

for dim in BOTTLENECK_DIMS:
    projection = create_projection(HIDDEN_SIZE, dim, SEED)

    for intervention_layer in range(NUM_LAYERS):
        print(
            f"\nBottleneck d={dim}; intervention at layer "
            f"{intervention_layer} (zero-indexed)"
        )

        layer = model.distilbert.transformer.layer[intervention_layer]
        handle = layer.register_forward_hook(
            make_intervention_hook(projection)
        )

        try:
            result = evaluate(capture_activations=True)
        finally:
            handle.remove()

        # Include the intervened block output and every later block output.
        for hidden_state_index in range(
            intervention_layer + 1,
            NUM_LAYERS + 1,
        ):
            similarity = mean_cosine_by_layer(
                baseline["activations"][hidden_state_index],
                result["activations"][hidden_state_index],
            )

            records.append({
                "bottleneck_dim": dim,
                "intervention_layer": intervention_layer,
                "measured_after_layer": hidden_state_index - 1,
                "cosine_similarity_vs_baseline": similarity,
                "baseline_accuracy": baseline["accuracy"],
                "intervention_accuracy": result["accuracy"],
                "accuracy_delta": result["accuracy"] - baseline["accuracy"],
                "baseline_macro_f1": baseline["macro_f1"],
                "intervention_macro_f1": result["macro_f1"],
                "macro_f1_delta": result["macro_f1"] - baseline["macro_f1"],
            })

        print(
            f"Intervened accuracy={result['accuracy']:.4f}, "
            f"macro-F1={result['macro_f1']:.4f}, "
            f"accuracy delta={result['accuracy'] - baseline['accuracy']:+.4f}"
        )

results_df = pd.DataFrame(records)
metrics_path = os.path.join(OUTPUT_DIR, "propagation_metrics.csv")
results_df.to_csv(metrics_path, index=False)

summary_df = (
    results_df[
        [
            "bottleneck_dim",
            "intervention_layer",
            "intervention_accuracy",
            "accuracy_delta",
            "intervention_macro_f1",
            "macro_f1_delta",
        ]
    ]
    .drop_duplicates()
    .sort_values(["bottleneck_dim", "intervention_layer"])
)

summary_path = os.path.join(OUTPUT_DIR, "propagation_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\nFinished.")
print("Detailed results:", metrics_path)
print("Summary:", summary_path)
print(summary_df.to_string(index=False))
