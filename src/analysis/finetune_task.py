"""
Fine-tune DistilBERT on SST-2, MNLI or CoNLL-2003 and save the checkpoint where
backward_restoration.py looks for it:  {out_root}/{task}_finetuned

    python -m src.analysis.finetune_task --tasks sst2,mnli,conll2003

This is a plain PyTorch loop (no Trainer / accelerate / evaluate packages), so
it needs nothing beyond torch + transformers + datasets. It reuses the task
registry from backward_restoration.py, so training and probing read the data
in exactly the same way:
    sst2       single sentence            -> 2 classes
    mnli       premise + hypothesis pair  -> 3 classes (0 entail, 1 neutral, 2 contradict)
    conll2003  word tokens                -> 9 NER tags, loss on each word's
                                              FIRST subword only (the same
                                              positions accuracy is scored on)

Training uses the TRAIN split. The probe examples that M_bwd uses come from
the validation split, so the model has never seen them.

SPEED: defaults are sized for a laptop CPU (1 epoch, at most 20,000 training
examples per task, roughly 10-25 minutes per task). Full data / more epochs
need a GPU (e.g. Colab): raise --max_train_samples (0 = use everything) and
--epochs there.

SAME MODEL AS THE REST OF THE TEAM: M_fwd and M_bwd are only comparable if they
come from the same fine-tuned model. If your team already has an SST-2
checkpoint (e.g. from finetune_sst2.py on Drive), copy it to
{out_root}/sst2_finetuned and only run this for mnli and conll2003.
An existing checkpoint is never overwritten unless you pass --overwrite.
"""

import argparse
import dataclasses
import math
import os
import time

import numpy as np
import torch

from src.analysis.backward_restoration import (
    BASE_MODEL_NAME,
    TASKS,
    load_hf_split,
    row_to_example,
    tokenize,
)

NUM_LABELS = {"sst2": 2, "mnli": 3, "conll2003": 9}
IGNORE_INDEX = -100  # ignored by the loss and by accuracy


def build_features(tokenizer, spec, dataset, max_samples, seed):
    """Tokenised examples: {"input_ids", "attention_mask", "labels"}.
    For CoNLL, labels has one entry per token: the word's tag on its first
    subword, IGNORE_INDEX on [CLS]/[SEP] and on later subwords."""
    order = np.random.default_rng(seed).permutation(len(dataset))
    feats = []
    for idx in order:
        parsed = row_to_example(spec, dataset[int(idx)])
        if parsed is None:
            continue
        text, label = parsed
        enc = tokenize(tokenizer, spec, text)
        if spec.input_kind == "tokens":
            labels, prev = [], None
            for word in enc.word_ids(0):
                first_subword = word is not None and word != prev
                labels.append(label[word] if first_subword else IGNORE_INDEX)
                prev = word
        else:
            labels = label
        feats.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": labels,
            }
        )
        if max_samples and len(feats) >= max_samples:
            break
    return feats


def collate(tokenizer, batch, is_token):
    padded = tokenizer.pad(
        {
            "input_ids": [b["input_ids"] for b in batch],
            "attention_mask": [b["attention_mask"] for b in batch],
        },
        return_tensors="pt",
    )
    if is_token:
        width = padded["input_ids"].shape[1]
        labels = [
            b["labels"] + [IGNORE_INDEX] * (width - len(b["labels"])) for b in batch
        ]
    else:
        labels = [b["labels"] for b in batch]
    return {
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "labels": torch.tensor(labels),
    }


def evaluate_accuracy(model, tokenizer, feats, is_token, batch_size, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for start in range(0, len(feats), batch_size):
            batch = collate(tokenizer, feats[start : start + batch_size], is_token)
            labels = batch.pop("labels").to(device)
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            pred = logits.argmax(-1)
            keep = labels != IGNORE_INDEX
            correct += int(((pred == labels) & keep).sum())
            total += int(keep.sum())
    return correct / max(total, 1)


def train(model, tokenizer, spec, train_feats, eval_feats, args, device):
    from transformers import get_linear_schedule_with_warmup

    is_token = spec.input_kind == "tokens"
    torch.manual_seed(args["seed"])
    rng = np.random.default_rng(args["seed"])

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {
            "params": [
                p for n, p in model.named_parameters() if not n.endswith(no_decay)
            ],
            "weight_decay": args["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params, lr=args["lr"])
    steps_per_epoch = math.ceil(len(train_feats) / args["batch_size"])
    total_steps = steps_per_epoch * args["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    history = []
    step, started = 0, time.time()
    for epoch in range(args["epochs"]):
        model.train()
        order = rng.permutation(len(train_feats))
        for start in range(0, len(order), args["batch_size"]):
            idx = order[start : start + args["batch_size"]]
            batch = collate(tokenizer, [train_feats[i] for i in idx], is_token)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 50 == 0 or step == total_steps:
                print(
                    f"      step {step}/{total_steps}  loss={loss.item():.4f}  "
                    f"elapsed={time.time() - started:.0f}s"
                )
        acc = evaluate_accuracy(
            model, tokenizer, eval_feats, is_token, args["batch_size"], device
        )
        history.append(acc)
        print(
            f"      epoch {epoch + 1}/{args['epochs']}: validation accuracy = {acc:.4f}"
        )
    return history


def finetune(
    task,
    out_root="./checkpoints",
    epochs=1,
    max_train_samples=20000,
    max_eval_samples=2000,
    batch_size=16,
    lr=2e-5,
    weight_decay=0.01,
    seed=0,
    overwrite=False,
    model=None,
    tokenizer=None,
):
    """Returns the checkpoint path, or None if one already exists and
    overwrite is False. `model`/`tokenizer` can be injected (used by tests)."""
    spec = TASKS[task]
    save_path = os.path.join(out_root, f"{task}_finetuned")
    print(f"\n{'=' * 70}\nFINE-TUNING: {task}  ->  {save_path}\n{'=' * 70}")
    if os.path.isdir(save_path) and not overwrite:
        print(
            f"      '{save_path}' already exists - SKIPPING (pass --overwrite to retrain)."
        )
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoTokenizer,
        )

        cls = (
            AutoModelForTokenClassification
            if spec.input_kind == "tokens"
            else AutoModelForSequenceClassification
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        model = cls.from_pretrained(BASE_MODEL_NAME, num_labels=NUM_LABELS[task])
    model.to(device)
    print(f"      device: {device}")

    train_spec = dataclasses.replace(spec, split="train")
    print("      tokenising training data...")
    train_feats = build_features(
        tokenizer, train_spec, load_hf_split(train_spec), max_train_samples, seed
    )
    eval_feats = build_features(
        tokenizer, spec, load_hf_split(spec), max_eval_samples, seed
    )
    print(f"      {len(train_feats)} train / {len(eval_feats)} validation examples")

    train(
        model,
        tokenizer,
        spec,
        train_feats,
        eval_feats,
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
        },
        device,
    )
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"      saved -> {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT per task")
    parser.add_argument("--tasks", default="sst2,mnli,conll2003")
    parser.add_argument("--out_root", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_samples", type=int, default=20000, help="0 = use all"
    )
    parser.add_argument("--max_eval_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for task in args.tasks.split(","):
        task = task.strip()
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Available: {list(TASKS)}")
        finetune(
            task,
            out_root=args.out_root,
            epochs=args.epochs,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()