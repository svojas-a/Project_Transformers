import os
import numpy as np
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODEL = "bert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 32
OUTPUT = "data/processed"

os.makedirs(OUTPUT, exist_ok=True)

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

print("Loading dataset...")

dataset = load_dataset("glue", "stsb", split="validation")

sentences = []

for row in dataset:
    sentences.append(row["sentence1"])
    sentences.append(row["sentence2"])

# remove duplicates
sentences = list(dict.fromkeys(sentences))

print(f"{len(sentences)} unique sentences")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)

model.eval()
model.to(device)

all_hidden = []

with torch.no_grad():

    for i in tqdm(range(0, len(sentences), BATCH_SIZE)):

        batch = sentences[i:i+BATCH_SIZE]

        encoded = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = model(**encoded)

        hidden = outputs.last_hidden_state.cpu().numpy()

        all_hidden.append(hidden)

hidden = np.concatenate(all_hidden)

print(hidden.shape)

np.save(
    "data/processed/hidden_states.npy",
    hidden,
)

print("Saved.")