import os
import numpy as np
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


MODEL_NAME = "bert-base-uncased"

OUTPUT_DIR = "data/processed"

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


os.makedirs(OUTPUT_DIR, exist_ok=True)


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()

    summed = torch.sum(last_hidden * mask, dim=1)

    counts = torch.clamp(mask.sum(dim=1), min=1e-9)

    return summed / counts


print("Loading STS-B...")

dataset = load_dataset("glue", "stsb", split="validation")

sentences = []

for row in dataset:
    sentences.append(row["sentence1"])
    sentences.append(row["sentence2"])

sentences = list(dict.fromkeys(sentences))

print(f"Unique sentences : {len(sentences)}")


print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(MODEL_NAME)

model.to(DEVICE)

model.eval()


embeddings = []


with torch.no_grad():

    for sentence in tqdm(sentences):

        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        inputs = {
            k: v.to(DEVICE)
            for k, v in inputs.items()
        }

        outputs = model(**inputs)

        emb = mean_pool(
            outputs.last_hidden_state,
            inputs["attention_mask"],
        )

        embeddings.append(
            emb.squeeze().cpu().numpy()
        )


embeddings = np.stack(embeddings)

print(embeddings.shape)


np.save(
    os.path.join(OUTPUT_DIR, "embeddings.npy"),
    embeddings,
)

with open(
    os.path.join(OUTPUT_DIR, "sentences.txt"),
    "w",
) as f:

    for s in sentences:
        f.write(s + "\n")


print("Embeddings saved.")