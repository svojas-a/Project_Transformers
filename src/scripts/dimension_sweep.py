import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from shared_lib.metrics import (
    compute_effective_rank,
    compute_stable_rank,
    compute_token_cosine_similarity,
)

os.makedirs("data/results", exist_ok=True)

hidden = np.load("data/processed/hidden_states.npy")

N, L, D = hidden.shape

print(hidden.shape)

#################################################
# Fit PCA ONCE
#################################################

flat = hidden.reshape(-1, D)

print("Fitting PCA...")

pca = PCA(n_components=D)
pca.fit(flat)

mean = pca.mean_
components = pca.components_

#################################################
# Dimension schedule
#################################################

dims = []

d = D

while d >= 8:

    dims.append(d)

    d = int(d * 0.8)

    d -= d % 4

#################################################
# Sweep
#################################################

results = []

centered = flat - mean

scores = centered @ components.T

for d in dims:

    print(f"Dimension {d}")

    reduced_scores = scores[:, :d]

    reconstructed = (
        reduced_scores @ components[:d]
    ) + mean

    reconstructed = reconstructed.reshape(
        N,
        L,
        D,
    )

    stable = compute_stable_rank(reconstructed).item()

    effective = compute_effective_rank(reconstructed).item()

    cosine = compute_token_cosine_similarity(
        reconstructed
    ).item()

    explained = np.sum(
        pca.explained_variance_ratio_[:d]
    )

    results.append(
        {
            "dimension": d,
            "stable_rank": stable,
            "effective_rank": effective,
            "token_cosine": cosine,
            "explained_variance": explained,
        }
    )

#################################################
# Save
#################################################

df = pd.DataFrame(results)

df.to_csv(
    "data/results/dimension_sweep.csv",
    index=False,
)

print(df)