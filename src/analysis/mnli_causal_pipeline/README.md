# MNLI causal propagation pipeline

## Contents
- `train_mnli.py`: fine-tunes DistilBERT on GLUE MNLI.
- `mnli_causal_propagation.py`: evaluates a rank-constrained activation intervention at each of six Transformer layers.
- `requirements.txt`: remaining Python dependencies.

## Environment
Use the same Python virtual environment as the CoNLL-2003 experiment if it has compatible packages.

Install PyTorch separately if needed. For an NVIDIA GPU, choose a CUDA build using https://pytorch.org/get-started/locally/. Verify:
```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Install requirements:
```powershell
python -m pip install -r requirements.txt
```

## Run
From this directory:
```powershell
python train_mnli.py
python mnli_causal_propagation.py
```

The fine-tuned checkpoint is saved to `./mnli_checkpoint`.
Propagation outputs are saved to `./mnli_causal_results/`.

## Experimental details
- Dataset: GLUE MNLI.
- Primary split: `validation_matched`.
- Model: `distilbert-base-uncased`, six Transformer blocks, hidden size 768.
- Labels: entailment, neutral, contradiction.
- Intervention: fixed random orthogonal rank-d projection, applied to one block output at a time and reconstructed to hidden size 768.
- Model parameters remain frozen during intervention.
- `intervention_layer` and `measured_after_layer` are zero-indexed.
- Report accuracy and macro-F1, plus layer-wise cosine similarity.

## Research cautions
1. This is a random orthogonal projection experiment, not automatically the same as a learned projection used by a separate dimensional-collapse method.
2. One seed is a smoke-test. Repeat multiple projection seeds and report mean and standard deviation.
3. Use validation_matched for development. After fixing the protocol, evaluate the final selected experiment on the appropriate held-out split.
4. Cosine similarity is a geometric comparison, not proof of semantic information loss.
5. Compare MNLI matched and mismatched validation splits as a separate generalization analysis, not as interchangeable scores.
