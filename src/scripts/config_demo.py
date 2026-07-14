from transformers import DistilBertConfig

config = DistilBertConfig(
    dim=64,
    hidden_dim=256,
    n_heads=4,
    n_layers=6,
)

print(config)