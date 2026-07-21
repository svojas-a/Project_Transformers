from transformers import DistilBertConfig, DistilBertModel

# Create a custom configuration
config = DistilBertConfig(
    dim=64,
    hidden_dim=256,
    n_heads=4,
    n_layers=6,
)

# Build the model from the configuration
model = DistilBertModel(config)

print("✅ Custom model created!\n")

print(model)

print("\nConfiguration:")
print(model.config)
