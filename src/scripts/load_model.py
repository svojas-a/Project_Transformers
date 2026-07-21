from transformers import DistilBertModel, DistilBertTokenizer

print("Loading tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

print("Loading model...")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

print("\n✅ Model loaded successfully!\n")

print("Model Architecture:")
print(model)

print("\nModel Configuration:")
print(model.config)
