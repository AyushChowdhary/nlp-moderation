from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import torch


MODEL_PATH = "onnx_sentiment_model"

# Load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load the ONNX model.
model = ORTModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

# Test sentence.
text = "This is a terrible product."

# Convert text into model inputs.
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True
)

# Run inference.
with torch.no_grad():
    outputs = model(**inputs)

# Convert logits into probabilities.
probabilities = torch.softmax(outputs.logits, dim=-1)

# Find the predicted class.
predicted_class_id = probabilities.argmax(dim=-1).item()

# Get the label associated with that class.
predicted_label = model.config.id2label[predicted_class_id]

print("Input text:", text)
print("Predicted sentiment:", predicted_label)
print("Class probabilities:", probabilities.tolist())
