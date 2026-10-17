import numpy as np
import onnxruntime as ort

from transformers import AutoTokenizer, AutoConfig


# Model directories
ORIGINAL_MODEL_PATH = "onnx_sentiment_model"
INT8_MODEL_PATH = "onnx_sentiment_model_int8"

# Test sentence
text = "This is an amazing product and I love it!"


# Load tokenizer and label configuration
tokenizer = AutoTokenizer.from_pretrained(
    ORIGINAL_MODEL_PATH
)

config = AutoConfig.from_pretrained(
    ORIGINAL_MODEL_PATH
)


# Create CPU inference sessions
original_session = ort.InferenceSession(
    f"{ORIGINAL_MODEL_PATH}/model.onnx",
    providers=["CPUExecutionProvider"]
)

int8_session = ort.InferenceSession(
    f"{INT8_MODEL_PATH}/model.onnx",
    providers=["CPUExecutionProvider"]
)


# Convert text into numerical inputs
inputs = tokenizer(
    text,
    return_tensors="np",
    truncation=True
)


def predict(session, inputs):

    # Run inference
    outputs = session.run(
        ["logits"],
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
    )

    logits = outputs[0]

    # Convert logits into probabilities
    logits = logits - np.max(logits, axis=-1, keepdims=True)

    probabilities = np.exp(logits)

    probabilities = probabilities / np.sum(
        probabilities,
        axis=-1,
        keepdims=True
    )

    # Identify predicted class
    class_id = int(np.argmax(probabilities, axis=-1)[0])

    label = config.id2label[class_id]

    confidence = float(probabilities[0][class_id])

    return label, confidence


# Test original ONNX model
original_label, original_score = predict(
    original_session,
    inputs
)

# Test INT8 model
int8_label, int8_score = predict(
    int8_session,
    inputs
)


# Display results
print("\n========== INT8 MODEL TEST ==========")

print("\nInput:", text)

print("\nOriginal ONNX prediction:")
print("Label:", original_label)
print("Confidence:", round(original_score, 6))

print("\nINT8 prediction:")
print("Label:", int8_label)
print("Confidence:", round(int8_score, 6))

print("\n========== COMPARISON ==========")

print("Labels match:", original_label == int8_label)

print(
    "Confidence difference:",
    round(abs(original_score - int8_score), 6)
)
