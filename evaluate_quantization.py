import numpy as np
import onnxruntime as ort

from transformers import AutoTokenizer, AutoConfig


ORIGINAL_MODEL = "onnx_sentiment_model/model.onnx"
INT8_MODEL = "onnx_sentiment_model_int8/model.onnx"

TOKENIZER_PATH = "onnx_sentiment_model"


# Test sentences covering different types of sentiment.
TEST_CASES = [
    ("I absolutely love this product!", "POSITIVE"),
    ("This is the worst experience ever.", "NEGATIVE"),
    ("The product works exactly as expected.", "POSITIVE"),
    ("I am extremely disappointed with the service.", "NEGATIVE"),
    ("The movie was fantastic and entertaining.", "POSITIVE"),
    ("The food was terrible and cold.", "NEGATIVE"),
    ("I really enjoyed using this application.", "POSITIVE"),
    ("The service was awful.", "NEGATIVE"),
    ("This is an excellent purchase.", "POSITIVE"),
    ("I hate this product.", "NEGATIVE"),
]


# Load tokenizer and label configuration.
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

config = AutoConfig.from_pretrained(TOKENIZER_PATH)


# Use the CPU execution provider for both models.
original_session = ort.InferenceSession(
    ORIGINAL_MODEL,
    providers=["CPUExecutionProvider"]
)

int8_session = ort.InferenceSession(
    INT8_MODEL,
    providers=["CPUExecutionProvider"]
)


def predict(session, text):
    """Return predicted label and class probabilities."""

    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True
    )

    outputs = session.run(
        ["logits"],
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
    )

    logits = outputs[0]

    # Stable softmax calculation.
    logits = logits - np.max(logits, axis=-1, keepdims=True)

    probabilities = np.exp(logits)

    probabilities /= np.sum(
        probabilities,
        axis=-1,
        keepdims=True
    )

    probabilities = probabilities[0]

    predicted_id = int(np.argmax(probabilities))

    predicted_label = config.id2label[predicted_id]

    return predicted_label, probabilities


# Evaluation.
matching_labels = 0
confidence_differences = []

print("\n========== QUANTIZATION EVALUATION ==========")

for text, expected_label in TEST_CASES:

    original_label, original_probs = predict(
        original_session,
        text
    )

    int8_label, int8_probs = predict(
        int8_session,
        text
    )

    labels_match = original_label == int8_label

    if labels_match:
        matching_labels += 1

    confidence_difference = float(
        np.max(np.abs(original_probs - int8_probs))
    )

    confidence_differences.append(confidence_difference)

    print("\nText:", text)

    print("Expected label:", expected_label)

    print("Original ONNX:", original_label)

    print("INT8:", int8_label)

    print("Labels match:", labels_match)

    print(
        "Maximum probability difference:",
        round(confidence_difference, 6)
    )


agreement = matching_labels / len(TEST_CASES) * 100

mean_difference = float(
    np.mean(confidence_differences)
)

print("\n========== SUMMARY ==========")

print(f"Test cases: {len(TEST_CASES)}")

print(f"Matching predictions: {matching_labels}")

print(f"Prediction agreement: {agreement:.2f}%")

print(
    "Mean maximum probability difference:",
    round(mean_difference, 6)
)
