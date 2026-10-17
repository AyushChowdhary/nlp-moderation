import os

import numpy as np
import onnxruntime as ort

from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoConfig,
    pipeline
)


# Load environment configuration
load_dotenv()

TOXICITY_THRESHOLD = float(
    os.getenv("TOXICITY_THRESHOLD", "0.70")
)

if not 0.0 <= TOXICITY_THRESHOLD <= 1.0:
    raise ValueError(
        "TOXICITY_THRESHOLD must be between 0 and 1"
    )


# ==========================================
# LOAD INT8 SENTIMENT MODEL
# ==========================================

INT8_MODEL_PATH = "onnx_sentiment_model_int8"

sentiment_tokenizer = AutoTokenizer.from_pretrained(
    INT8_MODEL_PATH
)

sentiment_config = AutoConfig.from_pretrained(
    INT8_MODEL_PATH
)

sentiment_session = ort.InferenceSession(
    os.path.join(INT8_MODEL_PATH, "model.onnx"),
    providers=["CPUExecutionProvider"]
)


# ==========================================
# LOAD TOXICITY MODEL
# ==========================================

toxicity_model = pipeline(
    task="text-classification",
    model="unitary/toxic-bert",
    top_k=None,
    device=-1
)


# ==========================================
# SENTIMENT PREDICTION
# ==========================================

def predict_sentiment(text: str) -> dict:
    """Predict sentiment using the INT8 ONNX model."""

    # Convert text into numerical inputs
    inputs = sentiment_tokenizer(
        text,
        return_tensors="np",
        truncation=True
    )

    # Run INT8 model inference
    outputs = sentiment_session.run(
        ["logits"],
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
    )

    logits = outputs[0]

    # Convert logits into probabilities
    logits = logits - np.max(
        logits,
        axis=-1,
        keepdims=True
    )

    probabilities = np.exp(logits)

    probabilities /= np.sum(
        probabilities,
        axis=-1,
        keepdims=True
    )

    probabilities = probabilities[0]

    # Find the predicted class
    predicted_id = int(np.argmax(probabilities))

    predicted_label = sentiment_config.id2label[
        predicted_id
    ]

    confidence = float(probabilities[predicted_id])

    return {
        "sentiment": predicted_label.lower(),
        "sentiment_score": round(confidence, 4)
    }


# ==========================================
# TOXICITY PREDICTION
# ==========================================

def predict_toxicity(text: str) -> float:
    """Return the model's score for the TOXIC label."""

    results = toxicity_model(text)[0]

    for result in results:

        if result["label"].lower() == "toxic":
            return float(result["score"])

    return 0.0


# ==========================================
# COMPLETE MODERATION PIPELINE
# ==========================================

def moderate_text(text: str) -> dict:
    """Analyze sentiment and toxicity."""

    sentiment_result = predict_sentiment(text)

    toxicity_score = predict_toxicity(text)

    if toxicity_score >= TOXICITY_THRESHOLD:
        decision = "flag"
    else:
        decision = "allow"

    return {
        "text": text,
        "sentiment": sentiment_result["sentiment"],
        "sentiment_score": sentiment_result["sentiment_score"],
        "toxicity_score": round(toxicity_score, 4),
        "toxicity_threshold": TOXICITY_THRESHOLD,
        "moderation_decision": decision
    }