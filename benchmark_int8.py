import os
import time
import statistics

import onnxruntime as ort

from transformers import AutoTokenizer


ORIGINAL_MODEL = "onnx_sentiment_model/model.onnx"
INT8_MODEL = "onnx_sentiment_model_int8/model.onnx"

TEST_TEXT = "This is an amazing product and I love it!"

WARMUP_RUNS = 10
BENCHMARK_RUNS = 100


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "onnx_sentiment_model"
)

# Create CPU sessions for a fairer comparison
original_session = ort.InferenceSession(
    ORIGINAL_MODEL,
    providers=["CPUExecutionProvider"]
)

int8_session = ort.InferenceSession(
    INT8_MODEL,
    providers=["CPUExecutionProvider"]
)


# Prepare input once so we measure model execution,
# rather than including tokenization in every measurement.
inputs = tokenizer(
    TEST_TEXT,
    return_tensors="np",
    truncation=True
)

model_inputs = {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"]
}


def benchmark(session):

    # Warm up the model
    for _ in range(WARMUP_RUNS):
        session.run(["logits"], model_inputs)

    latencies = []

    # Measure inference latency
    for _ in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        session.run(["logits"], model_inputs)

        elapsed = time.perf_counter() - start

        latencies.append(elapsed * 1000)

    return {
        "average_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies)
    }


def display_results(name, results):

    print(f"\n{name}")

    print("-" * 35)

    for metric, value in results.items():
        print(f"{metric}: {value:.3f} ms")


print("\nBenchmarking original ONNX model...")

original_results = benchmark(original_session)

print("Original benchmark completed.")


print("\nBenchmarking INT8 model...")

int8_results = benchmark(int8_session)

print("INT8 benchmark completed.")


# Display results
print("\n========== BENCHMARK RESULTS ==========")

display_results(
    "Original ONNX",
    original_results
)

display_results(
    "INT8 ONNX",
    int8_results
)


# Calculate speedup
speedup = (
    original_results["average_ms"]
    / int8_results["average_ms"]
)

print("\n========== PERFORMANCE COMPARISON ==========")

print(f"Measured speedup: {speedup:.2f}x")


# Measure model file sizes
original_size = os.path.getsize(ORIGINAL_MODEL)

int8_size = os.path.getsize(INT8_MODEL)

reduction = (
    (original_size - int8_size)
    / original_size
) * 100

print("\n========== MODEL SIZE ==========")

print(f"Original ONNX: {original_size / (1024 ** 2):.2f} MB")

print(f"INT8 ONNX: {int8_size / (1024 ** 2):.2f} MB")

print(f"Size reduction: {reduction:.2f}%")
