import os
import time
import statistics

from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_NAME = (
    "distilbert/"
    "distilbert-base-uncased-finetuned-sst-2-english"
)

ONNX_MODEL_PATH = "onnx_sentiment_model"

TEST_TEXT = "This is an amazing product and I love it!"

WARMUP_RUNS = 5
BENCHMARK_RUNS = 30


# ==========================================
# LOAD PYTORCH MODEL
# ==========================================

print("\nLoading PyTorch model...")

pytorch_model = pipeline(
    task="sentiment-analysis",
    model=MODEL_NAME,
    device=-1
)

print("PyTorch model loaded successfully.")


# ==========================================
# LOAD ONNX MODEL
# ==========================================

print("\nLoading ONNX model...")

tokenizer = AutoTokenizer.from_pretrained(
    ONNX_MODEL_PATH
)

onnx_model = ORTModelForSequenceClassification.from_pretrained(
    ONNX_MODEL_PATH
)

onnx_pipeline = pipeline(
    task="sentiment-analysis",
    model=onnx_model,
    tokenizer=tokenizer
)

print("ONNX model loaded successfully.")


# ==========================================
# BENCHMARK FUNCTION
# ==========================================

def benchmark(model, text):
    """
    Measure inference latency.

    Returns average, median, minimum,
    and maximum latency in milliseconds.
    """

    # Warm-up runs
    for _ in range(WARMUP_RUNS):
        model(text)

    latencies = []

    # Actual benchmark runs
    for _ in range(BENCHMARK_RUNS):

        start_time = time.perf_counter()

        model(text)

        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        latencies.append(latency_ms)

    return {
        "average_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies)
    }


# ==========================================
# RUN BENCHMARKS
# ==========================================

print("\nBenchmarking PyTorch model...")

pytorch_results = benchmark(
    pytorch_model,
    TEST_TEXT
)

print("PyTorch benchmark completed.")


print("\nBenchmarking ONNX model...")

onnx_results = benchmark(
    onnx_pipeline,
    TEST_TEXT
)

print("ONNX benchmark completed.")


# ==========================================
# DISPLAY RESULTS
# ==========================================

def display_results(name, results):

    print(f"\n{name}")

    print("-" * 35)

    for metric, value in results.items():

        print(f"{metric}: {value:.3f} ms")


print("\n========== BENCHMARK RESULTS ==========")

display_results(
    "PyTorch",
    pytorch_results
)

display_results(
    "ONNX Runtime",
    onnx_results
)


# ==========================================
# CALCULATE SPEEDUP
# ==========================================

pytorch_latency = pytorch_results["average_ms"]

onnx_latency = onnx_results["average_ms"]

if onnx_latency > 0:

    speedup = pytorch_latency / onnx_latency

    print("\n========== PERFORMANCE COMPARISON ==========")

    print(f"PyTorch average latency: {pytorch_latency:.3f} ms")

    print(f"ONNX average latency: {onnx_latency:.3f} ms")

    print(f"Measured speedup: {speedup:.2f}x")

    if speedup > 1:
        print("ONNX was faster in this benchmark.")

    elif speedup < 1:
        print("PyTorch was faster in this benchmark.")

    else:
        print("Both models had approximately equal average latency.")


# ==========================================
# COMPARE MODEL PREDICTIONS
# ==========================================

print("\n========== PREDICTION COMPARISON ==========")

pytorch_prediction = pytorch_model(TEST_TEXT)[0]

onnx_prediction = onnx_pipeline(TEST_TEXT)[0]

print("\nInput text:")
print(TEST_TEXT)

print("\nPyTorch prediction:")
print(pytorch_prediction)

print("\nONNX prediction:")
print(onnx_prediction)

same_label = (
    pytorch_prediction["label"] == onnx_prediction["label"]
)

print("\nPredicted labels match:", same_label)


# ==========================================
# MEASURE ONNX MODEL FILE SIZE
# ==========================================

onnx_file = os.path.join(
    ONNX_MODEL_PATH,
    "model.onnx"
)

if os.path.exists(onnx_file):

    size_mb = os.path.getsize(onnx_file) / (1024 * 1024)

    print("\n========== MODEL SIZE ==========")

    print(f"ONNX model size: {size_mb:.2f} MB")

else:

    print("\nONNX model file not found at:", onnx_file)
    