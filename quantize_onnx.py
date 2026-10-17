import os
import shutil

from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType
)


# Paths
SOURCE_MODEL = "onnx_sentiment_model"
OUTPUT_MODEL = "onnx_sentiment_model_int8"


def main():

    original_file = os.path.join(
        SOURCE_MODEL,
        "model.onnx"
    )

    output_file = os.path.join(
        OUTPUT_MODEL,
        "model.onnx"
    )

    # Check that the original model exists.
    if not os.path.isfile(original_file):
        raise FileNotFoundError(
            f"Original ONNX model not found: {original_file}"
        )

    # Create the output directory with the model's
    # configuration and tokenizer files.
    if os.path.exists(OUTPUT_MODEL):
        shutil.rmtree(OUTPUT_MODEL)

    shutil.copytree(
        SOURCE_MODEL,
        OUTPUT_MODEL
    )

    print("Original model found.")
    print("Starting INT8 quantization...")

    # Quantize supported model weights.
    quantize_dynamic(
        model_input=original_file,
        model_output=output_file,
        weight_type=QuantType.QInt8
    )

    # Measure file sizes.
    original_size = os.path.getsize(original_file)

    quantized_size = os.path.getsize(output_file)

    original_mb = original_size / (1024 * 1024)

    quantized_mb = quantized_size / (1024 * 1024)

    reduction = (
        (original_size - quantized_size)
        / original_size
    ) * 100

    print("\n========== QUANTIZATION RESULTS ==========")

    print(f"Original model size: {original_mb:.2f} MB")

    print(f"INT8 model size: {quantized_mb:.2f} MB")

    print(f"Size reduction: {reduction:.2f}%")

    print("\nINT8 quantization completed successfully.")

    print(f"Saved model: {output_file}")


if __name__ == "__main__":
    main()
    