# Real-Time NLP Moderation Platform

A machine learning application that analyzes text using Natural Language Processing (NLP) to perform sentiment analysis and toxicity detection.

The application exposes a REST API using FastAPI and uses pretrained transformer models to analyze incoming text. The sentiment classification model is exported to ONNX and dynamically quantized to INT8 to reduce model size and improve inference performance.

## Features

- Sentiment analysis using a pretrained DistilBERT model.
- Toxicity detection using a pretrained Toxic-BERT model.
- REST API built with FastAPI.
- Input validation using Pydantic.
- ONNX Runtime inference for sentiment classification.
- Dynamic INT8 quantization of the sentiment model.
- Configurable toxicity threshold.
- Request processing-time measurement.
- Automated API tests using Pytest.
- Benchmarking of model latency and size.
- Prediction consistency checks between original and quantized models.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Hugging Face Transformers | Loading pretrained NLP models |
| DistilBERT | Sentiment classification |
| Toxic-BERT | Toxicity detection |
| FastAPI | REST API development |
| Pydantic | Request validation |
| PyTorch | Original model inference and model tooling |
| ONNX | Portable machine learning model format |
| ONNX Runtime | Executing the exported models |
| INT8 Quantization | Reducing model weight precision |
| Pytest | Automated testing |

## Project Architecture

The application follows this workflow:

1. A client submits text through the FastAPI endpoint.
2. FastAPI validates the request.
3. The INT8-quantized ONNX model predicts sentiment.
4. The Toxic-BERT model estimates the toxicity score.
5. The application compares the toxicity score against a configurable threshold.
6. The API returns the predictions and moderation decision as JSON.

### Architecture Diagram

```text
                  Client
                    |
                    v
                FastAPI API
                    |
                    v
               Input Validation
                    |
          +---------+---------+
          |                   |
          v                   v
    INT8 ONNX Model       Toxic-BERT
    Sentiment Analysis    Toxicity Detection
          |                   |
          +---------+---------+
                    |
                    v
             Moderation Logic
                    |
                    v
               JSON Response
```

## Project Structure

```text
real-time-nlp-moderation/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── moderation.py
│
├── tests/
│   └── test_api.py
│
├── onnx_sentiment_model/
│   ├── model.onnx
│   └── tokenizer and configuration files
│
├── onnx_sentiment_model_int8/
│   ├── model.onnx
│   └── tokenizer and configuration files
│
├── benchmark_onnx.py
├── benchmark_int8.py
├── test_onnx.py
├── test_int8.py
├── evaluate_quantization.py
├── quantize_onnx.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### Prerequisites

- Python 3.11 or a compatible Python version supported by the installed dependencies.
- pip
- Internet access for downloading pretrained models.
- Sufficient memory and storage for the NLP models.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd real-time-nlp-moderation
```

Replace the placeholder URL with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "optimum[onnxruntime]" accelerate
```

These commands install the application dependencies and the tools used for ONNX export and optimization.

## Configuration

Create a `.env` file in the project root:

```env
TOXICITY_THRESHOLD=0.70
```

The toxicity threshold determines when the application flags a comment.

For example, if the threshold is `0.70`, the application flags a comment when its predicted toxicity score is greater than or equal to `0.70`.

The threshold is a configurable decision rule, not a guarantee of moderation accuracy.

Keep `.env` out of version control.

## Running the Application

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

http://127.0.0.1:8000

Interactive API documentation is available at:

http://127.0.0.1:8000/docs

### Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Check whether the API is running |
| POST | `/moderate` | Analyze text for sentiment and toxicity |

## Example API Request

Send a POST request to:

```text
http://127.0.0.1:8000/moderate
```

Request body:

```json
{
  "text": "This is an amazing product!"
}
```

Example response structure:

```json
{
  "text": "This is an amazing product!",
  "sentiment": "positive",
  "sentiment_score": 0.9999,
  "toxicity_score": 0.0001,
  "toxicity_threshold": 0.7,
  "moderation_decision": "allow"
}
```

The numerical values above are illustrative. Actual predictions depend on model inference.

## Model Optimization

The sentiment classification model was exported to ONNX and optimized using dynamic INT8 quantization.

### Benchmark Results

The following measurements were obtained during local experiments on a Mac.

| Metric | Original ONNX | INT8 ONNX |
|---|---:|---:|
| Model size | 255.52 MB | 64.23 MB |
| Average CPU inference latency | 4.593 ms | 1.695 ms |
| Median CPU inference latency | 4.449 ms | 1.668 ms |

### Measured Improvements

The measured inference speedup was approximately 2.71x.

Speedup is calculated as:

Speedup = Original average latency / INT8 average latency

Using the measured values:

Speedup = 4.593 / 1.695 = approximately 2.71x

The model file size decreased by approximately 74.86%.

Size reduction = ((Original size - INT8 size) / Original size) × 100

These measurements describe the sentiment model's CPU inference benchmark. They do not represent the latency of the complete moderation API.

### Quantization Evaluation

The original ONNX model and the INT8 model were compared using ten sample sentences.

- Test cases: 10
- Matching predicted labels: 10
- Prediction agreement: 100%
- Mean maximum probability difference: 0.000189

This is a small consistency check, not a comprehensive model accuracy evaluation. A larger, independently labelled dataset is needed to assess real-world classification performance.

## Testing

Run the automated API tests:

```bash
python -m pytest -v
```

Run the ONNX sentiment prediction test:

```bash
python test_onnx.py
```

Run the INT8 prediction test:

```bash
python test_int8.py
```

Evaluate prediction consistency:

```bash
python evaluate_quantization.py
```

Benchmark the original PyTorch pipeline against ONNX:

```bash
python benchmark_onnx.py
```

Benchmark the original ONNX model against INT8:

```bash
python benchmark_int8.py
```

## Current Limitations

- The sentiment model supports positive and negative sentiment classification; it does not provide a dedicated neutral class.
- Toxicity detection can produce false positives and false negatives.
- A single toxicity threshold may not suit every application.
- The initial quantization evaluation uses only ten sample sentences.
- Benchmark results depend on hardware, execution provider, and measurement conditions.
- The sentiment model has been optimized; the toxicity model has not yet undergone the same optimization process.
- The application currently processes requests synchronously through its API endpoint.

## Future Improvements

Planned extensions include:

- Kafka-based streaming of incoming comments.
- Prometheus metrics for request counts, latency, and errors.
- Grafana dashboards for monitoring application performance.
- Data drift detection for incoming text.
- Larger-scale model evaluation with labelled datasets.
- Optimization and benchmarking of the toxicity model.
- Improved moderation policies and configurable review decisions.
- Containerization and deployment.

These are planned extensions and should not be considered implemented until they have been built and tested.

## Learning Outcomes

This project provides practical experience with:

- Natural Language Processing.
- Transformer-based text classification.
- REST API development using FastAPI.
- Model export using ONNX.
- Dynamic INT8 quantization.
- Inference latency benchmarking.
- Model-size comparison.
- Prediction consistency evaluation.
- Automated testing and input validation.

## Disclaimer

This project is an educational prototype. Its moderation decisions should not be treated as definitive judgments about user intent or as a substitute for human review in high-impact situations.