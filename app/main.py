import time

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.moderation import moderate_text


# Create the FastAPI application
app = FastAPI(
    title="Real-Time NLP Moderation Platform",
    description="An API for sentiment analysis and toxicity detection",
    version="1.0.0"
)


# Measure the processing time of each request
@app.middleware("http")
async def measure_request_time(request: Request, call_next):

    start_time = time.perf_counter()

    response = await call_next(request)

    elapsed_time = time.perf_counter() - start_time

    response.headers["X-Process-Time"] = f"{elapsed_time:.6f}"

    return response


# Define the structure of incoming requests
class ModerationRequest(BaseModel):

    text: str = Field(
        min_length=1,
        max_length=5000,
        description="Text to analyze"
    )


# Home endpoint
@app.get("/")
def home():

    return {
        "message": "NLP Moderation API is running"
    }


# Text moderation endpoint
@app.post("/moderate")
def moderate(request: ModerationRequest):

    # Remove leading and trailing whitespace
    text = request.text.strip()

    # Reject empty text
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    # Perform sentiment analysis and toxicity detection
    result = moderate_text(text)

    # Return the prediction results
    return result