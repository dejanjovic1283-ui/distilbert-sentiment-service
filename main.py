from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

# Global variable for the loaded classifier
classifier = None

# Label mapping for the fine-tuned model output
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Positive"
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the BERT sentiment classifier once when the API starts.
    """
    global classifier
    classifier = pipeline(
        "sentiment-analysis",
        model="imdb_distilbert_model",
        tokenizer="imdb_distilbert_model"
    )
    yield


app = FastAPI(
    title="DistilBERT Sentiment Service",
    description="FastAPI backend for sentiment analysis using a fine-tuned DistilBERT model.",
    version="1.0.0",
    lifespan=lifespan
)


class ReviewRequest(BaseModel):
    # Request body schema
    review: str = Field(..., min_length=3, description="A single movie review text")


class PredictionResponse(BaseModel):
    # Response body schema
    prediction: str
    confidence: float
    confidence_label: str


@app.get("/")
def read_root():
    """
    Root endpoint used to confirm that the API is running.
    """
    return {
        "message": "DistilBERT Sentiment Service is running",
        "docs": "/docs"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(payload: ReviewRequest):
    """
    Predict sentiment for a single review using the fine-tuned DistilBERT model.
    """
    result = classifier(payload.review)[0]

    raw_label = result["label"]
    mapped_label = label_map.get(raw_label, raw_label)
    score = float(result["score"])

    if score >= 0.90:
        confidence_label = "High"
    elif score >= 0.70:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    return PredictionResponse(
        prediction=mapped_label,
        confidence=round(score * 100, 2),
        confidence_label=confidence_label
    )