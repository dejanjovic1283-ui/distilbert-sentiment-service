from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

# Global variable for the loaded classifier
classifier = None

# Label mapping for the model output
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Positive"
}


def get_classifier():
    """
    Lazily load the sentiment model on the first request.
    This helps Cloud Run start the container faster.
    """
    global classifier
    if classifier is None:
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return classifier


app = FastAPI(
    title="DistilBERT Sentiment Service",
    description="FastAPI backend for sentiment analysis using a Hugging Face DistilBERT model.",
    version="1.0.1"
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


@app.get("/health")
def health_check():
    """
    Health check endpoint for Cloud Run.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(payload: ReviewRequest):
    """
    Predict sentiment for a single review.
    """
    model = get_classifier()
    result = model(payload.review)[0]

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
