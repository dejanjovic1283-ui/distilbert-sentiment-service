# 🎬 DistilBERT Sentiment Service

Production-ready FastAPI backend for sentiment analysis using a fine-tuned DistilBERT model.

---

## 🚀 Overview

This project exposes a REST API for movie review sentiment analysis using a transformer-based deep learning model.

It returns:
- sentiment prediction
- confidence score
- confidence level

---

## 🧠 Model

- Base model: `distilbert-base-uncased`
- Fine-tuned for IMDB sentiment classification
- Framework: Hugging Face Transformers + PyTorch

---

## 📡 API Endpoints

### `GET /`
Returns a simple status response.

### `POST /predict`
Predicts sentiment for a single review.

Example request:

```json
{
  "review": "This movie was absolutely amazing. I loved every minute of it."
}

Example response:

{
  "prediction": "Positive",
  "confidence": 98.99,
  "confidence_label": "High"
}

▶️ Run Locally
Install dependencies:
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload

📁 Project Structure
distilbert-sentiment-service/
├── imdb_distilbert_model/
├── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md

👨‍💻 Author
Dejan Jović
