🎬 DistilBERT Sentiment Service

Production-ready FastAPI backend for sentiment analysis using a Hugging Face DistilBERT model.
Deployed on Google Cloud Run.

---

🌐 Live API

- Base URL:
  https://distilbert-sentiment-service-342313441373.europe-west1.run.app

- API Docs (Swagger):
  https://distilbert-sentiment-service-342313441373.europe-west1.run.app/docs

---

🚀 Overview

This project exposes a REST API for movie review sentiment analysis using a transformer-based deep learning model.

It returns:

- sentiment prediction (Positive / Negative)
- confidence score (%)
- confidence level (Low / Medium / High)

---

🧠 Model

- Model: "distilbert-base-uncased-finetuned-sst-2-english"
- Framework: Hugging Face Transformers + PyTorch
- Task: Binary sentiment classification

---

⚙️ Tech Stack

- FastAPI
- Hugging Face Transformers
- PyTorch
- Docker
- Google Cloud Run
- GitHub (CI/CD via Cloud Build)

---

📡 API Endpoints

🔹 GET /

Returns API status.

Response:

{
  "message": "DistilBERT Sentiment Service is running",
  "docs": "/docs"
}

---

🔹 POST /predict

Predicts sentiment for a single review.

Request:

{
  "review": "This movie was absolutely amazing. I loved every minute of it."
}

Response:

{
  "prediction": "Positive",
  "confidence": 99.12,
  "confidence_label": "High"
}

---

▶️ Run Locally

1. Install dependencies

python -m pip install -r requirements.txt

2. Run server

python -m uvicorn main:app --reload

3. Open docs

http://127.0.0.1:8000/docs

---

🐳 Docker

Build image:

docker build -t sentiment-api .

Run container:

docker run -p 8080:8080 sentiment-api

---

☁️ Deployment

This project is deployed on Google Cloud Run using:

- Cloud Build (from GitHub repo)
- Docker container
- Automatic scaling

---

📁 Project Structure

distilbert-sentiment-service/
├── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md

---

👨‍💻 Author

Dejan Jović

--- 
