import time
from typing import List
from contextlib import asynccontextmanager
import keras
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import (
    SentimentRequest,
    SentimentResponse,
    BatchSentimentRequest,
    BatchSentimentResponse,
    SampleReview,
    HealthResponse
)
from app.samples import SAMPLE_REVIEWS
from app.model_service import model_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model is initialized on import, ensure it's healthy
    if not model_service.is_loaded:
        model_service.load_model()
    yield

app = FastAPI(
    title="KhushiDL - Movie Sentiment Review API",
    description="Deep Learning Movie Review Sentiment Classification Powered by Keras 3 & CNNs",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse, tags=["System"])
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check endpoint to verify backend service and model readiness."""
    return HealthResponse(
        status="healthy" if model_service.is_loaded else "degraded",
        app_name="KhushiDL - Movie Sentiment Review API",
        version="1.0.0",
        model_loaded=model_service.is_loaded,
        keras_version=keras.__version__,
        architecture="Sequential(TextVectorization -> Embedding -> Conv1D -> GlobalAveragePooling1D -> Dense)"
    )

@app.post("/api/predict", response_model=SentimentResponse, tags=["Inference"])
def predict_sentiment(payload: SentimentRequest):
    """
    Analyze sentiment of a single movie review text.
    Returns sentiment classification ('Positive' or 'Negative'), confidence %,
    probability score, star rating, and highlight cues.
    """
    text = payload.review.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Review text cannot be empty or solely whitespace."
        )

    try:
        result = model_service.predict_single(text)
        return SentimentResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@app.post("/api/predict/batch", response_model=BatchSentimentResponse, tags=["Inference"])
def predict_batch_sentiment(payload: BatchSentimentRequest):
    """
    Batch analyze multiple movie reviews.
    """
    if not payload.reviews:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The reviews list cannot be empty."
        )

    try:
        results, summary = model_service.predict_batch(payload.reviews)
        return BatchSentimentResponse(
            total=summary["total"],
            positive_count=summary["positive_count"],
            negative_count=summary["negative_count"],
            average_score=summary["average_score"],
            results=[SentimentResponse(**r) for r in results]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}"
        )

@app.get("/api/samples", response_model=List[SampleReview], tags=["Data"])
def get_sample_reviews():
    """Returns curated sample reviews for quick demonstration and testing."""
    return [SampleReview(**s) for s in SAMPLE_REVIEWS]

@app.get("/api/model-info", tags=["System"])
def get_model_info():
    """Returns technical details regarding the loaded deep learning neural network."""
    return {
        "model_name": "KhushiDL Sentiment CNN",
        "keras_version": keras.__version__,
        "pipeline": [
            {"layer": 1, "type": "InputLayer", "dtype": "string"},
            {"layer": 2, "type": "TextVectorization", "max_tokens": 10000, "sequence_length": 250},
            {"layer": 3, "type": "Embedding", "input_dim": 10001, "output_dim": 64},
            {"layer": 4, "type": "Dropout", "rate": 0.5},
            {"layer": 5, "type": "Conv1D", "filters": 64, "kernel_size": 5, "stride": 2, "activation": "relu"},
            {"layer": 6, "type": "GlobalAveragePooling1D"},
            {"layer": 7, "type": "Dense", "units": 1, "activation": "sigmoid"}
        ],
        "input_format": "Raw string text (e.g. 'This movie was phenomenal!')",
        "output_format": "Sigmoid probability [0.0 = Pure Negative, 1.0 = Pure Positive]",
        "decision_threshold": 0.5
    }
