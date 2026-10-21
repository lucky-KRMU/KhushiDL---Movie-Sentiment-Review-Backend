from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class SentimentRequest(BaseModel):
    review: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The movie review text to analyze"
    )

class SentimentResponse(BaseModel):
    review: str
    sentiment: str  # "Positive" or "Negative"
    score: float    # Raw model probability [0.0 - 1.0]
    confidence: float # Percentage [50.0 - 100.0]
    intensity: str  # e.g., "Overwhelmingly Positive", "Strongly Negative"
    star_rating: float # 1.0 to 5.0 scale
    word_count: int
    inference_time_ms: float
    detected_cues: Dict[str, List[str]] # Positive & Negative highlight tokens

class BatchSentimentRequest(BaseModel):
    reviews: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of movie reviews to analyze in batch"
    )

class BatchSentimentResponse(BaseModel):
    total: int
    positive_count: int
    negative_count: int
    average_score: float
    results: List[SentimentResponse]

class SampleReview(BaseModel):
    id: str
    title: str
    category: str # "positive", "negative", "mixed", "classic"
    review: str
    target_movie: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    model_loaded: bool
    keras_version: str
    architecture: str
