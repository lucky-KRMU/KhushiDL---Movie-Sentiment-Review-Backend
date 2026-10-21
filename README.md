# KhushiDL - Movie Sentiment Review Backend

High-performance deep learning REST API for movie review sentiment classification, built with **FastAPI**, **Keras 3**, and **TensorFlow**.

---

## 🧠 Neural Network Architecture

The backend serves an end-to-end Keras 3 neural network trained for binary sentiment classification:

1. **Input Layer**: Accepts raw string text reviews.
2. **TextVectorization Layer**: Standardizes text (`lower_and_strip_punctuation`), tokenizes via whitespace, max vocab size of 10,000 tokens, fixed output sequence length of 250 tokens.
3. **Embedding Layer**: 64-dimensional learned word representations (`input_dim=10001, output_dim=64, mask_zero=True`).
4. **Spatial Dropout**: 0.5 rate for regularization.
5. **1D Convolution (Conv1D)**: 64 filters, kernel size 5, stride 2, ReLU activation.
6. **Global Average Pooling (1D)**: Temporal feature reduction.
7. **Dense Output Layer**: Single sigmoid neuron outputting probability $P \in [0.0, 1.0]$.
   - $P \ge 0.50 \implies$ **Positive**
   - $P < 0.50 \implies$ **Negative**

---

## ⚡ API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check, model status, and architecture info |
| `POST` | `/api/predict` | Predict sentiment for a single review |
| `POST` | `/api/predict/batch` | Batch sentiment analysis for multiple reviews |
| `GET` | `/api/samples` | Curated sample movie reviews for quick testing |
| `GET` | `/api/model-info` | Detailed layer-by-layer neural network parameters |
| `GET` | `/docs` | Interactive Swagger API documentation |

### Example Request (`POST /api/predict`):
```json
{
  "review": "An absolute masterpiece! Stunning cinematography, brilliant direction, and unforgettable performances."
}
```

### Example Response:
```json
{
  "review": "An absolute masterpiece! Stunning cinematography, brilliant direction, and unforgettable performances.",
  "sentiment": "Positive",
  "score": 0.9412,
  "confidence": 94.12,
  "intensity": "Overwhelmingly Positive",
  "star_rating": 4.8,
  "word_count": 11,
  "inference_time_ms": 1.95,
  "detected_cues": {
    "positive": ["masterpiece", "brilliant", "unforgettable", "stunning"],
    "negative": []
  }
}
```

---

## 🚀 Quick Start

### 1. Installation & Running
```bash
./run.sh
```
Or manually:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Run Test Suite
```bash
./venv/bin/python test_api.py
```
