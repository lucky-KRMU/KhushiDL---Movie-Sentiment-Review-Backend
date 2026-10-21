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
./venv/bin/pytest
```

---

## ☁️ Deploying to Render

This backend has been pre-configured and optimized specifically for **[Render](https://render.com/)**, including low-memory container tuning for Render's 512MB RAM free tier.

### Method 1: Automatic Blueprint (Recommended - One Click)
1. Push your changes to GitHub.
2. In the Render Dashboard, click **New +** -> **Blueprint**.
3. Connect your repository: `https://github.com/lucky-KRMU/KhushiDL---Movie-Sentiment-Review-Backend`.
4. Render will automatically detect [`render.yaml`](render.yaml) and configure:
   - **Runtime**: Python 3.12
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`
   - **Health Check Path**: `/health`
   - Pre-configured environment variables for TensorFlow CPU optimization.
5. Click **Apply**.

---

### Method 2: Manual Web Service Setup
If creating a Web Service manually on Render:
1. Click **New +** -> **Web Service**.
2. Connect your GitHub repository.
3. Configure the following settings:
   - **Language / Runtime**: `Python`
   - **Branch**: `main`
   - **Region**: Oregon (or closest to your users)
   - **Instance Type**: `Free`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`
4. In **Advanced Settings**:
   - **Health Check Path**: `/health`
   - **Environment Variables**:
     - `PYTHON_VERSION`: `3.12.8`
     - `TF_ENABLE_ONEDNN_OPTS`: `0`
     - `TF_CPP_MIN_LOG_LEVEL`: `2`
     - `KERAS_BACKEND`: `tensorflow`
     - `ALLOWED_ORIGINS`: `*` (or your frontend's Render/Vercel URL)
5. Click **Create Web Service**.

---

### Method 3: Docker Deployment
Render also supports building from [`Dockerfile`](Dockerfile):
1. Create a **Web Service** on Render and choose **Docker** runtime.
2. Render will automatically build the image using the lightweight `python:3.12-slim` base and start the container with internal health checks.

---

### ⚙️ Cloud Optimizations Applied:
- **`tensorflow-cpu` on Linux**: Uses lightweight CPU-only wheels on Render (reducing slug size by ~1.5GB and speeding up build times by >5x).
- **Single Worker (`--workers 1`)**: Restricts memory footprint to ~250MB, safely within Render's 512MB RAM free tier limit.
- **Thread Pool Limiting**: Configured `inter_op` (1) and `intra_op` (2) threads in TensorFlow to prevent memory spikes and CPU throttling.
- **Dynamic Port Binding**: Listens dynamically on `$PORT` assigned by Render's routing proxy.
- **Configurable CORS**: Supports `ALLOWED_ORIGINS` for cross-origin frontend requests.

