import os
import re
import time
import logging
from typing import List, Dict, Any, Tuple

# Optimize TensorFlow & Keras for cloud deployment (e.g. Render 512MB RAM free tier)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress non-critical logs
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import tensorflow as tf
import keras

# Constrain thread pools to prevent memory ballooning and CPU contention on container runtimes
try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)
except Exception:
    pass

logger = logging.getLogger("khushidl.model")

# Lexicons for highlight tokens
POSITIVE_KEYWORDS = {
    "masterpiece", "triumph", "brilliant", "mesmerizing", "magnificent",
    "breathtaking", "magic", "unforgettable", "gem", "poetic", "authentic",
    "touching", "great", "excellent", "superb", "stellar", "phenomenal",
    "loved", "love", "best", "stunning", "captivating", "entertaining",
    "impressive", "hilarious", "heartwarming", "flawless", "delightful",
    "terrific", "wonderful", "sublime", "classic", "perfection", "genius",
    "compelling", "gripping", "charismatic", "cinematic", "masterful"
}

NEGATIVE_KEYWORDS = {
    "dreadful", "disaster", "cringe", "flatter", "wasted", "horrible",
    "cheap", "pathetic", "garbage", "trash", "awful", "terrible", "worst",
    "hate", "hated", "boring", "boredom", "cliché", "cliches", "dull",
    "pointless", "unwatchable", "cheesy", "disappointed", "disappointing",
    "predictable", "annoying", "senseless", "mess", "clunky", "atrocious",
    "lame", "abysmal", "stupid", "flop", "slog", "tedious", "waste", "bad"
}

class SentimentModelService:
    def __init__(self, model_path: str = None, eager_load: bool = False):
        if model_path is None:
            # Locate relative to backend root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.getenv("MODEL_PATH", os.path.join(base_dir, "model.keras"))
        
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        if eager_load:
            self.load_model()

    def ensure_loaded(self):
        """Ensures the model is loaded before inference."""
        if not self.is_loaded or self.model is None:
            self.load_model()

    def load_model(self):
        try:
            logger.info(f"Loading Keras model from {self.model_path}...")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")
            
            # Load model
            self.model = keras.models.load_model(self.model_path)
            self.is_loaded = True
            logger.info("Model loaded successfully. Performing warm-up...")
            
            # Warm-up inference
            _ = self.model(tf.constant(["Warmup review for initialization"]))
            logger.info("Warm-up complete. Model is ready for serving.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_loaded = False
            raise

    def extract_cues(self, text: str) -> Dict[str, List[str]]:
        # Tokenize words cleanly
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        found_pos = sorted(list({w for w in words if w in POSITIVE_KEYWORDS}))
        found_neg = sorted(list({w for w in words if w in NEGATIVE_KEYWORDS}))
        return {
            "positive": found_pos,
            "negative": found_neg
        }

    def compute_intensity(self, score: float) -> str:
        if score >= 0.90:
            return "Overwhelmingly Positive"
        elif score >= 0.75:
            return "Strongly Positive"
        elif score >= 0.60:
            return "Moderately Positive"
        elif score >= 0.50:
            return "Leaning Positive"
        elif score >= 0.40:
            return "Leaning Negative"
        elif score >= 0.25:
            return "Moderately Negative"
        elif score >= 0.10:
            return "Strongly Negative"
        else:
            return "Overwhelmingly Negative"

    def predict_single(self, review: str) -> Dict[str, Any]:
        self.ensure_loaded()

        clean_review = review.strip()
        start_time = time.perf_counter()

        # Execute model forward pass
        tensor_input = tf.constant([clean_review], dtype=tf.string)
        output = self.model(tensor_input)
        raw_score = float(output.numpy()[0][0])

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        is_positive = raw_score >= 0.5
        sentiment_label = "Positive" if is_positive else "Negative"
        confidence_pct = (raw_score if is_positive else (1.0 - raw_score)) * 100.0
        star_rating = round(raw_score * 4.0 + 1.0, 1)

        cues = self.extract_cues(clean_review)

        return {
            "review": clean_review,
            "sentiment": sentiment_label,
            "score": round(raw_score, 4),
            "confidence": round(confidence_pct, 2),
            "intensity": self.compute_intensity(raw_score),
            "star_rating": star_rating,
            "word_count": len(clean_review.split()),
            "inference_time_ms": round(elapsed_ms, 2),
            "detected_cues": cues
        }

    def predict_batch(self, reviews: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        self.ensure_loaded()

        clean_reviews = [r.strip() for r in reviews if r.strip()]
        if not clean_reviews:
            return [], {"total": 0, "positive_count": 0, "negative_count": 0, "average_score": 0.0}

        start_time = time.perf_counter()
        tensor_input = tf.constant(clean_reviews, dtype=tf.string)
        outputs = self.model(tensor_input).numpy()
        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        avg_per_item_ms = total_time_ms / len(clean_reviews)

        results = []
        pos_count = 0
        total_score = 0.0

        for r, out in zip(clean_reviews, outputs):
            raw_score = float(out[0])
            total_score += raw_score
            is_pos = raw_score >= 0.5
            if is_pos:
                pos_count += 1
            conf = (raw_score if is_pos else (1.0 - raw_score)) * 100.0

            results.append({
                "review": r,
                "sentiment": "Positive" if is_pos else "Negative",
                "score": round(raw_score, 4),
                "confidence": round(conf, 2),
                "intensity": self.compute_intensity(raw_score),
                "star_rating": round(raw_score * 4.0 + 1.0, 1),
                "word_count": len(r.split()),
                "inference_time_ms": round(avg_per_item_ms, 2),
                "detected_cues": self.extract_cues(r)
            })

        summary = {
            "total": len(results),
            "positive_count": pos_count,
            "negative_count": len(results) - pos_count,
            "average_score": round(total_score / len(results), 4)
        }
        return results, summary


# Global singleton instance (loads during FastAPI lifespan or on first call)
model_service = SentimentModelService(eager_load=False)
