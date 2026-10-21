"""Test suite for KhushiDL Sentiment API."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "KhushiDL" in data["app_name"]

def test_samples():
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 4

def test_predict_positive():
    res = client.post("/api/predict", json={
        "review": "An extraordinary cinematic masterpiece with unforgettable acting and genius direction!"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["sentiment"] == "Positive"
    assert data["score"] > 0.5
    assert data["confidence"] > 50.0
    assert data["star_rating"] >= 3.0

def test_predict_negative():
    res = client.post("/api/predict", json={
        "review": "A horrible, boring, unwatchable waste of time. Terrible acting and awful script."
    })
    assert res.status_code == 200
    data = res.json()
    assert data["sentiment"] == "Negative"
    assert data["score"] < 0.5
    assert data["confidence"] > 50.0

def test_empty_review():
    res = client.post("/api/predict", json={"review": "   "})
    assert res.status_code == 400

def test_batch_predict():
    res = client.post("/api/predict/batch", json={
        "reviews": [
            "Loved every single minute of this fantastic masterpiece!",
            "Terrible, dreadful movie with horrific acting."
        ]
    })
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 2
    assert data["positive_count"] == 1
    assert data["negative_count"] == 1

if __name__ == "__main__":
    print("Running tests...")
    test_health()
    print("✓ Health check passed")
    test_samples()
    print("✓ Samples test passed")
    test_predict_positive()
    print("✓ Predict positive passed")
    test_predict_negative()
    print("✓ Predict negative passed")
    test_empty_review()
    print("✓ Empty review validation passed")
    test_batch_predict()
    print("✓ Batch predict passed")
    print("\nALL BACKEND TESTS PASSED SUCCESSFULLY! 🚀")
