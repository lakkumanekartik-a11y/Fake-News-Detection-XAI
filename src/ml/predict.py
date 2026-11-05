from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_FILE = BASE_DIR / "models" / "fake_news_model.joblib"
VECTORIZER_FILE = BASE_DIR / "models" / "tfidf_vectorizer.joblib"


model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)


def predict_news(text):
    if not text or not text.strip():
        raise ValueError("News text cannot be empty.")

    features = vectorizer.transform([text])

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    confidence = float(max(probabilities))

    label = "REAL" if prediction == 1 else "FAKE"

    return label, confidence


if __name__ == "__main__":

    test_text = (
        "The government announced a new technology initiative "
        "to support artificial intelligence research."
    )

    label, confidence = predict_news(test_text)

    print("=" * 50)
    print("ML PREDICTION TEST")
    print("=" * 50)

    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.4f}")