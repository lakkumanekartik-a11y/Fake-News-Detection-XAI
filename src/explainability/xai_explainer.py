from typing import Any

import numpy as np
from lime.lime_text import LimeTextExplainer

from src.ml.predict import predict_news


explainer = LimeTextExplainer(
    class_names=["FAKE", "REAL"]
)


def explain_prediction(
    text: str,
    num_features: int = 10,
) -> dict[str, Any]:

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    text = text.strip()

    def predict_proba(texts: list[str]) -> np.ndarray:
        probabilities = []

        for item in texts:

            # LIME can occasionally create an empty variation.
            # Give it a neutral probability instead of calling
            # predict_news() with empty text.
            if not item or not item.strip():
                probabilities.append([0.5, 0.5])
                continue

            label, confidence = predict_news(item)

            confidence = float(confidence)

            if label == "FAKE":
                probabilities.append([
                    confidence,
                    1.0 - confidence,
                ])
            else:
                probabilities.append([
                    1.0 - confidence,
                    confidence,
                ])

        return np.asarray(
            probabilities,
            dtype=float,
        )

    explanation = explainer.explain_instance(
        text,
        predict_proba,
        num_features=num_features,
    )

    features = []

    for word, weight in explanation.as_list():
        features.append({
            "word": word,
            "weight": float(weight),
        })

    label, confidence = predict_news(text)

    return {
        "prediction": label,
        "confidence": float(confidence),
        "features": features,
    }