from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[2]

DATASET = BASE_DIR / "data" / "processed" / "news_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = MODEL_DIR / "fake_news_model.joblib"
VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATASET)

    X = df["combined_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    print("\nCreating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=100000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF features: {X_train_tfidf.shape[1]}")

    print("\nTraining Linear SVM...")

    svm_model = LinearSVC(
        random_state=42,
    )

    model = CalibratedClassifierCV(
        svm_model,
        method="sigmoid",
        cv=3,
    )

    model.fit(
        X_train_tfidf,
        y_train,
    )

    predictions = model.predict(
        X_test_tfidf
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
    )

    recall = recall_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
    )

    print("\n" + "=" * 50)
    print("CALIBRATED LINEAR SVM RESULTS")
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["FAKE", "REAL"],
        )
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE,
    )

    print("\nModels saved:")

    print(MODEL_FILE)
    print(VECTORIZER_FILE)


if __name__ == "__main__":
    main()