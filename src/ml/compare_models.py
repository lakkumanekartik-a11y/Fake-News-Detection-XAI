import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


DATA_PATH = "data/processed/news_dataset.csv"
RANDOM_STATE = 42


def main():

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    print(f"Total samples: {len(df)}")

    X = df["combined_text"].fillna("")
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    print("\n" + "=" * 60)
    print("TF-IDF VECTORIZATION")
    print("=" * 60)

    vectorizer = TfidfVectorizer(
        max_features=100000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF features: {X_train_tfidf.shape[1]}")

    # Exactly 3 models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),

        "Linear SVM": LinearSVC(
            random_state=RANDOM_STATE,
        ),

        "Multinomial Naive Bayes": MultinomialNB(),
    }

    results = []

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    for name, model in models.items():

        print(f"\nTraining: {name}")

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
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        })

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False,
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1 Score": "{:.4f}".format,
            }
        )
    )

    best_model = results_df.iloc[0]

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(f"Model     : {best_model['Model']}")
    print(f"Accuracy  : {best_model['Accuracy']:.4f}")
    print(f"Precision : {best_model['Precision']:.4f}")
    print(f"Recall    : {best_model['Recall']:.4f}")
    print(f"F1 Score  : {best_model['F1 Score']:.4f}")


if __name__ == "__main__":
    main()