from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

FAKE_FILE = BASE_DIR / "data" / "raw" / "Fake.csv"
REAL_FILE = BASE_DIR / "data" / "raw" / "True.csv"

OUTPUT_FILE = BASE_DIR / "data" / "processed" / "news_dataset.csv"


def load_dataset():
    fake = pd.read_csv(FAKE_FILE)
    real = pd.read_csv(REAL_FILE)

    fake["label"] = 0
    real["label"] = 1

    fake["text"] = (
        fake["title"].fillna("") + " " + fake["text"].fillna("")
    )

    real["text"] = (
        real["title"].fillna("") + " " + real["text"].fillna("")
    )

    fake = fake[["title", "text", "label"]]
    real = real[["title", "text", "label"]]

    data = pd.concat([fake, real], ignore_index=True)

    data["title"] = data["title"].fillna("").astype(str)
    data["text"] = data["text"].fillna("").astype(str)

    data["combined_text"] = (
        data["title"] + " " + data["text"]
    ).str.strip()

    data = data[data["combined_text"].str.len() > 20]

    data = data.drop_duplicates(
        subset=["combined_text"]
    ).reset_index(drop=True)

    return data


def save_dataset(data):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )


if __name__ == "__main__":
    data = load_dataset()

    save_dataset(data)

    print("=" * 50)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 50)

    print(f"Total samples : {len(data)}")
    print(f"Fake samples  : {(data['label'] == 0).sum()}")
    print(f"Real samples  : {(data['label'] == 1).sum()}")

    print("\nColumns:")
    print(data.columns.tolist())

    print("\nOutput:")
    print(OUTPUT_FILE)