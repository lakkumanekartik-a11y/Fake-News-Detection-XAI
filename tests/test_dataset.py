from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DATASET = BASE_DIR / "data" / "processed" / "news_dataset.csv"


def test_dataset_exists():
    assert DATASET.exists()


def test_dataset_structure():
    df = pd.read_csv(DATASET)

    required_columns = {
        "title",
        "text",
        "label",
        "combined_text",
    }

    assert required_columns.issubset(df.columns)


def test_dataset_not_empty():
    df = pd.read_csv(DATASET)

    assert len(df) > 1000


def test_labels_are_valid():
    df = pd.read_csv(DATASET)

    assert set(df["label"].unique()) == {0, 1}


def test_no_duplicate_text():
    df = pd.read_csv(DATASET)

    assert df["combined_text"].duplicated().sum() == 0


def test_no_empty_text():
    df = pd.read_csv(DATASET)

    assert (df["combined_text"].str.strip().str.len() > 0).all()