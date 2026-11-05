from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"

RANDOM_STATE = 42

ML_TEST_SIZE = 0.2

MIN_EVIDENCE_SCORE = 0.60
MIN_NLI_CONFIDENCE = 0.70

TOP_K_EVIDENCE = 5