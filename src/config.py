from pathlib import Path
import yaml

SEED = 42

DATA_DIR = Path("data")
RAW_DATA_PATH = DATA_DIR / "raw" / "data.csv"
SCHEMA_PATH = DATA_DIR / "schema.yaml"   

PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = Path("models")

TRAIN_RATIO = 0.8
MIN_ACCEPTABLE_SCORE = 0.6


def load_schema():
    with open(SCHEMA_PATH, "r") as f:
        return yaml.safe_load(f)
