from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT/"data"
RAW_DATA = DATA_DIR/"raw"
PROCESSED_DATA = DATA_DIR/"processed"

ARTIFACTS_DIR = PROJECT_ROOT/"artifacts"
MODEL_PATH = ARTIFACTS_DIR/"model.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2
