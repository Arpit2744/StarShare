# src/train_system.py

from pathlib import Path
import pandas as pd
import random
import numpy as np
import os

from src.schema import FEATURE_SCHEMA
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import save_object
from src.logger import logging

SEED = 42
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
RAW_DATA_PATH = Path("data/raw/data.csv")  


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(SEED)
def main():
    logging.info("Starting training system")

    # 1. Preconditions
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data missing: {RAW_DATA_PATH}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load data
    df = pd.read_csv(RAW_DATA_PATH)

    # 3. Split
    train_df = df.sample(frac=0.8, random_state=SEED)
    test_df = df.drop(train_df.index)

    # 4. Transform
    transformer = DataTransformation()
    train_arr, test_arr, preprocessor = transformer.transform(
        train_df,
        test_df,
        FEATURE_SCHEMA
    )

    # 5. Train
    trainer = ModelTrainer()
    model, report = trainer.train(train_arr, test_arr)

    best_score = max(report.values())
    logging.info(f"Best model score: {best_score}")

    # 6. Gate
    if best_score < 0.6:
        raise RuntimeError("Model rejected by quality gate")

    # 7. Persist
    save_object(MODEL_PATH, model)
    save_object(PREPROCESSOR_PATH, preprocessor)

    logging.info("Training completed successfully")


if __name__ == "__main__":
    main()

