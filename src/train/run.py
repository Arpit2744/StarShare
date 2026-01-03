import random
import numpy as np
import pandas as pd

from src.config import (
    SEED,
    RAW_DATA_PATH,
    TRAIN_RATIO,
    MIN_ACCEPTABLE_SCORE,
    MODEL_DIR,
    load_schema,
)
from src.features import build_features
from src.train import fit_models, evaluate_model
from src.model import save_model
from src.logger import logging


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def run():
    logging.info("Starting ML system run")

    # 1. Reproducibility
    set_seed(SEED)

    # 2. Preconditions
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing raw data: {RAW_DATA_PATH}")

    # 3. Load inputs
    schema = load_schema()
    df = pd.read_csv(RAW_DATA_PATH)

    # 4. Split
    train_df = df.sample(frac=TRAIN_RATIO, random_state=SEED)
    test_df = df.drop(train_df.index)

    # 5. Features
    train_arr, test_arr, preprocessor = build_features(
        train_df, test_df, schema
    )

    # 6. Model fitting
    from src.train.model_defs import MODEL_DEFS, PARAM_GRID
    model, report = fit_models(train_arr, test_arr, MODEL_DEFS, PARAM_GRID)

    # 7. Evaluation gate
    best_score = evaluate_model(report, MIN_ACCEPTABLE_SCORE)
    logging.info(f"Model accepted with score {best_score:.4f}")

    # 8. Persist
    version_dir = save_model(
        model=model,
        preprocessor=preprocessor,
        model_dir=MODEL_DIR,
    )

    logging.info(f"Model version saved: {version_dir}")
    logging.info("System run completed successfully")


if __name__ == "__main__":
    run()
