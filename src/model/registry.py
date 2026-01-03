# src/model/registry.py

from pathlib import Path
import pickle
from datetime import datetime

from src.logger import logging


def _timestamp():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def save_model(model, preprocessor, model_dir: Path):
    """
    Persist model and preprocessor.
    Versioned by timestamp.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    version = _timestamp()
    version_dir = model_dir / version
    version_dir.mkdir()

    model_path = version_dir / "model.pkl"
    preprocessor_path = version_dir / "preprocessor.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)

    logging.info(f"Model saved at {version_dir}")

    return version_dir


def load_model(version_dir: Path):
    """
    Load model + preprocessor from a given version directory.
    """
    model_path = version_dir / "model.pkl"
    preprocessor_path = version_dir / "preprocessor.pkl"

    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError("Model artifacts missing")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor
