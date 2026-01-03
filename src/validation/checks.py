# src/validation/checks.py

import pandas as pd
from src.logger import logging


def validate_schema(df: pd.DataFrame, schema: dict):
    """
    Enforces data contract.
    Fails fast on violations.
    """
    expected = (
        set(schema["numerical"])
        | set(schema["categorical"])
        | set(schema.get("identifier", []))
        | {schema["target"]}
    )

    missing = expected - set(df.columns)
    extra = set(df.columns) - expected

    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    if extra:
        logging.warning(f"Extra columns ignored: {extra}")
