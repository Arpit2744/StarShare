import pandas as pd
import pytest

from src.validation.checks import validate_schema


SCHEMA = {
    "numerical": ["a"],
    "categorical": ["b"],
    "target": "y"
}


def test_schema_passes_on_valid_data():
    df = pd.DataFrame({
        "a": [1, 2],
        "b": ["x", "y"],
        "y": [0, 1]
    })

    # Should not raise
    validate_schema(df, SCHEMA)


def test_schema_fails_on_missing_column():
    df = pd.DataFrame({
        "a": [1, 2],
        "y": [0, 1]
    })

    with pytest.raises(ValueError):
        validate_schema(df, SCHEMA)
