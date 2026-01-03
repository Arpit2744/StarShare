import pandas as pd
import numpy as np

from src.features.build import build_features


SCHEMA = {
    "numerical": ["num"],
    "categorical": ["cat"],
    "target": "y"
}


def test_feature_shapes_match():
    df = pd.DataFrame({
        "num": [1, 2, 3, 4],
        "cat": ["a", "b", "a", "b"],
        "y": [10, 20, 30, 40]
    })

    train_df = df.iloc[:3]
    test_df = df.iloc[3:]

    train_arr, test_arr, _ = build_features(train_df, test_df, SCHEMA)

    assert train_arr.shape[1] == test_arr.shape[1]
    assert train_arr.shape[0] == 3
    assert test_arr.shape[0] == 1
