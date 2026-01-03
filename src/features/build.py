# src/features/build.py

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.validation import validate_schema


def get_preprocessor(numerical_cols, categorical_cols):
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ("scaler", StandardScaler(with_mean=False))
        ]
    )

    return ColumnTransformer(
        [
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ]
    )


def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: dict
):
    """
    Deterministic feature construction.
    Assumes schema validity is enforced.
    """
    # 1. Enforce data contract (hard gate)
    validate_schema(train_df, schema)
    validate_schema(test_df, schema)

    target = schema["target"]
    num_cols = schema["numerical"]
    cat_cols = schema["categorical"]
    id_cols = schema.get("identifier", [])

    # 2. Split
    X_train = train_df.drop(columns=[target] + id_cols)
    y_train = train_df[target]

    X_test = test_df.drop(columns=[target] + id_cols)
    y_test = test_df[target]

    # 3. Transform
    preprocessor = get_preprocessor(num_cols, cat_cols)

    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    train_arr = np.c_[X_train_arr, y_train.to_numpy()]
    test_arr = np.c_[X_test_arr, y_test.to_numpy()]

    return train_arr, test_arr, preprocessor
