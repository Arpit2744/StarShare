import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging


class DataTransformation:

    def validate_schema(self, df: pd.DataFrame, schema: dict):
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

    def get_preprocessor(self, numerical_cols, categorical_cols):
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

    def transform(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        schema: dict
    ):
        try:
            # 1. Validate
            self.validate_schema(train_df, schema)
            self.validate_schema(test_df, schema)

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
            preprocessor = self.get_preprocessor(num_cols, cat_cols)

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_arr, y_train.to_numpy()]
            test_arr = np.c_[X_test_arr, y_test.to_numpy()]

            return train_arr, test_arr, preprocessor

        except Exception as e:
            raise CustomException(e, sys)
