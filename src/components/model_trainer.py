import sys
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models


class ModelTrainer:

    def train(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"]
                },
                "Random Forest": {
                    "n_estimators": [64, 128, 256]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [64, 128]
                },
                "Linear Regression": {},
                "K-Neighbors": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [64, 128]
                },
                "CatBoost": {
                    "depth": [6, 8],
                    "learning_rate": [0.05, 0.1],
                    "iterations": [50, 100]
                },
                "AdaBoost": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [64, 128]
                }
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(f"Best model selected: {best_model_name}")

            return best_model, model_report

        except Exception as e:
            raise CustomException(e, sys)
