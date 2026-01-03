from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


MODEL_DEFS = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "LinearRegression": LinearRegression(),
    "KNN": KNeighborsRegressor(),
    "XGB": XGBRegressor(random_state=42),
    "CatBoost": CatBoostRegressor(verbose=False, random_seed=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
}

PARAM_GRID = {
    "RandomForest": {"n_estimators": [64, 128]},
    "DecisionTree": {"criterion": ["squared_error", "friedman_mse"]},
    "GradientBoosting": {"learning_rate": [0.05, 0.1]},
    "LinearRegression": {},
    "KNN": {},
    "XGB": {"n_estimators": [64, 128]},
    "CatBoost": {"depth": [6, 8]},
    "AdaBoost": {"n_estimators": [64, 128]},
}
