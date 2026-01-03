import numpy as np

from src.utils import evaluate_models
from src.logger import logging


def fit_models(train_arr, test_arr, model_defs, param_grid):
    """
    Fit candidate models and return the best model + metrics.
    No persistence. No thresholds. No side effects.
    """
    X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
    X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

    model_report = evaluate_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        models=model_defs,
        param=param_grid
    )

    best_model_name = max(model_report, key=model_report.get)
    best_model = model_defs[best_model_name]

    logging.info(f"Best model candidate: {best_model_name}")

    return best_model, model_report
