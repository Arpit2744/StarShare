import numpy as np

from src.train.fit import fit_models


def test_training_runs():
    # Fake but valid arrays
    X = np.random.rand(20, 5)
    y = np.random.rand(20)

    train_arr = np.c_[X[:15], y[:15]]
    test_arr = np.c_[X[15:], y[15:]]

    from src.train.model_defs import MODEL_DEFS, PARAM_GRID

    model, report = fit_models(
        train_arr,
        test_arr,
        MODEL_DEFS,
        PARAM_GRID
    )

    assert model is not None
    assert isinstance(report, dict)
    assert len(report) > 0
