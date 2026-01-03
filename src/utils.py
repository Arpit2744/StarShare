import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    report = {}

    for model_name, model in models.items():
        params = param.get(model_name, {})

        gs = GridSearchCV(model, params, cv=3,n_jobs=-1)
        gs.fit(X_train, y_train)

        best_model = gs.best_estimator_
        y_pred = best_model.predict(X_test)

        score = r2_score(y_test, y_pred)
        report[model_name] = score

        models[model_name] = best_model  # IMPORTANT

    return report
