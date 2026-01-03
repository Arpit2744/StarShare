def evaluate_model(model_report, min_score):
    """
    Enforce quality gate.
    Raises if model is unacceptable.
    """
    best_score = max(model_report.values())

    if best_score < min_score:
        raise RuntimeError(
            f"Model rejected: best score {best_score:.4f} < {min_score}"
        )

    return best_score
