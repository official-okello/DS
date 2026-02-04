from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    metrics = {
        "RMSE": root_mean_squared_error(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds),
    }

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics