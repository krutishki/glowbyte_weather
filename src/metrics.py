import numpy as np
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score


def RMSE(y_true, y_pred):
    return np.sqrt(MSE(y_true, y_pred))


def evaluate(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    metrics = {"n_observations": len(y_true)}
    for name, metric in {
        "MAE": MAE,
        "MSE": MSE,
        "MAPE": MAPE,
        "RMSE": RMSE,
        "R^2": r2_score,
    }.items():
        metrics[name] = metric(y_true, y_pred)
    return metrics
