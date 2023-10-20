import numpy as np
from sklearn.metrics import mean_absolute_error as MAE, mean_absolute_percentage_error as MAPE, mean_squared_error as MSE, r2_score

def RMSE(y_true, y_pred):
    return np.sqrt(MSE(y_true, y_pred))

def evaluate(y_true, y_pred):
    metrics = {}
    for metric in [MAE, MSE, MAPE, RMSE, r2_score]:
        metrics[metric.__name__] = metric(y_true, y_pred)
    return metrics