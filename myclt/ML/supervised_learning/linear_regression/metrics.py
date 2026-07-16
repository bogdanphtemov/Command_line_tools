import numpy as np 

# simple model performance testing functions
def mse(y_true: np.ndarray , y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true: np.ndarray , y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true , y_pred)))

def r2_score(y_true: np.ndarray , y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if ss_tot == 0.0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)