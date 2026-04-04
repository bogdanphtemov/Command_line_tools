import numpy as np 
from typing import Tuple 

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float, seed:int) -> Tuple[np.ndarray , np.ndarray , np.ndarray , np.ndarray]:

    """
    Random split with fixed seed for reproducibility.
    This is a standard practice in machine learning 
    that helps detect overfitting of the model.
    """
    if not(0.05 <= test_size <= 0.5):
        raise ValueError("!test_size should be between 0.05 and 0.5 for this tool!")
    
    # we take the number of examples
    n = X.shape[0]
    
    # creating a random number generator with a fixed code for reproducibility of results
    rng = np.random.RandomState(seed)
    
    # randomize indices
    idx = np.arange(n)
    rng.shuffle(idx)

    test_n = int(round(n * test_size))
    test_idx = idx[:test_n]

    train_idx = idx[test_n:]

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train , X_test , y_train , y_test

def standardize_fit(X_train: np.ndarray) -> Tuple[np.ndarray , np.ndarray , np.ndarray]:
    """
    Fit standardization on TRAIN only:
    mean, std per feature.
    Returns scaled X_train + mean + std.
    """

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # division by zero protection
    std_safe = np.where(std == 0.0 , 1.0, std)
    X_scaled = (X_train - mean) / std_safe

    return X_scaled , mean , std_safe

def standardize_apply(X: np.ndarray , mean: np.ndarray , std: np.ndarray) -> np.ndarray:
    return (X - mean) / std