"""
Preprocessing utilities for SVM — reuses the universal base_models module.

Provides:
    - train_test_split()   : Split data into train/test sets
    - standardize_fit()    : Compute scaling parameters (mean, std)
    - standardize_apply()  : Apply standardization to data

All functions are re-exported from `myclt.ML.base_models` so that SVM
users can import them locally.

Example:
    >>> from myclt.ML.supervised_learning.svm.preprocessing import train_test_split, standardize_fit, standardize_apply
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> mean, std = standardize_fit(X_train)
    >>> X_train_scaled = standardize_apply(X_train, mean, std)
"""

from myclt.ML.base_models import (
    train_test_split,
    standardize_fit,
    standardize_apply,
)
import numpy as np
from typing import List, Tuple


def k_fold_split(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    K-Fold Cross-Validation split generator.
    Returns k tuples of (X_train, X_val, y_train, y_val)
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    
    fold_indices = np.array_split(idx, k)
    folds = []
    
    for fold_idx in range(k):
        val_idx = fold_indices[fold_idx]
        train_idx = np.concatenate([fold_indices[i] for i in range(k) if i != fold_idx])
        folds.append((X[train_idx], X[val_idx], y[train_idx], y[val_idx]))
    
    return folds


__all__ = [
    'train_test_split',
    'standardize_fit',
    'standardize_apply',
    'k_fold_split',
]
