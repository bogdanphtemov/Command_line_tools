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

__all__ = [
    'train_test_split',
    'standardize_fit',
    'standardize_apply',
]
