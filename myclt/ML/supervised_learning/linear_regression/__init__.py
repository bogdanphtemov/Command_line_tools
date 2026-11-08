"""
Linear Regression module for forecasting continuous values.

Provides a from-scratch implementation of Linear Regression trained with
batch Gradient Descent (optional L1/L2 regularization), together with data
helpers, preprocessing, evaluation metrics, hyperparameter tuning and an interactive CLI.

Example usage:
    from myclt.ML.supervised_learning.linear_regression import LinearRegressionGD

    model = LinearRegressionGD(learning_rate=0.05, epochs=2000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

Interactive CLI:
    python -m myclt                       # then choose the Linear Regression item
    python -m myclt.ML.supervised_learning.linear_regression.cli
"""

from .core import LinearRegressionGD
from .data import (
    Dataset, Prepareddata, load_csv_dataset,
    manual_input_dataset, select_features_and_target,
)
from .preprocessing import train_test_split, standardize_fit, standardize_apply
from .metrics import mse, rmse, r2_score
from .app_state import AppState, print_status, rebuild_split
from .hyperparameter_tuning import grid_search_regularization, auto_tune_learning_rate
from .cli import main

__all__ = [
    # Core model
    'LinearRegressionGD',
    # Data
    'Dataset',
    'Prepareddata',
    'load_csv_dataset',
    'manual_input_dataset',
    'select_features_and_target',
    # Preprocessing
    'train_test_split',
    'standardize_fit',
    'standardize_apply',
    # Metrics
    'mse',
    'rmse',
    'r2_score',
    # Hyperparameter tuning
    'grid_search_regularization',
    'auto_tune_learning_rate',
    # Session state
    'AppState',
    'print_status',
    'rebuild_split',
    # CLI entry point
    'main',
]