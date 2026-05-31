"""
Logistic Regression module for binary classification.

Example usage:
    model = LogisticRegressionGD(learning_rate=0.01, epochs=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
"""

from .core import LogisticRegressionGD
from .data import Dataset, Prepareddata, load_csv_dataset
from .preprocessing import train_test_split, standardize_fit, standardize_apply
from .metrics import accuracy, precision, recall, f1_score, confusion_matrix
from .app_state import AppState, print_status, rebuild_split

__all__ = [
    'LogisticRegressionGD',
    'Dataset',
    'Prepareddata',
    'load_csv_dataset',
    'train_test_split',
    'standardize_fit',
    'standardize_apply',
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'confusion_matrix',
    'AppState',
    'print_status',
    'rebuild_split',
]
