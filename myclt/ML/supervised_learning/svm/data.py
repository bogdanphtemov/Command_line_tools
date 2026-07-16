"""
Data loading utilities for SVM — reuses the universal base_data module.

Provides:
    - Dataset         : Universal data container (imported from base_data)
    - Prepareddata    : Selected features/target (imported from base_data)
    - load_csv_dataset()   : CSV loading with auto-delimiter detection
    - manual_input_dataset() : Manual data entry
    - select_features_and_target() : Interactive feature/target selection

All functions are re-exported from `myclt.ML.base.base_data` so that
SVM users can import them from `myclt.ML.supervised_learning.svm.data`.

Example:
    >>> from myclt.ML.supervised_learning.svm.data import load_csv_dataset, Dataset
    >>> ds = load_csv_dataset("data.csv")
    >>> print(ds.columns)
"""

from myclt.ML.base.base_data import (
    Dataset,
    Prepareddata,
    load_csv_dataset,
    manual_input_dataset,
    select_features_and_target,
)

__all__ = [
    'Dataset',
    'Prepareddata',
    'load_csv_dataset',
    'manual_input_dataset',
    'select_features_and_target',
]
