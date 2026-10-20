"""
Preprocessing functions for KNN algorithm.

This module re-exports train_test_split, standardize_fit, and standardize_apply
from the shared base_models module to provide a consistent interface across
supervised models.

Note:
    This is a thin re-export layer. The actual implementation lives in
    myclt.ML.base_models.
"""

from ...base_models import train_test_split, standardize_fit, standardize_apply

__all__ = ['train_test_split', 'standardize_fit', 'standardize_apply']