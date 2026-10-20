"""
Supervised learning algorithms package.

This package contains various supervised machine learning algorithms
including K-Nearest Neighbors, Linear Regression, Logistic Regression, and SVM.
"""

# Import KNN algorithms
from .knn import KNNClassifier, KNNRegressor

__all__ = [
    'KNNClassifier',
    'KNNRegressor',
]
