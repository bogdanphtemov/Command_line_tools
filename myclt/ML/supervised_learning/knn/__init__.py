"""
K-Nearest Neighbors (KNN) module for classification and regression.

This module implements the K-Nearest Neighbors algorithm with support for:
- Classification with majority voting
- Regression with mean or weighted averaging
- Multiple distance metrics (Euclidean, Manhattan, Minkowski, Cosine)
- Cross-validation hyperparameter tuning
- Interactive command-line interface

Example usage:
    >>> from myclt.ML.supervised_learning.knn import KNNClassifier, KNNRegressor
    >>> classifier = KNNClassifier(k=5, distance_metric='euclidean')
    >>> regressor = KNNRegressor(k=3, distance_metric='manhattan', weighted=True)

See also:
- knn.core: Core KNN algorithm implementation
- knn.metrics: Evaluation metrics
- knn.distance_metrics: Distance calculation functions
- knn.ui: Interactive interface
- knn.cli: Command-line interface
"""

from myclt.ML.supervised_learning.knn.classifier import KNNClassifier
from myclt.ML.supervised_learning.knn.regressor import KNNRegressor
from myclt.ML.supervised_learning.knn.distance_metrics import (
    euclidean_distance, 
    manhattan_distance, 
    minkowski_distance,
    cosine_distance,
    get_distance_metric
)
from myclt.ML.supervised_learning.knn import distance_metrics

__all__ = [
    'KNNClassifier',
    'KNNRegressor',
    'euclidean_distance',
    'manhattan_distance',
    'minkowski_distance',
    'cosine_distance',
    'get_distance_metric',
    'distance_metrics'
]