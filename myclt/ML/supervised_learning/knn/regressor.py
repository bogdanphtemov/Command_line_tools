"""
K-Nearest Neighbors Regressor implementation.
"""

import numpy as np
from .core import KNNBase


class KNNRegressor(KNNBase):
    """K-Nearest Neighbors Regressor."""
    
    model_type = "knn_regressor"
    
    def __init__(self, k=5, distance_metric='euclidean', weighted=False, p=2):
        """
        Initialize KNN regressor.
        
        Args:
            k: Number of neighbors to consider
            distance_metric: Distance metric to use ('euclidean', 'manhattan', 'minkowski', 'cosine')
            weighted: Whether to use weighted average based on distance
            p: Power parameter for Minkowski distance (default 2 for Euclidean)
        """
        super().__init__(k, distance_metric, p)
        self.weighted = weighted
    
    def _make_prediction(self, k_nearest):
        """
        Make regression prediction using average or weighted average of neighbors.
        
        Args:
            k_nearest: List of (distance, value) tuples for k nearest neighbors
            
        Returns:
            Average or weighted average value among k nearest neighbors
        """
        if not self.weighted:
            # Simple average
            values = [value for _, value in k_nearest]
            return np.mean(values)
        else:
            # Weighted average using inverse distance
            distances = np.array([dist for dist, _ in k_nearest])
            values = np.array([value for _, value in k_nearest])
            
            # Handle zero distances (exact match)
            if np.any(distances == 0):
                # Return value of exact match directly
                zero_idx = np.where(distances == 0)[0]
                return values[zero_idx[0]]
            
            # Use inverse distance weighting
            weights = 1 / distances
            return np.average(values, weights=weights)
    
    def predict(self, X_test):
        """
        Predict values for test data.
        
        Args:
            X_test: Test features (numpy array or list of lists)
            
        Returns:
            numpy array: Predicted values
        """
        return super().predict(X_test)
    
    def get_params(self):
        """Get all model parameters for saving."""
        params = super().get_params()
        params['weighted'] = self.weighted
        return params
    
    def set_params(self, params):
        """Set model parameters from loaded data."""
        super().set_params(params)
        self.weighted = params.get('weighted', False)