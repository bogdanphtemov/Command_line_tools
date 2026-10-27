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
        if not isinstance(weighted, bool):
            raise ValueError(f"weighted must be boolean, got {type(weighted).__name__}")
        self.weighted = weighted
    
    def fit(self, X, y):
        """
        Fit the model to training data.
        
        Args:
            X: Training features (numpy array or list of lists)
            y: Training values (numpy array or list)
            
        Returns:
            self: For method chaining
        """
        # Convert y to float for regression to ensure numeric values
        y = np.asarray(y, dtype=float)
        return super().fit(X, y)
    
    def _make_prediction_from_neighbors(self, distances, values):
        """
        Make regression prediction using average or weighted average of neighbors.
        
        Args:
            distances: Sorted array of distances (ascending) from k nearest neighbors
            values: Corresponding values from k nearest neighbors
            
        Returns:
            Average or weighted average value among k nearest neighbors
        """
        if len(values) == 0:
            # Return mean of training data as fallback
            return np.mean(self.y_train) if len(self.y_train) > 0 else 0
            
        if not self.weighted:
            # Simple average
            return np.mean(values)
        else:
            # Weighted average using inverse distance
            distances_arr = np.array(distances)
            values_arr = np.array(values)
            
            # Handle zero distances (exact match) using safe floating-point comparison
            eps = np.finfo(float).eps * 10
            zero_mask = np.isclose(distances_arr, 0, atol=eps)
            if np.any(zero_mask):
                # Return average of all exact matches (not just the first one)
                return np.mean(values_arr[zero_mask])
            
            # Use inverse distance weighting with epsilon for stability
            weights = 1.0 / np.maximum(distances_arr, eps)
            return np.average(values_arr, weights=weights)
    
    def get_params(self):
        """Get all model parameters for saving."""
        params = super().get_params()
        params['weighted'] = self.weighted
        return params
    
    def set_params(self, params):
        """Set model parameters from loaded data."""
        super().set_params(params)
        weighted = params.get('weighted', False)
        if not isinstance(weighted, bool):
            raise ValueError(f"weighted must be boolean, got {type(weighted).__name__}")
        self.weighted = weighted