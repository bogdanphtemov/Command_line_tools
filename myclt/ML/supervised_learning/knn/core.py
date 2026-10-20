"""
Core implementation of K-Nearest Neighbors algorithm.
"""

import numpy as np
from typing import Dict, Any
from myclt.ML.base_models import BaseModel, SupervisedModel
from .distance_metrics import get_distance_metric


class KNNBase(BaseModel, SupervisedModel):
    """Base class for KNN algorithms."""
    
    # Model type for session management
    model_type = "knn"
    
    def __init__(self, k=5, distance_metric='euclidean', p=2):
        """
        Initialize KNN model.
        
        Args:
            k: Number of neighbors to consider
            distance_metric: Distance metric to use ('euclidean', 'manhattan', 'minkowski')
            p: Power parameter for Minkowski distance (default 2 for Euclidean)
        """
        self.k = k
        self.distance_metric_name = distance_metric
        self.p = p
        
        # Get distance function with p parameter for Minkowski
        if distance_metric == 'minkowski':
            # Create partial function with p parameter
            from functools import partial
            base_metric = get_distance_metric(distance_metric)
            self.distance_metric_func = partial(base_metric, p=p)
        else:
            self.distance_metric_func = get_distance_metric(distance_metric)
            
        self.X_train = None
        self.y_train = None
    
    @property
    def is_trained(self) -> bool:
        """Returns True if model has been trained."""
        return self.X_train is not None and self.y_train is not None
    
    @property
    def distance_metric(self) -> str:
        """Returns the name of the distance metric."""
        return self.distance_metric_name
    
    def get_params(self) -> Dict[str, Any]:
        """Get all model parameters for saving."""
        if not self.is_trained:
            return {
                'k': self.k,
                'distance_metric': self.distance_metric_name,
                'p': self.p,
                'trained': False
            }
        
        return {
            'k': self.k,
            'distance_metric': self.distance_metric_name,
            'p': self.p,
            'X_train': self.X_train.tolist() if self.X_train is not None else None,
            'y_train': self.y_train.tolist() if self.y_train is not None else None,
            'trained': True
        }
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from loaded data."""
        self.k = params.get('k', 5)
        
        distance_metric = params.get('distance_metric', 'euclidean')
        self.distance_metric_name = distance_metric
        self.p = params.get('p', 2)
        
        # Recreate distance function with p parameter
        if distance_metric == 'minkowski':
            from functools import partial
            base_metric = get_distance_metric(distance_metric)
            self.distance_metric_func = partial(base_metric, p=self.p)
        else:
            self.distance_metric_func = get_distance_metric(distance_metric)
        
        if params.get('trained', False):
            X_train_data = params.get('X_train')
            y_train_data = params.get('y_train')
            
            if X_train_data is not None and y_train_data is not None:
                self.X_train = np.array(X_train_data)
                self.y_train = np.array(y_train_data)
        else:
            self.X_train = None
            self.y_train = None
    
    def fit(self, X, y):
        """
        Store training data.
        
        Args:
            X: Training features (numpy array or list of lists)
            y: Training labels/values (numpy array or list)
            
        Raises:
            ValueError: If inputs have incompatible shapes or empty dataset
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Basic validation
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length, got {len(X)} and {len(y)}")
        if len(X) == 0:
            raise ValueError("Cannot fit on empty dataset")
        if self.k > len(X):
            raise ValueError(f"k={self.k} cannot be greater than number of samples={len(X)}")
        
        self.X_train = X
        self.y_train = y
    
    def _get_k_nearest_neighbors(self, test_point):
        """
        Find k nearest neighbors for a test point.
        
        Args:
            test_point: Point to classify (list or numpy array)
            
        Returns:
            List of (distance, label) tuples for k nearest neighbors,
            sorted by distance in ascending order (closest first).
            
        Note:
            The returned list is guaranteed to be sorted by distance, which is
            essential for tie-breaking in classification.
        """
        if not self.is_trained:
            raise ValueError("Model must be fitted before prediction")
        
        test_point = np.array(test_point)
        k = min(self.k, len(self.X_train))  # Handle case when k > number of samples
        
        # Check feature dimensions match
        if test_point.shape[-1] != self.X_train.shape[1]:
            raise ValueError(f"Expected {self.X_train.shape[1]} features, got {test_point.shape[-1]}")
        
        # Vectorized implementation for supported metrics
        if self.distance_metric_name in ['euclidean', 'manhattan', 'minkowski']:
            # Vectorized distance calculation using NumPy broadcasting
            if self.distance_metric_name == 'euclidean':
                distances_all = np.sqrt(np.sum((self.X_train - test_point) ** 2, axis=1))
            elif self.distance_metric_name == 'manhattan':
                distances_all = np.sum(np.abs(self.X_train - test_point), axis=1)
            elif self.distance_metric_name == 'minkowski':
                # Use distance metric function for consistency
                distances_all = np.array([self.distance_metric_func(test_point, train_point) 
                                        for train_point in self.X_train])
            
            # Safe argpartition - handle case when k equals or exceeds array length
            if k >= len(distances_all):
                # All points are neighbors - no need to partition
                k_indices_sorted = np.argsort(distances_all)
            else:
                # Use argpartition for O(N) selection of k smallest distances
                k_smallest_indices = np.argpartition(distances_all, k-1)[:k]
                # Sort just the k smallest elements for correct ordering
                k_indices_sorted = k_smallest_indices[np.argsort(distances_all[k_smallest_indices])]
            
            k_distances = distances_all[k_indices_sorted]
            k_labels = self.y_train[k_indices_sorted]
            
            k_nearest = list(zip(k_distances, k_labels))
        else:
            # Fallback to iterative method for unsupported metrics (e.g., cosine)
            # Calculate distances to all training points
            distances = []
            for i, train_point in enumerate(self.X_train):
                dist = self.distance_metric_func(test_point, train_point)
                distances.append((dist, self.y_train[i]))
            
            # Sort by distance and get k nearest neighbors
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:k]
        
        return k_nearest
    
    def _get_k_nearest_indices(self, test_point):
        """Get indices of k nearest neighbors for visualization."""
        k_nearest = self._get_k_nearest_neighbors(test_point)
        if not k_nearest:
            return np.array([])
        
        # Find indices of k nearest neighbors
        k_indices = []
        test_point = np.array(test_point)
        for distance, label in k_nearest:
            # Find the original training point with matching distance and label
            for i in range(len(self.X_train)):
                try:
                    if (np.linalg.norm(self.X_train[i] - test_point) <= distance + 1e-8 and 
                        self.y_train[i] == label):
                        k_indices.append(i)
                        break
                except:
                    continue
        
        return np.array(k_indices)
    
    def predict(self, X_test):
        """
        Predict labels/values for test data.
        
        Args:
            X_test: Test features (numpy array or list of lists)
            
        Returns:
            numpy array: Predicted labels/values
        """
        X_test = np.array(X_test)
        
        # Handle single test point
        if len(X_test.shape) == 1:
            X_test = X_test.reshape(1, -1)
        
        predictions = []
        for test_point in X_test:
            k_nearest = self._get_k_nearest_neighbors(test_point)
            pred = self._make_prediction(k_nearest)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _make_prediction(self, k_nearest):
        """Make prediction based on k nearest neighbors. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _make_prediction method")