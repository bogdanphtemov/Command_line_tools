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
            k: Number of neighbors to consider (must be positive)
            distance_metric: Distance metric to use ('euclidean', 'manhattan', 'minkowski')
            p: Power parameter for Minkowski distance (default 2 for Euclidean)
            
        Raises:
            ValueError: If k is not positive
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
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
        self.n_features_in_ = None
    
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
            'X_train_dtype': str(self.X_train.dtype) if self.X_train is not None else 'float64',
            'y_train': self.y_train.tolist() if self.y_train is not None else None,
            'y_train_dtype': str(self.y_train.dtype) if self.y_train is not None else None,
            'n_features_in_': getattr(self, 'n_features_in_', None),
            'trained': True
        }
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from loaded data."""
        # Validate parameters
        k = params.get('k', 5)
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k
        
        distance_metric = params.get('distance_metric', 'euclidean')
        if distance_metric not in ['euclidean', 'manhattan', 'minkowski', 'cosine']:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        self.distance_metric_name = distance_metric
        
        self.p = params.get('p', 2)
        
        # Recreate distance function with p parameter
        try:
            if distance_metric == 'minkowski':
                from functools import partial
                base_metric = get_distance_metric(distance_metric)
                self.distance_metric_func = partial(base_metric, p=self.p)
            else:
                self.distance_metric_func = get_distance_metric(distance_metric)
        except Exception as e:
            raise ValueError(f"Failed to create distance function: {e}")
        
        if params.get('trained', False):
            X_train_data = params.get('X_train')
            y_train_data = params.get('y_train')
            
            if X_train_data is not None and y_train_data is not None:
                # Preserve dtype if specified
                x_dtype = params.get('X_train_dtype', 'float64')
                y_dtype = params.get('y_train_dtype', None)
                
                self.X_train = np.array(X_train_data, dtype=x_dtype)
                self.y_train = np.array(y_train_data, dtype=y_dtype)
                
                # Set n_features_in_ from params or calculate from X_train
                if 'n_features_in_' in params:
                    self.n_features_in_ = params['n_features_in_']
                else:
                    self.n_features_in_ = self.X_train.shape[1]
                
                # Validate loaded data
                if not self.is_trained:
                    raise ValueError("Failed to set training data")
        else:
            self.X_train = None
            self.y_train = None
            self.n_features_in_ = None
    
    def fit(self, X, y):
        """
        Store training data.
        
        Args:
            X: Training features (numpy array or list of lists)
            y: Training labels/values (numpy array or list)
            
        Returns:
            self: For method chaining
            
        Raises:
            ValueError: If inputs have incompatible shapes, empty dataset, or invalid k
        """
        X = np.asarray(X, dtype=float)
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
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.k > len(X):
            raise ValueError(f"k={self.k} cannot be greater than number of samples={len(X)}")
        
        # Check for NaN/Inf values
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values")
        # Check y only for numeric data (to avoid issues with string labels in classification)
        if np.issubdtype(y.dtype, np.number):
            if not np.all(np.isfinite(y)):
                raise ValueError("y contains NaN or infinite values")
        
        self.X_train = X
        self.y_train = y
        self.n_features_in_ = X.shape[1]
        
        return self
    
    def _get_k_nearest_neighbors(self, test_point, return_indices=False):
        """
        Find k nearest neighbors for a test point.
        
        Args:
            test_point: Point to classify (list or numpy array)
            return_indices: If True, returns tuples of (distance, label, index)
            
        Returns:
            List of (distance, label) tuples for k nearest neighbors,
            or (distance, label, index) tuples if return_indices=True,
            sorted by distance in ascending order (closest first).
            
        Note:
            The returned list is guaranteed to be sorted by distance, which is
            essential for tie-breaking in classification.
        """
        if not self.is_trained:
            raise ValueError("Model must be fitted before prediction")
        
        test_point = np.array(test_point)
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
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
                # Vectorized Minkowski distance
                distances_all = np.sum(np.abs(self.X_train - test_point) ** self.p, axis=1) ** (1.0 / self.p)
            
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
            
            if return_indices:
                k_nearest = list(zip(k_distances, k_labels, k_indices_sorted))
            else:
                k_nearest = list(zip(k_distances, k_labels))
        else:
            # Fallback to iterative method for unsupported metrics (e.g., cosine)
            # Calculate distances to all training points
            distances = []
            for i, train_point in enumerate(self.X_train):
                dist = self.distance_metric_func(test_point, train_point)
                distances.append((dist, self.y_train[i], i))
            
            # Sort by distance and get k nearest neighbors
            distances.sort(key=lambda x: x[0])
            if return_indices:
                k_nearest = distances[:k]
            else:
                k_nearest = [(dist, label) for dist, label, _ in distances[:k]]
        
        return k_nearest
    
    def _get_k_nearest_indices(self, test_point):
        """Get indices of k nearest neighbors for visualization."""
        k_nearest = self._get_k_nearest_neighbors(test_point, return_indices=True)
        if not k_nearest:
            return np.array([])
        
        # Extract indices from the result
        k_indices = [idx for _, _, idx in k_nearest]
        
        return np.array(k_indices)
    
    def predict(self, X_test):
        """
        Predict labels/values for test data.
        
        Args:
            X_test: Test features (numpy array or list of lists)
            
        Returns:
            numpy array: Predicted labels/values
            
        Note:
            For large test sets, this method uses vectorized operations when possible,
            but may fall back to iterative approach for unsupported metrics.
        """
        if not self.is_trained:
            raise ValueError("Model must be fitted before prediction")
            
        X_test = np.asarray(X_test, dtype=float)
        
        # Handle single test point
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        # Check for NaN/Inf values
        if not np.all(np.isfinite(X_test)):
            raise ValueError("X_test contains NaN or infinite values")
            
        # Check feature dimensions
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_test.shape[1]}")
            
        # Vectorized prediction for supported metrics
        if self.distance_metric_name in ['euclidean', 'manhattan', 'minkowski']:
            predictions = self._predict_vectorized(X_test)
        else:
            # Fallback to iterative method for unsupported metrics
            predictions = []
            for test_point in X_test:
                k_nearest = self._get_k_nearest_neighbors(test_point)
                # Extract distances and labels from k_nearest tuples
                distances = np.array([dist for dist, _ in k_nearest])
                labels = np.array([label for _, label in k_nearest])
                pred = self._make_prediction_from_neighbors(distances, labels)
                predictions.append(pred)
            predictions = np.array(predictions)
        
        return predictions
    
    def _predict_vectorized(self, X_test, batch_size=1000):
        """Vectorized prediction for supported distance metrics."""
        n_test = X_test.shape[0]
        k = min(self.k, len(self.X_train))
        
        # Process in batches to avoid memory issues
        predictions = []
        
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            X_batch = X_test[start:end]
            batch_size_current = X_batch.shape[0]
            
            # Calculate distance matrix (batch_size, n_train)
            if self.distance_metric_name == 'euclidean':
                distances_all = np.sqrt(np.sum((self.X_train - X_batch[:, np.newaxis]) ** 2, axis=2))
            elif self.distance_metric_name == 'manhattan':
                distances_all = np.sum(np.abs(self.X_train - X_batch[:, np.newaxis]), axis=2)
            elif self.distance_metric_name == 'minkowski':
                distances_all = np.sum(np.abs(self.X_train - X_batch[:, np.newaxis]) ** self.p, axis=2) ** (1.0 / self.p)
            
            # Get k nearest neighbors for each test point in batch
            if k >= len(self.X_train):
                k_indices = np.argsort(distances_all, axis=1)[:, :k]  # Trim to k elements
            else:
                k_indices = np.argpartition(distances_all, k-1, axis=1)[:, :k]
                # Sort k smallest per row
                row_indices = np.arange(batch_size_current)[:, np.newaxis]
                sorted_indices = np.argsort(distances_all[row_indices, k_indices], axis=1)
                k_indices = k_indices[row_indices, sorted_indices]
            
            # Also get the corresponding distances
            row_indices = np.arange(batch_size_current)[:, np.newaxis]
            k_distances = distances_all[row_indices, k_indices]
            k_labels = self.y_train[k_indices]
            
            # Make predictions using the nearest neighbors with distances
            batch_predictions = []
            for i in range(batch_size_current):
                pred = self._make_prediction_from_neighbors(k_distances[i], k_labels[i])
                batch_predictions.append(pred)
            
            predictions.extend(batch_predictions)
        
        return np.array(predictions)
    
    def _make_prediction_from_neighbors(self, distances, labels):
        """
        Make prediction from k nearest neighbors with distances and labels.
        To be implemented by subclasses.
        
        Args:
            distances: Sorted array of distances (ascending) from k nearest neighbors
            labels: Corresponding labels/values from k nearest neighbors
            
        Returns:
            prediction: Predicted label or value
        """
        raise NotImplementedError("Subclasses must implement _make_prediction_from_neighbors method")