"""
K-Nearest Neighbors Classifier implementation.

Implements classification with the standard API pattern supporting:
- Majority voting with tie-breaking by nearest neighbor  
- Probability estimates in standard matrix format (n_samples, n_classes)
- Comprehensive input validation for reliability and sklearn compatibility
"""

import numpy as np
from collections import Counter
from .core import KNNBase


class KNNClassifier(KNNBase):
    """K-Nearest Neighbors Classifier.
    
    Args:
        k: Number of neighbors to consider (default=5)
        distance_metric: Distance metric ('euclidean', 'manhattan', 'minkowski', 'cosine')
        p: Power parameter for Minkowski distance (default=2)
        
    Attributes:
        classes_: Unique class labels from training data
        n_features_in_: Number of features in training data
    """
    
    model_type = "knn_classifier"
    
    def __init__(self, k=5, distance_metric='euclidean', p=2):
        """
        Initialize KNN classifier.
        
        Args:
            k: Number of neighbors to consider
            distance_metric: Distance metric to use
            p: Power parameter for Minkowski distance
        """
        
        super().__init__(k, distance_metric, p)
        self._class_to_idx = None  # Cache for efficiency
        self.classes_ = None
        self.n_features_in_ = None
    
    def fit(self, X_train, y_train):
        """
        Fit the KNN classifier to training data.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
        
        Returns:
            self: Returns the instance itself
        
        Raises:
            ValueError: If X_train and y_train have incompatible shapes
        """
        X_train = self._as_numeric_2d(X_train, "X_train")
        y_train = np.asarray(y_train)
        
        if y_train.ndim != 1:
            raise ValueError(f"y_train must be 1D, got shape {y_train.shape}")
        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"X_train and y_train have incompatible shapes: "
                           f"{X_train.shape[0]} samples vs {y_train.shape[0]}")
        
        self.X_train = X_train
        self.y_train = y_train
        self.classes_ = np.unique(y_train)
        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self.n_features_in_ = X_train.shape[1]
        
        return self
    
    @staticmethod
    def _as_numeric_2d(X, name="X"):
        """
        Convert input to numpy array and validate as numeric 2D data.
        
        Args:
            X: Input data to validate
            name: Variable name for error messages
            
        Returns:
            Validated 2D numpy array
            
        Raises:
            ValueError: If input cannot be converted to numeric 2D array
        """
        try:
            X = np.asarray(X, dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(f"{name} must be numeric: {e}")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape {X.shape}")
        
        if not np.isfinite(X).all():
            raise ValueError(f"{name} contains NaN or infinity")
        
        return X
    
    def _validate_X(self, X):
        """
        Validate test data before prediction.
        
        Args:
            X: Test features to validate
        
        Returns:
            numpy array: Validated 2D array
        
        Raises:
            ValueError: If X has incorrect shape, features, or contains NaN/inf
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X = self._as_numeric_2d(X, "X")
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}")
        
        return X
    
    def _vote(self, k_nearest):
        """
        Perform voting on k nearest neighbors.
        
        Args:
            k_nearest: List of (distance, label) tuples sorted by distance
            
        Returns:
            tuple: (winner_label, counts_dict) where counts_dict shows votes per class
            
        Note:
            k_nearest must be pre-sorted by distance in ascending order.
            This contract is guaranteed by KNNBase._get_k_nearest_neighbors
            which sorts results for consistent tie-breaking.
        """
        # Use pre-sorted k_nearest (trusting the contract from KNNBase)
        # Optional assertion to verify the contract in debug mode
        if __debug__ and len(k_nearest) > 1:
            distances = [d for d, _ in k_nearest]
            assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1)), \
                "k_nearest must be sorted by distance (contract violation)"
        
        labels = [label for _, label in k_nearest]
        label_counts = Counter(labels)
        
        # Find labels with highest count
        max_count = max(label_counts.values())
        candidates = [label for label, cnt in label_counts.items() if cnt == max_count]
        
        # If only one candidate, return it
        if len(candidates) == 1:
            return candidates[0], dict(label_counts)
        
        # Tie-break: select from candidates based on nearest neighbor
        # k_nearest is guaranteed to be sorted by distance (closest first)
        for _, label in k_nearest:
            if label in candidates:
                return label, dict(label_counts)
        
        # Fallback - should never reach here as candidates are from k_nearest
        return candidates[0], dict(label_counts)
    

    
    def _make_prediction(self, k_nearest):
        """
        Make classification prediction using majority voting.
        
        Args:
            k_nearest: List of (distance, label) tuples for k nearest neighbors
            
        Returns:
            Most common label among k nearest neighbors
            
        Note:
            In case of ties, the label from the nearest neighbor is selected.
            k_nearest should be sorted by distance (closest first).
        """
        winner, _ = self._vote(k_nearest)
        return winner
    
    def predict_proba(self, X_test):
        """
        Predict probability estimates for test data.
        
        Args:
            X_test: Test features (numpy array or list of lists)
            
        Returns:
            numpy array: Probability estimates, shape (n_samples, n_classes)
            
        Note:
            Returns probability matrix in standard format (n_samples, n_classes).
            Use self.classes_ to map columns to class labels.
        """
        X_test = self._validate_X(X_test)
        n_samples = X_test.shape[0]
        
        if self.classes_ is None or len(self.classes_) == 0:
            raise ValueError("Model not fitted or no classes found")
        
        if n_samples == 0:
            return np.empty((0, len(self.classes_)))
        
        # Initialize probability matrix
        proba = np.zeros((n_samples, len(self.classes_)))
        class_to_idx = self._class_to_idx
        
        for i, test_point in enumerate(X_test):
            k_nearest = self._get_k_nearest_neighbors(test_point)
            labels = [label for _, label in k_nearest]
            
            if not labels:  # No neighbors found
                # Uniform distribution if no neighbors
                proba[i] = 1.0 / len(self.classes_)
                continue
            
            # Calculate frequency-based probabilities
            label_counts = Counter(labels)
            total_neighbors = len(labels)
            
            for label, count in label_counts.items():
                proba[i, class_to_idx[label]] = count / total_neighbors
        
        return proba
    
    def predict(self, X_test):
        """
        Predict labels for test data.
        
        Args:
            X_test: Test features (numpy array or list of lists)
            
        Returns:
            numpy array: Predicted labels, shape (n_samples,)
        """
        X_test = self._validate_X(X_test)
        
        predictions = []
        for test_point in X_test:
            k_nearest = self._get_k_nearest_neighbors(test_point)
            pred = self._make_prediction(k_nearest)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def score(self, X_test, y_test):
        """
        Return the mean accuracy on the given test data and labels.
        
        Args:
            X_test: Test features
            y_test: True labels for test data
            
        Returns:
            float: Mean accuracy
            
        Raises:
            ValueError: If lengths of predictions and true labels don't match
        """
        y_pred = self.predict(X_test)
        y_test = np.asarray(y_test)
        
        if y_pred.shape != y_test.shape:
            raise ValueError(f"Shape mismatch: predictions {y_pred.shape}, "
                           f"true labels {y_test.shape}")
        
        if len(y_pred) == 0:
            raise ValueError("Cannot score on empty test set")
        
        return np.mean(y_pred == y_test)