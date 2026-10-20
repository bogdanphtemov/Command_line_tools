"""
Hyperparameter tuning utilities for K-Nearest Neighbors.
"""

import numpy as np
import warnings
from typing import List, Dict, Any, Tuple, Optional, Callable, Union, Type
from functools import partial
from .classifier import KNNClassifier
from .regressor import KNNRegressor
from .metrics import evaluate_knn_model
from .preprocessing import standardize_fit, standardize_apply

# Type alias for model factory
ModelFactory = Callable[[], Union[KNNClassifier, KNNRegressor]]

# Model classes mapping for cleaner code
MODEL_CLASSES = {
    'classification': KNNClassifier,
    'regression': KNNRegressor
}


def k_fold_split(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    K-Fold Cross-Validation split generator.
    Returns k tuples of (X_train, X_val, y_train, y_val)
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    
    n = X.shape[0]
    if k > n:
        raise ValueError(f"k_folds={k} cannot exceed n_samples={n}")
    
    rng = np.random.default_rng(seed)  
    
    # Shuffle indices
    idx = np.arange(n)
    rng.shuffle(idx)
    
    fold_indices = np.array_split(idx, k)
    
    folds = []
    
    for fold_idx in range(k):
        val_idx = fold_indices[fold_idx]
        train_idx = np.concatenate([fold_indices[i] for i in range(k) if i != fold_idx])
        
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        folds.append((X_train, X_val, y_train, y_val))
    
    return folds


def _cross_validate(
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    model_factory: ModelFactory,
    model_type: str,
    use_scale: bool
) -> Tuple[float, float, List[float]]:
    """
    Run cross-validation for a single hyperparameter configuration.
    
    Args:
        folds: List of (X_train, X_val, y_train, y_val) folds
        model_factory: Function that creates a new model instance with no arguments
        model_type: 'classification' or 'regression'
        use_scale: Whether to apply feature scaling
        
    Returns:
        Tuple of (mean_score, std_score, fold_scores)
        
    Note:
        Each fold uses separate scaler fitting on X_train for that fold.
        The scaler is not returned or reused across folds.
    """
    fold_scores = []
    
    for X_train, X_val, y_train, y_val in folds:
        # Apply scaling if enabled
        if use_scale:
            X_train_scaled, scaler_mean, scaler_std = standardize_fit(X_train)
            X_val_scaled = standardize_apply(X_val, scaler_mean, scaler_std)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # Create and train model
        model = model_factory()
        model.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set
        metrics = evaluate_knn_model(model, X_val_scaled, y_val, model_type)
        
        # Convert to scoring metric (higher is better for both)
        if model_type == 'classification':
            fold_scores.append(metrics['accuracy'])
        else:
            fold_scores.append(-metrics['mse'])  # Negative MSE for maximization
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    return mean_score, std_score, fold_scores