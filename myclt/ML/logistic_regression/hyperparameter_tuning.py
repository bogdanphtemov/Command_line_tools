"""
Hyperparameter tuning utilities for Logistic Regression.

Provides tools for:
    - Grid search over hyperparameter space
    - Cross-validation
    - Parameter optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from itertools import product

from .core import LogisticRegressionGD
from .preprocessing import train_test_split, standardize_fit, standardize_apply
from .metrics import accuracy, precision, recall, f1_score


def _evaluate_params_cv(X: np.ndarray, y: np.ndarray, 
                       params: Dict[str, Any],
                       cv_folds: int = 5) -> float:
    """
    Evaluate a single parameter configuration using k-fold CV.
    
    Args:
        X: Feature matrix
        y: Target vector
        params: Dictionary of hyperparameters for LogisticRegressionGD
        cv_folds: Number of cross-validation folds
    
    Returns:
        Mean F1-score across folds
    """
    fold_scores = []
    n_samples = len(X)
    fold_size = n_samples // cv_folds
    
    for fold in range(cv_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < cv_folds - 1 else n_samples
        
        test_mask = np.zeros(n_samples, dtype=bool)
        test_mask[test_start:test_end] = True
        train_mask = ~test_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        X_train_scaled, mean, std = standardize_fit(X_train)
        X_test_scaled = standardize_apply(X_test, mean, std)
        
        model = LogisticRegressionGD(**params)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        fold_scores.append(f1_score(y_test, y_pred))
    
    return np.mean(fold_scores)


def grid_search_cv(X: np.ndarray, y: np.ndarray, 
                   param_grid: Dict[str, List[Any]],
                   cv_folds: int = 5,
                   verbose: bool = True) -> Tuple[Dict[str, Any], float]:
    """
    Grid search with k-fold cross-validation.
    
    Tests all parameter combinations and returns best configuration.
    
    Args:
        X: Feature matrix
        y: Target vector
        param_grid: Dictionary of parameter names -> list of values
                    Example: {'learning_rate': [0.001, 0.01, 0.1],
                              'epochs': [100, 1000],
                              'lambda_l2': [0.0, 0.01]}
        cv_folds: Number of cross-validation folds
        verbose: Print progress
    
    Returns:
        Tuple of (best_params, best_score)
    
    Example:
        >>> params = {
        ...     'learning_rate': [0.001, 0.01],
        ...     'epochs': [100, 1000],
        ...     'lambda_l2': [0.0, 0.01]
        ... }
        >>> best_params, best_score = grid_search_cv(X, y, params)
        >>> print(f"Best params: {best_params}")
        >>> print(f"Best F1 score: {best_score:.4f}")
    """
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    combinations = list(product(*param_values))
    
    best_score = -1
    best_params = None
    
    if verbose:
        print(f"\nGrid Search: {len(combinations)} combinations to test")
        print("=" * 70)
    
    # Test each combination
    for combo_idx, values in enumerate(combinations, 1):
        params = dict(zip(param_names, values))
        
        # Cross-validation
        avg_score = _evaluate_params_cv(X, y, params, cv_folds)
        
        # Update best if improved
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
        
        if verbose and combo_idx % max(1, len(combinations) // 10) == 0:
            print(f"Progress: {combo_idx}/{len(combinations)} | Best F1: {best_score:.4f}")
    
    if verbose:
        print("=" * 70)
        print(f"Best params found: {best_params}")
        print(f"Best F1 score: {best_score:.4f}\n")
    
    return best_params, best_score


def random_search_cv(X: np.ndarray, y: np.ndarray,
                     param_distributions: Dict[str, List[Any]],
                     n_iter: int = 10,
                     cv_folds: int = 5,
                     seed: int = 42,
                     verbose: bool = True) -> Tuple[Dict[str, Any], float]:
    """
    Random search with k-fold cross-validation.
    
    Randomly samples parameter combinations (faster than grid search).
    
    Args:
        X: Feature matrix
        y: Target vector
        param_distributions: Dictionary of parameter names -> list of values
        n_iter: Number of random combinations to test
        cv_folds: Number of cross-validation folds
        seed: Random seed for reproducibility
        verbose: Print progress
    
    Returns:
        Tuple of (best_params, best_score)
    """
    
    rng = np.random.RandomState(seed)
    best_score = -1
    best_params = None
    
    if verbose:
        print(f"\nRandom Search: {n_iter} random combinations to test")
        print("=" * 70)
    
    # Generate random parameter combinations
    for iteration in range(n_iter):
        # Randomly sample one value from each parameter
        params = {}
        for param_name, values in param_distributions.items():
            params[param_name] = rng.choice(values)
        
        # Cross-validation
        avg_score = _evaluate_params_cv(X, y, params, cv_folds)
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
        
        if verbose and (iteration + 1) % max(1, n_iter // 5) == 0:
            print(f"Progress: {iteration + 1}/{n_iter} | Best F1: {best_score:.4f}")
    
    if verbose:
        print("=" * 70)
        print(f"Best params found: {best_params}")
        print(f"Best F1 score: {best_score:.4f}\n")
    
    return best_params, best_score