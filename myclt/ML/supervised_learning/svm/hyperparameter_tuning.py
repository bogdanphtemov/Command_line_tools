"""
Hyperparameter tuning utilities for SVM models.

Provides:
    - Grid search with cross-validation
    - Random search with cross-validation

Supported parameter ranges (typical):
    C:        [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
    gamma:    [0.0001, 0.001, 0.01, 0.1, 1.0, 10]
    degree:   [2, 3, 4, 5]
    epsilon:  [0.01, 0.05, 0.1, 0.2, 0.5]
    kernels:  ['linear', 'rbf', 'poly', 'sigmoid']

Example:
    >>> from svm.hyperparameter_tuning import grid_search_cv
    >>> param_grid = {'C': [0.1, 1.0, 10], 'gamma': [0.01, 0.1, 1.0]}
    >>> best_params, best_score = grid_search_cv(X, y, model_class=LinearSVM, param_grid=param_grid)
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Type
from itertools import product

from .core import LinearSVM, KernelSVM, LinearSVR, KernelSVR
from .preprocessing import standardize_fit, standardize_apply
from .metrics import accuracy, multiclass_f1_score, mean_squared_error


def _create_model(model_class: Type, params: Dict[str, Any]):
    """
    Create a model instance with given parameters.

    Filters out irrelevant parameters for the model class.

    Args:
        model_class: LinearSVM, KernelSVM, LinearSVR, or KernelSVR
        params: Hyperparameter dictionary

    Returns:
        Model instance
    """
    # Common params
    valid_params = {}
    for key in ['C', 'learning_rate', 'epochs', 'batch_size']:
        if key in params:
            valid_params[key] = params[key]

    # Kernel params
    if model_class in (KernelSVM, KernelSVR):
        for key in ['kernel', 'gamma', 'degree', 'coef0']:
            if key in params:
                valid_params[key] = params[key]

    # SVR params
    if model_class in (LinearSVR, KernelSVR):
        if 'epsilon' in params:
            valid_params['epsilon'] = params['epsilon']

    return model_class(**valid_params)


def _evaluate_params_cv(X: np.ndarray, y: np.ndarray,
                        model_class: Type, params: Dict[str, Any],
                        cv_folds: int = 5,
                        task: str = 'classifier',
                        verbose: bool = False) -> float:
    """
    Evaluate a single parameter configuration using k-fold CV.

    Uses random permutation of indices for fold assignment, and applies
    stratified sampling for classification tasks to preserve class ratios.

    Args:
        X: Feature matrix
        y: Target vector
        model_class: Model class to evaluate
        params: Hyperparameters
        cv_folds: Number of CV folds
        task: 'classifier' or 'regressor'
        verbose: If True, print fold progress

    Returns:
        Mean score across folds (macro-F1 for classification, -MSE for regression)
    """
    fold_scores = []
    n_samples = len(X)

    if task == 'classifier':
        # Stratified k-fold: preserve class proportions in each fold
        classes = np.unique(y)
        class_indices = {cls: np.where(y == cls)[0] for cls in classes}

        fold_indices = [[] for _ in range(cv_folds)]
        for cls in classes:
            cls_idx = class_indices[cls].copy()
            np.random.shuffle(cls_idx)
            # Distribute class samples across folds
            splits = np.array_split(cls_idx, cv_folds)
            for fold in range(cv_folds):
                fold_indices[fold].extend(splits[fold].tolist())

        for fold in range(cv_folds):
            if verbose:
                print(f"    Fold {fold + 1}/{cv_folds}", end="\r")

            test_idx = np.array(fold_indices[fold], dtype=int)
            train_idx = np.setdiff1d(np.arange(n_samples), test_idx)

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Scale
            X_train_scaled, mean, std = standardize_fit(X_train)
            X_test_scaled = standardize_apply(X_test, mean, std)

            # Train model
            model = _create_model(model_class, params)
            model.fit(X_train_scaled, y_train)

            # Evaluate with macro-averaged F1 (stable across all class distributions)
            y_pred = model.predict(X_test_scaled)
            score = multiclass_f1_score(y_test, y_pred, average='macro')
            fold_scores.append(score)
    else:
        # Random permutation for regression (no stratification needed)
        indices = np.random.permutation(n_samples)
        fold_size = n_samples // cv_folds

        for fold in range(cv_folds):
            if verbose:
                print(f"    Fold {fold + 1}/{cv_folds}", end="\r")

            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < cv_folds - 1 else n_samples

            test_indices = indices[test_start:test_end]
            train_indices = np.concatenate([indices[:test_start], indices[test_end:]])

            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            # Scale
            X_train_scaled, mean, std = standardize_fit(X_train)
            X_test_scaled = standardize_apply(X_test, mean, std)

            # Train model
            model = _create_model(model_class, params)
            model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = model.predict(X_test_scaled)
            score = -mean_squared_error(y_test, y_pred)
            fold_scores.append(score)

    if verbose:
        print()  # Clear the progress line

    return float(np.mean(fold_scores))


def _print_results(best_params: Dict[str, Any], best_score: float,
                   task: str, search_type: str = "SEARCH") -> None:
    """Print formatted search results."""
    metric = 'F1' if task == 'classifier' else '(neg) MSE'
    print(f"{'=' * 70}")
    print(f"Best params: {best_params}")
    print(f"Best {metric}: {best_score:.6f}")
    print(f"{'=' * 70}\n")


def grid_search_cv(X: np.ndarray, y: np.ndarray,
                   model_class: Type,
                   param_grid: Dict[str, List[Any]],
                   cv_folds: int = 5,
                   task: str = 'classifier',
                   verbose: bool = True) -> Tuple[Dict[str, Any], float]:
    """
    Grid search with k-fold cross-validation.

    Tests all parameter combinations and returns best configuration.

    Args:
        X: Feature matrix
        y: Target vector
        model_class: Model class (LinearSVM, KernelSVM, LinearSVR, KernelSVR)
        param_grid: Dictionary of parameter names -> list of values
                    Example: {'C': [0.1, 1.0, 10], 'gamma': [0.01, 0.1, 1.0]}
        cv_folds: Number of CV folds
        task: 'classifier' or 'regressor'
        verbose: Print progress

    Returns:
        Tuple of (best_params, best_score)

    Raises:
        ValueError: If param_grid is empty
    """
    if not param_grid:
        raise ValueError("param_grid cannot be empty")

    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    # Calculate total combinations without materialising all of them
    total_combinations = 1
    for v in param_values:
        total_combinations *= len(v)

    best_score = -float('inf')
    best_params = None

    show_fold_progress = verbose and cv_folds > 5

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"GRID SEARCH: {total_combinations} combinations to test")
        print(f"{'=' * 70}")

    # Use generator instead of list(product(...)) to save memory
    for combo_idx, values in enumerate(product(*param_values), 1):
        params = dict(zip(param_names, values))

        avg_score = _evaluate_params_cv(
            X, y, model_class, params, cv_folds, task,
            verbose=show_fold_progress
        )

        if avg_score > best_score:
            best_score = avg_score
            best_params = params.copy()

        if verbose and combo_idx % max(1, total_combinations // 10 + 1) == 0:
            metric = 'F1' if task == 'classifier' else '(neg) MSE'
            print(f"  Progress: {combo_idx}/{total_combinations} | Best {metric}: {best_score:.6f}")

    if verbose:
        _print_results(best_params, best_score, task, "GRID SEARCH")

    return best_params, best_score


def random_search_cv(X: np.ndarray, y: np.ndarray,
                     model_class: Type,
                     param_distributions: Dict[str, List[Any]],
                     n_iter: int = 10,
                     cv_folds: int = 5,
                     task: str = 'classifier',
                     seed: int = 42,
                     verbose: bool = True) -> Tuple[Dict[str, Any], float]:
    """
    Random search with k-fold cross-validation.

    Randomly samples parameter combinations (faster than grid search).

    Args:
        X: Feature matrix
        y: Target vector
        model_class: Model class (LinearSVM, KernelSVM, LinearSVR, KernelSVR)
        param_distributions: Dictionary of parameter names -> list of values
        n_iter: Number of random combinations to test
        cv_folds: Number of CV folds
        task: 'classifier' or 'regressor'
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        Tuple of (best_params, best_score)

    Raises:
        ValueError: If param_distributions is empty
    """
    if not param_distributions:
        raise ValueError("param_distributions cannot be empty")

    rng = np.random.RandomState(seed)
    best_score = -float('inf')
    best_params = None

    show_fold_progress = verbose and cv_folds > 5

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"RANDOM SEARCH: {n_iter} random combinations to test")
        print(f"{'=' * 70}")

    for iteration in range(n_iter):
        params = {}
        for param_name, values in param_distributions.items():
            params[param_name] = rng.choice(values)

        avg_score = _evaluate_params_cv(
            X, y, model_class, params, cv_folds, task,
            verbose=show_fold_progress
        )

        if avg_score > best_score:
            best_score = avg_score
            best_params = params.copy()

        if verbose and (iteration + 1) % max(1, n_iter // 5 + 1) == 0:
            metric = 'F1' if task == 'classifier' else '(neg) MSE'
            print(f"  Progress: {iteration + 1}/{n_iter} | Best {metric}: {best_score:.6f}")

    if verbose:
        _print_results(best_params, best_score, task, "RANDOM SEARCH")

    return best_params, best_score
