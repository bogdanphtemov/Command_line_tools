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


def k_fold_split(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42, 
                stratified: bool = True) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    K-Fold Cross-Validation split generator.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        k: Number of folds (must be at least 2)
        seed: Random seed for reproducibility
        stratified: Whether to use stratified splitting for classification problems
                  (ensures each fold has representative class distributions)
                  
    Returns:
        List of k tuples of (X_train, X_val, y_train, y_val)
        
    Note:
        For classification problems with stratified=True, ensures each fold
        contains approximately the same percentage of samples of each class.
        For regression or when stratified=False, uses simple random shuffling.
        
        Folds may have slightly different sizes when n_samples is not divisible by k.
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    
    n = X.shape[0]
    if k > n:
        raise ValueError(f"k_folds={k} cannot exceed n_samples={n}")
    
    rng = np.random.default_rng(seed)
    
    # Handle stratified splitting for classification problems
    # Check if y appears to be for classification (categorical data)
    is_classification = (
        y.ndim == 1 and 
        (np.issubdtype(y.dtype, np.integer) or 
         np.issubdtype(y.dtype, np.object_) or 
         np.issubdtype(y.dtype, np.str_))
    )
    
    if stratified and is_classification:
        # Classification: use stratified splitting
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        # Check if stratified splitting is possible
        if len(unique_classes) > 1:
            # Minimum samples per class should allow at least 1 sample per fold
            min_class_count = np.min(class_counts)
            if min_class_count < k:
                warnings.warn(
                    f"Stratified splitting impossible: class with {min_class_count} samples "
                    f"cannot be split into {k} folds. Using random splitting instead.",
                    UserWarning
                )
                stratified = False
    
    idx = np.arange(n)
    
    if stratified and is_classification and len(unique_classes) > 1:
        # Stratified splitting: create balanced fold assignments per class
        fold_assignments = np.zeros(n, dtype=int)
        
        for class_label in unique_classes:
            class_indices = idx[y == class_label]
            class_size = len(class_indices)
            
            # Shuffle class indices
            rng.shuffle(class_indices)
            
            # Assign class samples evenly to folds
            for i, sample_idx in enumerate(class_indices):
                fold_assignments[sample_idx] = i % k
        
        fold_indices = []
        for fold in range(k):
            fold_indices.append(idx[fold_assignments == fold])
    else:
        # Random splitting
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
    use_scale: bool,
    error_on_fail: bool = False
) -> Tuple[float, float, List[float], int]:
    """
    Run cross-validation for a single hyperparameter configuration.
    
    Args:
        folds: List of (X_train, X_val, y_train, y_val) folds
        model_factory: Function that creates a new model instance with no arguments
        model_type: 'classification' or 'regression'
        use_scale: Whether to apply feature scaling
        error_on_fail: Whether to raise errors or handle them gracefully
        
    Returns:
        Tuple of (mean_score, std_score, fold_scores, n_successful_folds)
        
    Note:
        For regression model_type, the mean_score represents -MSE (negative MSE) to 
        maintain the "higher is better" convention used in optimization algorithms.
        To get the actual MSE, use -mean_score.
        
        Each fold uses separate scaler fitting on X_train for that fold.
        The scaler is not returned or reused across folds.
        
        If a fold fails and error_on_fail=False, that fold is skipped and n_successful_folds
        will be less than the total number of folds.
    """
    fold_scores = []
    n_successful_folds = 0
    
    for i, (X_train, X_val, y_train, y_val) in enumerate(folds):
        try:
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
            
            # Ensure model trained successfully (check if it has training data)
            if model.X_train is None:
                raise ValueError("Model failed to train properly")
            
            # Evaluate on validation set
            metrics = evaluate_knn_model(model, X_val_scaled, y_val, model_type)
            
            # Convert to scoring metric (higher is better for both)
            if model_type == 'classification':
                fold_scores.append(metrics['accuracy'])
            else:
                # Use negative MSE to maintain "higher is better" convention
                fold_scores.append(-metrics['mse'])  # Negative MSE for maximization
            
            n_successful_folds += 1
            
        except Exception as e:
            if error_on_fail:
                raise
            warnings.warn(
                f"Fold {i+1} failed: {str(e)}. Skipping this fold.",
                UserWarning
            )
    
    if n_successful_folds == 0:
        warnings.warn(
            "Cross-validation failed for all folds. Returning invalid scores.",
            UserWarning
        )
        return float('-inf'), float('inf'), [], 0
    
    mean_score = np.mean(fold_scores) if fold_scores else float('-inf')
    std_score = np.std(fold_scores) if len(fold_scores) > 1 else float('inf')
    
    return mean_score, std_score, fold_scores, n_successful_folds


def grid_search(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, List[Any]],
    model_type: str,
    k_folds: int = 5,
    use_scale: bool = True,
    stratified: bool = True,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform grid search for hyperparameter tuning.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        param_grid: Dictionary of parameter names mapped to lists of values to try
        model_type: 'classification' or 'regression'
        k_folds: Number of cross-validation folds
        use_scale: Whether to apply feature scaling
        stratified: Whether to use stratified splitting for classification
        seed: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        Dictionary with best parameters, best score, and full search results
    """
    from itertools import product
    
    # Validate model_type
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Invalid model_type: {model_type}. Must be one of {list(MODEL_CLASSES.keys())}")
    
    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(product(*param_grid.values()))
    
    if verbose:
        print(f"Grid search with {len(param_values)} parameter combinations and {k_folds}-fold CV")
    
    # Generate folds once for consistency
    folds = k_fold_split(X, y, k=k_folds, seed=seed, stratified=stratified)
    
    best_score = float('-inf')
    best_params = None
    best_results = None
    all_results = []
    
    for param_set in param_values:
        params = dict(zip(param_names, param_set))
        
        # Create model factory with current parameters
        model_class = MODEL_CLASSES[model_type]
        model_factory = partial(model_class, **params)
        
        try:
            mean_score, std_score, fold_scores, n_successful = _cross_validate(
                folds, model_factory, model_type, use_scale, error_on_fail=False
            )
            
            result = {
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'fold_scores': fold_scores,
                'n_successful_folds': n_successful,
                'total_folds': k_folds
            }
            
            all_results.append(result)
            
            # Update best result
            if mean_score > best_score and n_successful > 0:
                best_score = mean_score
                best_params = params.copy()
                best_results = result
                
            if verbose:
                actual_score = best_score if model_type == 'classification' else -best_score
                score_name = 'accuracy' if model_type == 'classification' else 'MSE'
                print(f"Params: {params} -> {score_name}: {actual_score:.4f} ± {std_score:.4f}")
                
        except Exception as e:
            if verbose:
                print(f"Params: {params} -> Failed: {str(e)}")
            
            # Record failed attempt
            all_results.append({
                'params': params,
                'error': str(e),
                'mean_score': float('-inf'),
                'std_score': float('inf'),
                'fold_scores': [],
                'n_successful_folds': 0,
                'total_folds': k_folds
            })
    
    if best_params is None:
        raise ValueError("Grid search failed for all parameter combinations")
    
    # Create and train the best model on full dataset
    model_class = MODEL_CLASSES[model_type]
    best_model = model_class(**best_params)
    
    # Standardize if needed and train on full data
    if use_scale:
        X_scaled, scaler_mean, scaler_std = standardize_fit(X)
        best_model.fit(X_scaled, y)
    else:
        best_model.fit(X, y)
    
    # Final message with best result
    if verbose:
        actual_score = best_score if model_type == 'classification' else -best_score
        score_name = 'accuracy' if model_type == 'classification' else 'MSE'
        print(f"\nBest parameters: {best_params}")
        print(f"Best {score_name}: {actual_score:.4f} ± {best_results['std_score']:.4f}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_results': best_results,
        'all_results': all_results,
        'param_grid': param_grid,
        'model_type': model_type,
        'best_model': best_model,
        'scaler_used': use_scale,
        'scaler_mean': scaler_mean if use_scale else None,
        'scaler_std': scaler_std if use_scale else None
    }


def random_search(
    X: np.ndarray,
    y: np.ndarray,
    param_distributions: Dict[str, List[Any]],
    model_type: str,
    n_iter: int = 10,
    k_folds: int = 5,
    use_scale: bool = True,
    stratified: bool = True,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform random search for hyperparameter tuning.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        param_distributions: Dictionary of parameter names mapped to lists of possible values
        model_type: 'classification' or 'regression'
        n_iter: Number of random parameter combinations to try
        k_folds: Number of cross-validation folds
        use_scale: Whether to apply feature scaling
        stratified: Whether to use stratified splitting for classification
        seed: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        Dictionary with best parameters, best score, and search results
    """
    from itertools import product
    
    # Validate model_type
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Invalid model_type: {model_type}. Must be one of {list(MODEL_CLASSES.keys())}")
    
    rng = np.random.default_rng(seed)
    
    if verbose:
        print(f"Random search with {n_iter} iterations and {k_folds}-fold CV")
    
    # Generate folds once for consistency
    folds = k_fold_split(X, y, k=k_folds, seed=seed, stratified=stratified)
    
    best_score = float('-inf')
    best_params = None
    best_results = None
    all_results = []
    
    for i in range(n_iter):
        # Randomly sample parameters
        params = {}
        for param_name, values in param_distributions.items():
            sampled_param = rng.choice(values)
            # Convert numpy types to Python types for compatibility
            if hasattr(sampled_param, 'item'):
                params[param_name] = sampled_param.item()
            else:
                params[param_name] = sampled_param
        
        # Create model factory with current parameters
        model_class = MODEL_CLASSES[model_type]
        model_factory = partial(model_class, **params)
        
        try:
            mean_score, std_score, fold_scores, n_successful = _cross_validate(
                folds, model_factory, model_type, use_scale, error_on_fail=False
            )
            
            result = {
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'fold_scores': fold_scores,
                'n_successful_folds': n_successful,
                'total_folds': k_folds,
                'iteration': i
            }
            
            all_results.append(result)
            
            # Update best result
            if mean_score > best_score and n_successful > 0:
                best_score = mean_score
                best_params = params.copy()
                best_results = result
                
            if verbose:
                actual_score = mean_score if model_type == 'classification' else -mean_score
                score_name = 'accuracy' if model_type == 'classification' else 'MSE'
                print(f"Iter {i+1}: {params} -> {score_name}: {actual_score:.4f} ± {std_score:.4f}")
                
        except Exception as e:
            if verbose:
                print(f"Iter {i+1}: {params} -> Failed: {str(e)}")
            
            # Record failed attempt
            all_results.append({
                'params': params,
                'error': str(e),
                'mean_score': float('-inf'),
                'std_score': float('inf'),
                'fold_scores': [],
                'n_successful_folds': 0,
                'total_folds': k_folds,
                'iteration': i
            })
    
    if best_params is None:
        raise ValueError("Random search failed for all parameter combinations")
    
    # Create and train the best model on full dataset
    model_class = MODEL_CLASSES[model_type]
    best_model = model_class(**best_params)
    
    # Standardize if needed and train on full data
    if use_scale:
        X_scaled, scaler_mean, scaler_std = standardize_fit(X)
        best_model.fit(X_scaled, y)
    else:
        best_model.fit(X, y)
    
    # Final message with best result
    if verbose:
        actual_score = best_score if model_type == 'classification' else -best_score
        score_name = 'accuracy' if model_type == 'classification' else 'MSE'
        print(f"\nBest parameters: {best_params}")
        print(f"Best {score_name}: {actual_score:.4f} ± {best_results['std_score']:.4f}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_results': best_results,
        'all_results': all_results,
        'n_iter': n_iter,
        'model_type': model_type,
        'best_model': best_model,
        'scaler_used': use_scale,
        'scaler_mean': scaler_mean if use_scale else None,
        'scaler_std': scaler_std if use_scale else None
    }


def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    search_method: str = 'grid',
    param_config: Union[Dict[str, List[Any]], Dict[str, List[Any]]] = None,
    k_folds: int = 5,
    use_scale: bool = True,
    stratified: bool = True,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    High-level hyperparameter tuning function.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        model_type: 'classification' or 'regression'
        search_method: 'grid' or 'random'
        param_config: Parameter configuration (grid or distribution)
        k_folds: Number of cross-validation folds
        use_scale: Whether to apply feature scaling
        stratified: Whether to use stratified splitting for classification
        seed: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        Dictionary with tuning results
    """
    # Default parameter configurations
    if param_config is None:
        if model_type == 'classification':
            param_config = {
                'k': [3, 5, 7, 9, 11],
                'distance_metric': ['euclidean', 'manhattan', 'minkowski'],
            }
        else:  # regression
            param_config = {
                'k': [3, 5, 7, 9, 11],
                'distance_metric': ['euclidean', 'manhattan', 'minkowski'],
                'weighted': [False, True]
            }
    
    if search_method == 'grid':
        return grid_search(
            X, y, param_config, model_type, k_folds, use_scale, stratified, seed, verbose
        )
    elif search_method == 'random':
        return random_search(
            X, y, param_config, model_type, 10, k_folds, use_scale, stratified, seed, verbose
        )
    else:
        raise ValueError(f"Unknown search method: {search_method}. Use 'grid' or 'random'")


# Update __all__ to include new functions
__all__ = [
    'k_fold_split',
    '_cross_validate',
    'grid_search',
    'random_search',
    'tune_hyperparameters'
]