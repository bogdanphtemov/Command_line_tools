"""
Hyperparameter tuning utilities for Logistic Regression.

Provides tools for:
    - Grid search over hyperparameter space
    - Cross-validation
    - Parameter optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Any
def k_fold_split(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    K-Fold Cross-Validation split generator.
    Returns k tuples of (X_train, X_val, y_train, y_val)
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    
    fold_indices = np.array_split(idx, k)
    folds = []
    
    for fold_idx in range(k):
        val_idx = fold_indices[fold_idx]
        train_idx = np.concatenate([fold_indices[i] for i in range(k) if i != fold_idx])
        folds.append((X[train_idx], X[val_idx], y[train_idx], y[val_idx]))
    
    return folds
from itertools import product

from .core import LogisticRegressionGD
from .preprocessing import train_test_split, standardize_fit, standardize_apply
from .metrics import accuracy, precision, recall, f1_score, log_loss


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
def auto_tune_learning_rate(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate_grid = None,
    max_epochs: int = 10000,
    k_folds: int = 3,
    seed: int = 42,
    use_scaling: bool = True,
    early_stopping_patience: int = 50,
    do_refine: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Auto-tune the learning rate via grid search with K-fold CV.
    Uses log loss for evaluation (appropriate for logistic regression).
    
    Two-phase: Phase 1 coarse log-scale, Phase 2 fine-grain refinement.
    Detects divergence (NaN/inf) and excludes those candidates.
    """
    if learning_rate_grid is None:
        learning_rate_grid = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    
    if verbose:
        print("Auto-tuning learning rate...")
        print("=" * 72)
        print(f"Grid: {learning_rate_grid}")
        print(f"Using {k_folds}-fold CV, max_epochs={max_epochs}")
        if use_scaling:
            print("Scaling inside fold: ON")
        else:
            print("Data already scaled externally")
        print("=" * 72)
    
    phase1_results = _evaluate_lr_grid(
        X, y, learning_rate_grid, max_epochs, k_folds, seed,
        use_scaling, early_stopping_patience, verbose
    )
    
    valid_results = [r for r in phase1_results if r['mean_log_loss'] is not None]
    if not valid_results:
        raise RuntimeError("All learning rates diverged! Cannot auto-tune.")
    
    valid_results.sort(key=lambda x: x['mean_log_loss'])
    best_lr = valid_results[0]['lr']
    best_loss = valid_results[0]['mean_log_loss']
    
    phase2_results = None
    
    if do_refine and best_lr > 0:
        factor = 2.0
        refine_grid = [
            best_lr / factor,
            best_lr / (factor ** 0.5),
            best_lr,
            best_lr * (factor ** 0.5),
            best_lr * factor,
        ]
        refine_grid = sorted(set(round(lr, 10) for lr in refine_grid))
        refine_grid = [lr for lr in refine_grid if 1e-8 <= lr <= 1.0]
        
        if len(refine_grid) > 1:
            if verbose:
                print("\nRefining search around best LR...")
                print(f"Phase 2 grid: {refine_grid}")
                print("-" * 72)
            
            phase2_results = _evaluate_lr_grid(
                X, y, refine_grid, max_epochs, k_folds, seed,
                use_scaling, early_stopping_patience, verbose
            )
            
            valid_p2 = [r for r in phase2_results if r['mean_log_loss'] is not None]
            if valid_p2:
                valid_p2.sort(key=lambda x: x['mean_log_loss'])
                if valid_p2[0]['mean_log_loss'] < best_loss:
                    best_lr = valid_p2[0]['lr']
                    best_loss = valid_p2[0]['mean_log_loss']
    
    if verbose:
        print("=" * 72)
        print(f"Best learning rate: {best_lr}")
        print(f"Best validation Log Loss: {best_loss:.6e}")
        print("=" * 72)
    
    return {
        'best_lr': best_lr,
        'best_log_loss': best_loss,
        'results': phase1_results,
        'phase2_results': phase2_results,
    }


def _evaluate_lr_grid(
    X: np.ndarray, y: np.ndarray, lr_grid: list,
    max_epochs: int, k_folds: int, seed: int, use_scaling: bool,
    early_stopping_patience: int, verbose: bool
) -> list:
    """
    Evaluate a list of learning rates using K-fold CV and log loss.
    """
    from .core import LogisticRegressionGD
    from .preprocessing import standardize_fit, standardize_apply
    
    folds = k_fold_split(X, y, k=k_folds, seed=seed)
    results = []
    
    for lr in lr_grid:
        fold_losses = []
        diverged = False
        
        for _, (X_train, X_val, y_train, y_val) in enumerate(folds):
            if use_scaling:
                X_train_scaled, scaler_mean, scaler_std = standardize_fit(X_train)
                X_val_scaled = standardize_apply(X_val, scaler_mean, scaler_std)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
            
            model = LogisticRegressionGD(
                learning_rate=lr, epochs=max_epochs,
                lambda_l2=0.0, threshold=0.5
            )
            model.fit_with_early_stopping(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                patience=early_stopping_patience
            )
            
            if model.w is None or not np.all(np.isfinite(model.w)):
                diverged = True
                break
            
            y_prob = model.predict_proba(X_val_scaled)
            fold_losses.append(log_loss(y_val, y_prob))
        
        if diverged:
            if verbose:
                print(f"  Testing {lr:.4f} ... DIVERGED")
            results.append({
                'lr': lr, 'mean_log_loss': None, 'std_log_loss': None,
                'n_folds': 0, 'diverged': True,
            })
        else:
            mean_loss = float(np.mean(fold_losses))
            std_loss = float(np.std(fold_losses))
            if verbose:
                print(f"  Testing {lr:.4f} ... Log Loss: {mean_loss:.4e}")
            results.append({
                'lr': lr, 'mean_log_loss': mean_loss, 'std_log_loss': std_loss,
                'n_folds': len(fold_losses), 'diverged': False,
            })
    
    return results
def auto_tune_l2(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    max_epochs: int = 10000,
    k_folds: int = 3,
    seed: int = 42,
    use_scaling: bool = True,
    early_stopping_patience: int = 50,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Auto-tune the L2 regularization strength via K-fold CV with log loss.
    
    Tests L2 values in [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0] using 3-fold CV.
    Returns best lambda_l2 and its validation log loss.
    """
    lambda_l2_grid = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]
    
    if verbose:
        print("\nAuto-tuning L2 regularization...")
        print("=" * 72)
        print(f"Grid: {lambda_l2_grid}")
        print(f"Using {k_folds}-fold CV, max_epochs={max_epochs}")
        print(f"Learning rate fixed at: {learning_rate}")
        print("=" * 72)
    
    from .core import LogisticRegressionGD
    from .preprocessing import standardize_fit, standardize_apply
    
    folds = k_fold_split(X, y, k=k_folds, seed=seed)
    results = []
    
    for l2 in lambda_l2_grid:
        fold_losses = []
        diverged = False
        
        for _, (X_train, X_val, y_train, y_val) in enumerate(folds):
            if use_scaling:
                X_train_scaled, scaler_mean, scaler_std = standardize_fit(X_train)
                X_val_scaled = standardize_apply(X_val, scaler_mean, scaler_std)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
            
            model = LogisticRegressionGD(
                learning_rate=learning_rate, epochs=max_epochs,
                lambda_l2=l2, threshold=0.5
            )
            model.fit_with_early_stopping(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                patience=early_stopping_patience
            )
            
            if model.w is None or not np.all(np.isfinite(model.w)):
                diverged = True
                break
            
            y_prob = model.predict_proba(X_val_scaled)
            fold_losses.append(log_loss(y_val, y_prob))
        
        if diverged:
            if verbose:
                print(f"  Testing L2={l2:.6f} ... DIVERGED")
            results.append({
                'lambda_l2': l2, 'mean_log_loss': None,
                'std_log_loss': None, 'n_folds': 0, 'diverged': True,
            })
        else:
            mean_loss = float(np.mean(fold_losses))
            std_loss = float(np.std(fold_losses))
            if verbose:
                print(f"  Testing L2={l2:.6f} ... Log Loss: {mean_loss:.4e}")
            results.append({
                'lambda_l2': l2, 'mean_log_loss': mean_loss,
                'std_log_loss': std_loss, 'n_folds': len(fold_losses), 'diverged': False,
            })
    
    valid_results = [r for r in results if r['mean_log_loss'] is not None]
    if not valid_results:
        raise RuntimeError("All L2 values diverged! Cannot auto-tune.")
    
    valid_results.sort(key=lambda x: x['mean_log_loss'])
    best_l2 = valid_results[0]['lambda_l2']
    best_loss = valid_results[0]['mean_log_loss']
    
    if verbose:
        print("=" * 72)
        print(f"Best L2: {best_l2}")
        print(f"Best validation Log Loss: {best_loss:.6e}")
        print("=" * 72)
    
    return {
        'best_lambda_l2': best_l2,
        'best_log_loss': best_loss,
        'results': results,
    }