import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from .core import LinearRegressionGD
from .preprocessing import standardize_fit, standardize_apply
from .metrics import mse

def k_fold_split(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    K-Fold Cross-Validation split generator.
    Returns k tuples of (X_train, X_val, y_train, y_val)
    
    Args:
        X: Feature matrix, 2D (n_samples, n_features)
        y: Target values, 1D (n_samples,)
        k: Number of folds (>= 2)
        seed: Random seed for reproducibility
    
    Returns:
        List of k tuples, each containing (X_train, X_val, y_train, y_val)
    
    Raises:
        ValueError: If k < 2, X not 2D, y not 1D, samples mismatch, 
                   n_samples < k, or data contains NaN/inf
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    
    # Validate inputs (same contract as LinearRegressionGD._validate_inputs / metrics._validate_metric_inputs)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D (n_samples,), got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y sample mismatch: {X.shape[0]} vs {y.shape[0]}")
    if X.shape[0] < k:
        raise ValueError(f"n_samples ({X.shape[0]}) must be >= k ({k})")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must not contain NaN or infinity")
    
    n = X.shape[0]
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

def grid_search_regularization(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.05,
    epochs: int = 2000,
    lambda_l1_grid: List[float] = None,
    lambda_l2_grid: List[float] = None,
    k_folds: int = 5,
    seed: int = 42,
    verbose: bool = False,
    use_scaling: bool = True,
    early_stopping: bool = True,
    early_stopping_patience: int = 50
) -> Dict[str, Any]:
    """
    Searches for the best regularization parameters (L1 and L2) over a grid.

    The algorithm iterates over all possible combinations of lambda_l1 and lambda_l2 values, 
    using K-Fold cross-validation to assess the stability of the model.
    
    Args:
        X: Feature matrix, 2D (n_samples, n_features).
        y: Target values, 1D (n_samples,).
        learning_rate: Gradient-descent step size used for every candidate (> 0).
        epochs: Maximum epochs per fold (with early_stopping=True training usually
            stops earlier, see early_stopping_patience).
        lambda_l1_grid: L1 (Lasso) strengths to try; 0.0 disables L1. Defaults to
            [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]. Must be non-empty.
        lambda_l2_grid: L2 (Ridge) strengths to try. Defaults to the same 7 values.
            Must be non-empty.
        k_folds: Number of CV folds (2 <= k_folds <= n_samples, validated by k_fold_split).
        seed: Seed for the index shuffle in k_fold_split (reproducible folds).
        verbose: If True, prints one line per combination, including excluded ones
            (DIVERGED). If False, prints a "Progress: X%" line roughly every 5% of
            the combinations (every combination for grids smaller than 20);
            excluded combinations print nothing in this mode.
        use_scaling: If True, each fold is standardized using mean/std of its own
            training part only (no leakage from the validation part).
        early_stopping: If True (default), uses early stopping in each fold. With
            early_stopping=False, each fold runs the full epochs iterations.
            For the default 7×7 grid and 5 folds this is ≈ 49 × 5 × epochs
            iterations; expect minutes for epochs=10000.
        early_stopping_patience: Epochs without validation-loss improvement before
            a fold stops early (used only when early_stopping=True).

    Returns:
        Dictionary with keys:
          * 'best_lambda_l1', 'best_lambda_l2': best strengths found (values taken
            from the grids, not coerced).
          * 'best_mse': mean CV MSE of the best combination; always finite on return.
          * 'results': accepted combinations only, sorted by 'mean_mse' ascending.
            Each entry is {'l1', 'l2', 'mean_mse', 'std_mse'} (floats);
            'std_mse' is the population std across folds (np.std, ddof=0).
        On success 'results' is never empty and 'best_mse' is always finite.

    Raises:
        ValueError: If either lambda grid is empty, or if k_fold_split rejects the
            data (k < 2, X not 2D, y not 1D, X/y sample mismatch, n_samples < k,
            NaN or infinity in X/y).
        RuntimeError: If every candidate diverges, i.e. every combination produces
            non-finite weights/bias (LinearRegressionGD.is_trained is False) or a
            non-finite CV MSE. The search then has no valid answer: lower the
            learning rate or increase the number of epochs.

    Notes:
        A combination is excluded as soon as one of its folds diverges; it is not
        averaged over the remaining folds. With early_stopping=True a diverged fold
        usually restores its best finite weights, so exclusion means divergence
        happened before any valid epoch. The grid banner is printed unconditionally,
        regardless of `verbose`.
    """
   
    if lambda_l1_grid is None:
        lambda_l1_grid = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    if lambda_l2_grid is None:
        lambda_l2_grid = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    lambda_l1_grid = list(lambda_l1_grid)
    lambda_l2_grid = list(lambda_l2_grid)

    if not lambda_l1_grid or not lambda_l2_grid:
        raise ValueError(
            "lambda_l1_grid and lambda_l2_grid must be non-empty "
            f"(got {len(lambda_l1_grid)} and {len(lambda_l2_grid)} values)"
        )
    
    # Generate k-fold splits
    folds = k_fold_split(X, y, k=k_folds, seed=seed)
    
    results = []
    best_mse = float('inf')
    best_params = {'lambda_l1': 0.0, 'lambda_l2': 0.0}
    
    total_combinations = len(lambda_l1_grid) * len(lambda_l2_grid)
    current_combo = 0
    
    print(f"\n{'='*72}")
    print(f"Grid Search: {total_combinations} combinations × {k_folds}-fold CV")
    print(f"Scaling: {'ON' if use_scaling else 'OFF'}")
    print(f"Early Stopping: {'ON' if early_stopping else 'OFF'}")
    print(f"{'='*72}\n")
    
    for l1 in lambda_l1_grid:
        for l2 in lambda_l2_grid:
            current_combo += 1
            fold_mses = []
            diverged = False
            
            for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(folds):
                
                if use_scaling:
                    X_train_scaled, scaler_mean, scaler_std = standardize_fit(X_train)
                    X_val_scaled = standardize_apply(X_val, scaler_mean, scaler_std)
                else:
                    X_train_scaled = X_train
                    X_val_scaled = X_val
                
                # Create and train model
                model = LinearRegressionGD(
                    learning_rate=learning_rate,
                    epochs=epochs,
                    lambda_l1=l1,
                    lambda_l2=l2
                )
                
                # Train with or without early stopping
                if early_stopping:
                    model.fit_with_early_stopping(
                        X_train_scaled, 
                        y_train,
                        X_val_scaled,
                        y_val,
                        patience=early_stopping_patience
                    )
                else:
                    model.fit(X_train_scaled, y_train)
                
                # Reject candidate if training diverged (non-finite weights/bias)
                if not model.is_trained:
                    diverged = True
                    break
                
                # Evaluate on validation fold
                y_pred = model.predict(X_val_scaled)
                fold_mse = mse(y_val, y_pred)
                fold_mses.append(fold_mse)
            
            # Divergent combinations are excluded from results (UI formats mean_mse
            # with :.6f, so None must not leak through).
            if diverged:
                if verbose:
                    print(f"[{current_combo}/{total_combinations}] L1={l1:.4f} "
                          f"L2={l2:.4f} | DIVERGED (excluded)")
                continue
            
            # Calculate mean and std across folds
            mean_mse = float(np.mean(fold_mses))
            std_mse = float(np.std(fold_mses))
            
            # Finite weights can still overflow the loss (huge but finite w). Such a
            # candidate is unusable: it would leave best_mse = inf on return.
            if not np.isfinite(mean_mse):
                if verbose:
                    print(f"[{current_combo}/{total_combinations}] L1={l1:.4f} "
                          f"L2={l2:.4f} | DIVERGED (non-finite MSE, excluded)")
                continue
            
            result_dict = {
                'l1': float(l1),
                'l2': float(l2),
                'mean_mse': mean_mse,
                'std_mse': std_mse
            }
            results.append(result_dict)
            
            # Update best params
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_params = {'lambda_l1': l1, 'lambda_l2': l2}
            
            # Verbose output
            if verbose:
                status = "BEST" if mean_mse == best_mse else ""
                print(f"[{current_combo}/{total_combinations}] L1={l1:.4f} L2={l2:.4f} | "
                      f"MSE: {mean_mse:.6f} ± {std_mse:.6f} {status}")
            else:
                # Progress bar
                if current_combo % max(1, total_combinations // 20) == 0:
                    progress = (current_combo / total_combinations) * 100
                    print(f"Progress: {progress:.1f}% completed.")
    
    
    if not results:
        raise RuntimeError(
            "All regularization combinations diverged (non-finite weights or MSE). "
            "Try a lower learning rate or more epochs."
        )

    results_sorted = sorted(results, key=lambda x: x['mean_mse'])
    
    return {
        'best_lambda_l1': best_params['lambda_l1'],
        'best_lambda_l2': best_params['lambda_l2'],
        'best_mse': best_mse,
        'results': results_sorted
    }



def auto_tune_learning_rate(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate_grid: Optional[List[float]] = None,
    epochs: int = 10000,
    k_folds: int = 3,
    seed: int = 42,
    use_scaling: bool = True,
    early_stopping_patience: int = 50,
    do_refine: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Auto-tune the learning rate via grid search with K-fold CV.
    
    Two-phase approach:
      Phase 1: Coarse log-scale search
      Phase 2: Fine-grain search around the best Phase 1 value
    
    Returns:
        Dictionary with keys 'best_lr', 'best_mse', 'results', 'phase2_results'.
        'results': Phase 1 search results (list of dicts), 
                   including candidates with mean_mse=None for divergence.
        'phase2_results': Phase 2 results (list of dicts), or None if do_refine=False
                         or best_lr <= 0 (indicating no valid candidate in Phase 1).
    
    Notes:
        This function excludes candidates that produce non-finite weights (catastrophic 
        divergence); candidates that converge poorly are still included with their 
        (large) MSE. Phase 2 uses a factor of 2.0 to build a refined grid around 
        the best Phase 1 learning rate.
    """
    if learning_rate_grid is None:
        learning_rate_grid = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    
    if verbose:
        print("Auto-tuning learning rate...")
        print("=" * 72)
        print(f"Grid: {learning_rate_grid}")
        print(f"Using {k_folds}-fold CV, epochs={epochs}")
        if use_scaling:
            print("Scaling inside fold: ON")
        else:
            print("Data already scaled externally")
        print("=" * 72)
    
    # Phase 1
    phase1_results = _evaluate_lr_grid(
        X, y, learning_rate_grid, epochs, k_folds, seed, use_scaling,
        early_stopping_patience, verbose
    )
    
    valid_results = [r for r in phase1_results if r['mean_mse'] is not None]
    if not valid_results:
        raise RuntimeError("All learning rates diverged! Cannot auto-tune.")
    
    valid_results.sort(key=lambda x: x['mean_mse'])
    best_lr = valid_results[0]['lr']
    best_mse = valid_results[0]['mean_mse']
    
    phase2_results = None
    
    # Phase 2: refine
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
                X, y, refine_grid, epochs, k_folds, seed, use_scaling,
                early_stopping_patience, verbose
            )
            
            valid_p2 = [r for r in phase2_results if r['mean_mse'] is not None]
            if valid_p2:
                valid_p2.sort(key=lambda x: x['mean_mse'])
                if valid_p2[0]['mean_mse'] < best_mse:
                    best_lr = valid_p2[0]['lr']
                    best_mse = valid_p2[0]['mean_mse']
    
    if verbose:
        print("=" * 72)
        print(f"Best learning rate: {best_lr}")
        print(f"Best validation MSE: {best_mse:.6e}")
        print("=" * 72)
    
    return {
        'best_lr': best_lr,
        'best_mse': best_mse,
        'results': phase1_results,
        'phase2_results': phase2_results,
    }


def _evaluate_lr_grid(
    X: np.ndarray, y: np.ndarray, lr_grid: List[float],
    epochs: int, k_folds: int, seed: int, use_scaling: bool,
    early_stopping_patience: int, verbose: bool
) -> List[Dict[str, Any]]:
    """
    Evaluate a list of learning rates using K-fold CV.
    Returns list of dicts with 'lr', 'mean_mse', 'std_mse', 'n_folds', 'diverged'.
    """
    folds = k_fold_split(X, y, k=k_folds, seed=seed)
    results = []
    
    for lr in lr_grid:
        fold_mses = []
        diverged = False
        
        for _, (X_train, X_val, y_train, y_val) in enumerate(folds):
            if use_scaling:
                X_train_scaled, scaler_mean, scaler_std = standardize_fit(X_train)
                X_val_scaled = standardize_apply(X_val, scaler_mean, scaler_std)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
            
            model = LinearRegressionGD(
                learning_rate=lr, epochs=epochs,
                lambda_l1=0.0, lambda_l2=0.0
            )
            model.fit_with_early_stopping(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                patience=early_stopping_patience
            )
            
            # Reject candidate if training diverged (non-finite weights/bias)
            if not model.is_trained:
                diverged = True
                break
            
            y_pred = model.predict(X_val_scaled)
            fold_mses.append(mse(y_val, y_pred))
        
        if diverged:
            if verbose:
                print(f"  Testing {lr:.4f} ... DIVERGED")
            results.append({
                'lr': lr, 'mean_mse': None, 'std_mse': None,
                'n_folds': 0, 'diverged': True,
            })
        else:
            mean_mse_val = float(np.mean(fold_mses))
            std_mse_val = float(np.std(fold_mses))
            if verbose:
                print(f"  Testing {lr:.4f} ... MSE: {mean_mse_val:.4e} "
                      f"(epochs: {len(model.loss_history)})")
            results.append({
                'lr': lr, 'mean_mse': mean_mse_val, 'std_mse': std_mse_val,
                'n_folds': len(fold_mses), 'diverged': False,
            })
    
    return results
